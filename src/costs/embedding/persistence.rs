use hashbrown::HashMap;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::path::Path;

use super::EmbeddingStore;
use crate::costs::persistence::MetadataReader;
use crate::errors::{Error, Result};
use crate::persistence::TokenCodec;
use crate::persistence::format::{FileKind, PersistedFile, SectionData, SectionKind, write_file};
use crate::storage::Storage;

const METADATA_VERSION: u32 = 1;

impl<T> EmbeddingStore<T>
where
    T: Eq + Hash,
{
    /// Saves this store independently from a search engine or cost policy.
    ///
    /// Rows are streamed from their existing owned or mapped backing in
    /// deterministic token order; saving does not copy the vector matrix into
    /// a second heap allocation.
    pub fn save_with<C: TokenCodec<T>>(&self, path: impl AsRef<Path>, codec: &C) -> Result<()> {
        self.verify()?;

        let mut rows = Vec::with_capacity(self.len());
        for (token, &index) in &self.embedding_indices {
            let mut encoded = Vec::new();
            codec.encode(token, &mut encoded)?;
            if codec.decode(&encoded)? != *token {
                return Err(Error::InvalidTokenEncoding(
                    "token codec does not round-trip the embedding index".into(),
                ));
            }
            rows.push((encoded, index as usize));
        }
        rows.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        if rows.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(Error::InvalidTokenEncoding(
                "token codec maps distinct embedding tokens to the same bytes".into(),
            ));
        }

        let mut token_offsets = Vec::with_capacity(rows.len() + 1);
        let mut token_blob = Vec::new();
        let mut row_indices = Vec::with_capacity(rows.len());
        token_offsets.push(0);
        for (encoded, index) in rows {
            token_blob.extend_from_slice(&encoded);
            token_offsets.push(token_blob.len() as u64);
            row_indices.push(index);
        }

        let mut metadata = Vec::with_capacity(12);
        metadata.extend_from_slice(&METADATA_VERSION.to_le_bytes());
        metadata.extend_from_slice(&(self.dimension.get() as u64).to_le_bytes());
        let sections = [
            (
                SectionKind::EmbeddingTokenOffsets,
                SectionData::U64(&token_offsets),
            ),
            (
                SectionKind::EmbeddingTokenBlob,
                SectionData::Bytes {
                    bytes: &token_blob,
                    element_count: self.len() as u64,
                },
            ),
            (
                SectionKind::Embeddings,
                SectionData::F32Rows {
                    values: &self.embeddings,
                    row_indices: &row_indices,
                    row_len: self.dimension.get(),
                },
            ),
            (
                SectionKind::CostMetadata,
                SectionData::Bytes {
                    bytes: &metadata,
                    element_count: 1,
                },
            ),
        ];
        write_file(path.as_ref(), FileKind::EmbeddingStore, codec, &sections)
    }

    /// Opens a store while keeping its vector matrix memory-mapped.
    ///
    /// The published snapshot must not be modified or truncated while this
    /// store is alive. Row validation is cached on first access. Call
    /// [`EmbeddingStore::verify`] after opening an untrusted snapshot to reject
    /// corruption instead of treating an invalid row as a missing embedding.
    pub fn open_with<C: TokenCodec<T>>(path: impl AsRef<Path>, codec: &C) -> Result<Self> {
        let file = PersistedFile::open(path.as_ref(), FileKind::EmbeddingStore, codec)?;
        let mut metadata = MetadataReader::new(file.bytes(SectionKind::CostMetadata)?);
        if metadata.u32()? != METADATA_VERSION {
            return Err(Error::InvalidFile("unsupported embedding metadata version"));
        }
        let dimension = usize::try_from(metadata.u64()?)
            .map_err(|_| Error::PlatformSizeOverflow)
            .and_then(|dimension| {
                NonZeroUsize::new(dimension)
                    .ok_or(Error::InvalidFile("embedding dimension is zero"))
            })?;
        metadata.finish()?;

        let offsets = file.mapped_slice::<u64>(SectionKind::EmbeddingTokenOffsets)?;
        let blob = file.bytes(SectionKind::EmbeddingTokenBlob)?;
        let embeddings = file.mapped_slice::<f32>(SectionKind::Embeddings)?;
        let row_count = offsets.len().saturating_sub(1);
        let expected_values = row_count
            .checked_mul(dimension.get())
            .ok_or(Error::InvalidFile("embedding matrix length overflows"))?;
        if embeddings.len() != expected_values {
            return Err(Error::InvalidFile(
                "embedding matrix length does not match its shape",
            ));
        }

        let mut embedding_indices = HashMap::with_capacity(row_count);
        for (index, bounds) in offsets.windows(2).enumerate() {
            let start = usize::try_from(bounds[0]).map_err(|_| Error::PlatformSizeOverflow)?;
            let end = usize::try_from(bounds[1]).map_err(|_| Error::PlatformSizeOverflow)?;
            let token = codec.decode(&blob[start..end])?;
            let index = u32::try_from(index).map_err(|_| Error::EmbeddingIndexOverflow)?;
            if embedding_indices.insert(token, index).is_some() {
                return Err(Error::InvalidFile(
                    "decoded embedding index contains duplicate tokens",
                ));
            }
        }

        Ok(Self {
            dimension,
            embedding_indices,
            embeddings: Storage::Mapped(embeddings),
            validated_rows: Some(
                std::iter::repeat_with(std::sync::OnceLock::new)
                    .take(row_count)
                    .collect(),
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::num::NonZeroUsize;

    use tempfile::tempdir;

    use super::*;
    use crate::costs::embedding::EmbeddingStoreBuilder;
    use crate::persistence::CharCodec;

    fn store() -> EmbeddingStore<char> {
        let mut builder = EmbeddingStoreBuilder::new(NonZeroUsize::new(2).unwrap());
        builder.insert('b', [0.0, 1.0]).unwrap();
        builder.insert('a', [1.0, 0.0]).unwrap();
        builder.build()
    }

    fn reordered_store() -> EmbeddingStore<char> {
        let mut builder = EmbeddingStoreBuilder::new(NonZeroUsize::new(2).unwrap());
        builder.insert('a', [1.0, 0.0]).unwrap();
        builder.insert('b', [0.0, 1.0]).unwrap();
        builder.build()
    }

    #[test]
    fn mapped_store_round_trips_and_verifies() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("embeddings.yurine");
        store().save_with(&path, &CharCodec).unwrap();

        let mapped = EmbeddingStore::open_with(path, &CharCodec).unwrap();
        assert_eq!(mapped.get(&'a'), Some([1.0, 0.0].as_slice()));
        assert_eq!(mapped.get(&'b'), Some([0.0, 1.0].as_slice()));
        mapped.verify().unwrap();
    }

    #[test]
    fn empty_store_round_trips() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("empty.yurine");
        let store = EmbeddingStoreBuilder::<char>::new(NonZeroUsize::new(2).unwrap()).build();
        store.save_with(&path, &CharCodec).unwrap();

        let mapped = EmbeddingStore::open_with(path, &CharCodec).unwrap();
        assert!(mapped.is_empty());
        mapped.verify().unwrap();
    }

    #[test]
    fn mapped_store_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>(_: &T) {}

        let directory = tempdir().unwrap();
        let path = directory.path().join("embeddings.yurine");
        store().save_with(&path, &CharCodec).unwrap();
        let mapped = EmbeddingStore::open_with(path, &CharCodec).unwrap();
        assert_send_sync(&mapped);
    }

    #[test]
    fn embedding_files_are_byte_deterministic() {
        let directory = tempdir().unwrap();
        let first = directory.path().join("first.yurine");
        let second = directory.path().join("second.yurine");
        store().save_with(&first, &CharCodec).unwrap();
        reordered_store().save_with(&second, &CharCodec).unwrap();
        assert_eq!(fs::read(first).unwrap(), fs::read(second).unwrap());
    }

    #[test]
    fn corrupt_mapped_row_is_absent_and_verify_reports_it() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("corrupt.yurine");
        store().save_with(&path, &CharCodec).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let offset = section_offset(&bytes, SectionKind::Embeddings as u32);
        bytes[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let mapped = EmbeddingStore::open_with(path, &CharCodec).unwrap();
        assert_eq!(mapped.get(&'a'), None);
        assert!(matches!(
            mapped.verify(),
            Err(Error::InvalidEmbeddingValue { .. })
        ));
    }

    fn section_offset(bytes: &[u8], wanted_kind: u32) -> usize {
        let section_count = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;
        let table = u64::from_le_bytes(bytes[48..56].try_into().unwrap()) as usize;
        (0..section_count)
            .find_map(|index| {
                let entry = table + index * 32;
                let kind = u32::from_le_bytes(bytes[entry..entry + 4].try_into().unwrap());
                (kind == wanted_kind).then(|| {
                    u64::from_le_bytes(bytes[entry + 8..entry + 16].try_into().unwrap()) as usize
                })
            })
            .unwrap()
    }
}
