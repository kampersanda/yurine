use std::path::Path;

use super::{CosineEmbeddingCosts, EmbeddingStore};
use crate::costs::Cost;
use crate::costs::persistence::{MetadataReader, NoTokenCodec};
use crate::errors::{Error, Result};
use crate::persistence::format::{FileKind, PersistedFile, SectionData, SectionKind, write_file};

const METADATA_VERSION: u32 = 1;

impl<T> CosineEmbeddingCosts<T> {
    /// Saves only the constant costs; the embedding store remains independent.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut metadata = Vec::with_capacity(16);
        metadata.extend_from_slice(&METADATA_VERSION.to_le_bytes());
        for cost in [self.deletion, self.insertion, self.missing_substitution] {
            metadata.extend_from_slice(&cost.get().to_le_bytes());
        }
        let sections = [(
            SectionKind::CostMetadata,
            SectionData::Bytes {
                bytes: &metadata,
                element_count: 1,
            },
        )];
        write_file(
            path.as_ref(),
            FileKind::CosineEmbeddingCosts,
            &NoTokenCodec,
            &sections,
        )
    }

    /// Opens constant costs and combines them with a separately opened store.
    pub fn open(path: impl AsRef<Path>, embeddings: EmbeddingStore<T>) -> Result<Self> {
        let file =
            PersistedFile::open(path.as_ref(), FileKind::CosineEmbeddingCosts, &NoTokenCodec)?;
        let mut reader = MetadataReader::new(file.bytes(SectionKind::CostMetadata)?);
        if reader.u32()? != METADATA_VERSION {
            return Err(Error::InvalidFile(
                "unsupported cosine-cost metadata version",
            ));
        }
        let deletion = Cost::new(reader.f32()?)?;
        let insertion = Cost::new(reader.f32()?)?;
        let missing_substitution = Cost::new(reader.f32()?)?;
        reader.finish()?;
        Ok(Self {
            embeddings,
            deletion,
            insertion,
            missing_substitution,
        })
    }
}

impl<T> CosineEmbeddingCosts<T>
where
    T: Eq + std::hash::Hash,
{
    /// Fully validates the constant costs and every embedding row.
    pub fn verify(&self) -> Result<()> {
        for cost in [self.deletion, self.insertion, self.missing_substitution] {
            Cost::new(cost.get())?;
        }
        self.embeddings.verify()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::num::NonZeroUsize;

    use tempfile::tempdir;

    use super::*;
    use crate::costs::EditCosts;
    use crate::costs::embedding::EmbeddingStoreBuilder;

    #[test]
    fn cosine_costs_round_trip_with_external_store() {
        let mut builder = EmbeddingStoreBuilder::new(NonZeroUsize::new(2).unwrap());
        builder.insert('a', [1.0, 0.0]).unwrap();
        let costs = CosineEmbeddingCosts::new(builder.build())
            .with_deletion_cost(Cost::new_const(0.25))
            .with_insertion_cost(Cost::new_const(0.5))
            .with_missing_substitution_cost(Cost::new_const(0.75));
        let directory = tempdir().unwrap();
        let path = directory.path().join("cosine.yurine");
        costs.save(&path).unwrap();

        let opened = CosineEmbeddingCosts::open(path, costs.embeddings.clone()).unwrap();
        assert_eq!(opened.deletion(&'a'), Cost::new_const(0.25));
        assert_eq!(opened.insertion(&'a'), Cost::new_const(0.5));
        assert_eq!(opened.substitution(&'a', &'x'), Cost::new_const(0.75));
        opened.verify().unwrap();
    }

    #[test]
    fn corrupt_cost_is_rejected_without_panicking() {
        let builder = EmbeddingStoreBuilder::<char>::new(NonZeroUsize::new(2).unwrap());
        let costs = CosineEmbeddingCosts::new(builder.build());
        let directory = tempdir().unwrap();
        let path = directory.path().join("corrupt.yurine");
        costs.save(&path).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let metadata = section_offset(&bytes, SectionKind::CostMetadata as u32);
        bytes[metadata + 4..metadata + 8].copy_from_slice(&f32::NAN.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let result = CosineEmbeddingCosts::open(path, costs.embeddings.clone());
        assert!(matches!(result, Err(Error::InvalidCost(value)) if value.is_nan()));
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
