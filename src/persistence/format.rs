use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::mem::{align_of, size_of};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use zerocopy::byteorder::little_endian::{U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use super::TokenCodec;
use super::storage::{MappedSlice, map_file};
use crate::errors::{Error, Result};

pub(crate) const MAGIC: [u8; 8] = *b"YURINE\0\0";
pub(crate) const FORMAT_VERSION: u32 = 1;
pub(crate) const MAX_CODEC_ID_LEN: usize = 255;
const ENDIAN_MARKER: u32 = 0x0102_0304;

#[repr(C)]
#[derive(Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
struct DiskHeader {
    magic: [u8; 8],
    format_version: U32,
    endian_marker: U32,
    file_kind: U32,
    header_len: U32,
    codec_version: U32,
    section_count: U32,
    codec_offset: U64,
    codec_len: U64,
    section_table_offset: U64,
    file_len: U64,
}

#[repr(C)]
#[derive(Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
struct DiskSectionEntry {
    kind: U32,
    flags: U32,
    offset: U64,
    byte_len: U64,
    element_count: U64,
}

pub(crate) const HEADER_LEN: usize = size_of::<DiskHeader>();
pub(crate) const SECTION_ENTRY_LEN: usize = size_of::<DiskSectionEntry>();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum FileKind {
    SearchEngine = 1,
    EmbeddingStore = 2,
    LevenshteinCosts = 3,
    CustomCosts = 4,
    CosineEmbeddingCosts = 5,
}

impl FileKind {
    fn from_raw(raw: u32) -> Result<Self> {
        match raw {
            1 => Ok(Self::SearchEngine),
            2 => Ok(Self::EmbeddingStore),
            3 => Ok(Self::LevenshteinCosts),
            4 => Ok(Self::CustomCosts),
            5 => Ok(Self::CosineEmbeddingCosts),
            _ => Err(Error::InvalidFile("unknown file kind")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub(crate) enum SectionKind {
    VocabularyTokenOffsets = 1,
    VocabularyTokenBlob = 2,
    SequenceOffsets = 3,
    CorpusSymbols = 4,
    PostingOffsets = 5,
    Postings = 6,
    EmbeddingTokenOffsets = 7,
    EmbeddingTokenBlob = 8,
    Embeddings = 9,
    CostMetadata = 10,
}

impl SectionKind {
    fn from_raw(raw: u32) -> Result<Self> {
        match raw {
            1 => Ok(Self::VocabularyTokenOffsets),
            2 => Ok(Self::VocabularyTokenBlob),
            3 => Ok(Self::SequenceOffsets),
            4 => Ok(Self::CorpusSymbols),
            5 => Ok(Self::PostingOffsets),
            6 => Ok(Self::Postings),
            7 => Ok(Self::EmbeddingTokenOffsets),
            8 => Ok(Self::EmbeddingTokenBlob),
            9 => Ok(Self::Embeddings),
            10 => Ok(Self::CostMetadata),
            _ => Err(Error::InvalidFile("unknown section kind")),
        }
    }

    fn element_layout(self) -> Option<(usize, usize)> {
        match self {
            Self::VocabularyTokenBlob | Self::EmbeddingTokenBlob | Self::CostMetadata => None,
            Self::VocabularyTokenOffsets
            | Self::EmbeddingTokenOffsets
            | Self::SequenceOffsets
            | Self::PostingOffsets => Some((size_of::<u64>(), align_of::<u64>())),
            Self::CorpusSymbols => Some((size_of::<DiskSymbol>(), align_of::<DiskSymbol>())),
            Self::Postings => Some((size_of::<DiskPosting>(), align_of::<DiskPosting>())),
            Self::Embeddings => Some((size_of::<f32>(), align_of::<f32>())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SectionDescriptor {
    pub(crate) kind: SectionKind,
    pub(crate) offset: u64,
    pub(crate) byte_len: u64,
    pub(crate) element_count: u64,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, Immutable, KnownLayout)]
pub(crate) struct DiskSymbol(u32);

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, Immutable, KnownLayout)]
pub(crate) struct DiskPosting {
    sequence_id: u32,
    position: u32,
}

pub(crate) struct PersistedFile {
    mmap: Arc<Mmap>,
    kind: FileKind,
    sections: BTreeMap<SectionKind, SectionDescriptor>,
}

impl PersistedFile {
    pub(crate) fn open<T, C: TokenCodec<T>>(
        path: &Path,
        expected_kind: FileKind,
        codec: &C,
    ) -> Result<Self> {
        validate_codec_id_len(codec.id())?;

        let mut file = File::open(path)?;
        let mut header_bytes = [0; HEADER_LEN];
        if let Err(error) = file.read_exact(&mut header_bytes) {
            return if error.kind() == std::io::ErrorKind::UnexpectedEof {
                Err(Error::InvalidFile("header is truncated"))
            } else {
                Err(error.into())
            };
        }
        let header = DiskHeader::ref_from_bytes(&header_bytes)
            .map_err(|_| Error::InvalidFile("header has an invalid layout"))?;
        validate_header(header, expected_kind)?;
        if header.file_len.get() != file.metadata()?.len() {
            return Err(Error::InvalidFile("recorded file length does not match"));
        }

        let mmap = map_file(&file)?;
        Self::parse(mmap, expected_kind, codec.id(), codec.version())
    }

    fn parse(
        mmap: Arc<Mmap>,
        expected_kind: FileKind,
        expected_codec: &str,
        expected_codec_version: u32,
    ) -> Result<Self> {
        validate_codec_id_len(expected_codec)?;
        let bytes: &[u8] = &mmap;
        let header = DiskHeader::ref_from_prefix(bytes)
            .map_err(|_| Error::InvalidFile("header is truncated"))?
            .0;
        let kind = validate_header(header, expected_kind)?;
        if header.file_len.get() != u64::try_from(bytes.len()).unwrap() {
            return Err(Error::InvalidFile("recorded file length does not match"));
        }

        let codec_len = header.codec_len.get();
        if codec_len > MAX_CODEC_ID_LEN as u64 {
            return Err(Error::InvalidFile("codec identifier is too long"));
        }
        let codec_offset = header.codec_offset.get();
        let codec_bytes = checked_range(bytes, codec_offset, codec_len)?;
        let actual_codec = std::str::from_utf8(codec_bytes)
            .map_err(|_| Error::InvalidFile("codec identifier is not UTF-8"))?;
        if actual_codec != expected_codec {
            return Err(Error::CodecMismatch {
                expected: expected_codec.to_owned(),
                actual: actual_codec.to_owned(),
            });
        }
        let codec_version = header.codec_version.get();
        if codec_version != expected_codec_version {
            return Err(Error::CodecVersionMismatch {
                expected: expected_codec_version,
                actual: codec_version,
            });
        }

        let section_count = u64::from(header.section_count.get());
        let table_len = section_count
            .checked_mul(SECTION_ENTRY_LEN as u64)
            .ok_or(Error::InvalidFile("section table length overflows"))?;
        let section_table_offset = header.section_table_offset.get();
        let table = checked_range(bytes, section_table_offset, table_len)?;
        let entries = <[DiskSectionEntry]>::ref_from_bytes(table)
            .map_err(|_| Error::InvalidFile("section table has an invalid length"))?;

        let mut sections = BTreeMap::new();
        for entry in entries {
            let kind = SectionKind::from_raw(entry.kind.get())?;
            if entry.flags.get() != 0 {
                return Err(Error::InvalidFile("section flags are not zero"));
            }
            let descriptor = SectionDescriptor {
                kind,
                offset: entry.offset.get(),
                byte_len: entry.byte_len.get(),
                element_count: entry.element_count.get(),
            };
            validate_section(bytes, descriptor)?;
            if sections.insert(kind, descriptor).is_some() {
                return Err(Error::InvalidFile("section kind is duplicated"));
            }
        }

        validate_non_overlapping(
            codec_offset,
            codec_len,
            section_table_offset,
            table_len,
            sections.values().copied(),
        )?;

        let file = Self {
            mmap,
            kind,
            sections,
        };
        file.validate_structure()?;
        Ok(file)
    }

    pub(crate) fn section(&self, kind: SectionKind) -> Result<SectionDescriptor> {
        self.sections
            .get(&kind)
            .copied()
            .ok_or(Error::InvalidFile("required section is missing"))
    }

    pub(crate) fn mapped_slice<T>(&self, kind: SectionKind) -> Result<MappedSlice<T>>
    where
        T: FromBytes + Immutable + KnownLayout,
    {
        let section = self.section(kind)?;
        MappedSlice::new(
            Arc::clone(&self.mmap),
            section.offset,
            section.byte_len,
            section.element_count,
        )
    }

    fn validate_structure(&self) -> Result<()> {
        if self.kind != FileKind::SearchEngine {
            return Ok(());
        }

        let vocabulary_offsets = self.mapped_slice::<u64>(SectionKind::VocabularyTokenOffsets)?;
        let vocabulary_blob = self.section(SectionKind::VocabularyTokenBlob)?;
        let sequence_offsets = self.mapped_slice::<u64>(SectionKind::SequenceOffsets)?;
        let corpus = self.section(SectionKind::CorpusSymbols)?;
        let posting_offsets = self.mapped_slice::<u64>(SectionKind::PostingOffsets)?;
        let postings = self.section(SectionKind::Postings)?;

        validate_offsets(&vocabulary_offsets, vocabulary_blob.byte_len)?;
        validate_offsets(&sequence_offsets, corpus.element_count)?;
        validate_offsets(&posting_offsets, postings.element_count)?;

        let expected_vocabulary_offsets = vocabulary_blob
            .element_count
            .checked_add(1)
            .ok_or(Error::InvalidFile("vocabulary offset count overflows"))?;
        if vocabulary_offsets.len() as u64 != expected_vocabulary_offsets {
            return Err(Error::InvalidFile(
                "vocabulary offset count does not match token count",
            ));
        }
        if posting_offsets.len() as u64 != expected_vocabulary_offsets {
            return Err(Error::InvalidFile(
                "posting offset count does not match vocabulary",
            ));
        }
        Ok(())
    }
}

fn validate_header(header: &DiskHeader, expected_kind: FileKind) -> Result<FileKind> {
    if header.magic != MAGIC {
        return Err(Error::InvalidFile("magic does not match"));
    }
    if header.format_version.get() != FORMAT_VERSION {
        return Err(Error::UnsupportedFormatVersion(header.format_version.get()));
    }
    if header.endian_marker.get() != ENDIAN_MARKER {
        return Err(Error::EndiannessMismatch);
    }
    if cfg!(target_endian = "big") {
        return Err(Error::UnsupportedHostEndianness);
    }
    let kind = FileKind::from_raw(header.file_kind.get())?;
    if kind != expected_kind {
        return Err(Error::InvalidFile("file kind does not match"));
    }
    if header.header_len.get() as usize != HEADER_LEN {
        return Err(Error::InvalidFile("header size does not match"));
    }
    Ok(kind)
}

fn validate_codec_id_len(codec_id: &str) -> Result<()> {
    if codec_id.len() > MAX_CODEC_ID_LEN {
        return Err(Error::CodecIdTooLong {
            length: codec_id.len(),
            max: MAX_CODEC_ID_LEN,
        });
    }
    Ok(())
}

fn validate_section(bytes: &[u8], section: SectionDescriptor) -> Result<()> {
    checked_range(bytes, section.offset, section.byte_len)?;
    if let Some((element_size, alignment)) = section.kind.element_layout() {
        let expected_len = section
            .element_count
            .checked_mul(element_size as u64)
            .ok_or(Error::InvalidFile("section byte length overflows"))?;
        if section.byte_len != expected_len {
            return Err(Error::InvalidFile(
                "section byte length does not match element count",
            ));
        }
        if !section.offset.is_multiple_of(alignment as u64) {
            return Err(Error::InvalidFile("section offset is not aligned"));
        }
    }
    Ok(())
}

fn validate_non_overlapping(
    codec_offset: u64,
    codec_len: u64,
    table_offset: u64,
    table_len: u64,
    sections: impl Iterator<Item = SectionDescriptor>,
) -> Result<()> {
    let mut ranges = vec![
        (0, HEADER_LEN as u64),
        (codec_offset, codec_len),
        (table_offset, table_len),
    ];
    ranges.extend(sections.map(|section| (section.offset, section.byte_len)));
    ranges.retain(|(_, len)| *len != 0);
    ranges.sort_unstable_by_key(|(offset, _)| *offset);
    for pair in ranges.windows(2) {
        let previous_end = pair[0]
            .0
            .checked_add(pair[0].1)
            .ok_or(Error::InvalidFile("file range overflows"))?;
        if previous_end > pair[1].0 {
            return Err(Error::InvalidFile("file regions overlap"));
        }
    }
    Ok(())
}

fn validate_offsets(offsets: &[u64], payload_len: u64) -> Result<()> {
    if offsets.first() != Some(&0) || offsets.last() != Some(&payload_len) {
        return Err(Error::InvalidFile("offset endpoints do not match payload"));
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(Error::InvalidFile("offsets are not monotonic"));
    }
    Ok(())
}

fn checked_range(bytes: &[u8], offset: u64, len: u64) -> Result<&[u8]> {
    let end = offset
        .checked_add(len)
        .ok_or(Error::InvalidFile("file range overflows"))?;
    let start = usize::try_from(offset).map_err(|_| Error::PlatformSizeOverflow)?;
    let end = usize::try_from(end).map_err(|_| Error::PlatformSizeOverflow)?;
    bytes
        .get(start..end)
        .ok_or(Error::InvalidFile("file range is truncated"))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;

    use memmap2::MmapOptions;
    use zerocopy::IntoBytes;

    use super::{
        DiskHeader, DiskPosting, DiskSectionEntry, DiskSymbol, ENDIAN_MARKER, FORMAT_VERSION,
        FileKind, HEADER_LEN, MAGIC, MAX_CODEC_ID_LEN, PersistedFile, SECTION_ENTRY_LEN,
        SectionKind, U32, U64,
    };
    use crate::errors::Error;
    use crate::persistence::{CharCodec, TokenCodec};

    struct Fixture {
        bytes: Vec<u8>,
        entries: Vec<(SectionKind, usize)>,
    }

    impl Fixture {
        fn valid() -> Self {
            let codec = b"yurine:char:u32le";
            let section_kinds = [
                SectionKind::VocabularyTokenOffsets,
                SectionKind::VocabularyTokenBlob,
                SectionKind::SequenceOffsets,
                SectionKind::CorpusSymbols,
                SectionKind::PostingOffsets,
                SectionKind::Postings,
            ];
            let mut bytes = vec![0; HEADER_LEN];
            let codec_offset = bytes.len();
            bytes.extend_from_slice(codec);
            align(&mut bytes, 8);
            let table_offset = bytes.len();
            bytes.resize(table_offset + section_kinds.len() * SECTION_ENTRY_LEN, 0);

            let mut sections = Vec::new();
            append_section(
                &mut bytes,
                &mut sections,
                SectionKind::VocabularyTokenOffsets,
                3,
                8,
                |bytes| append_u64s(bytes, [0, 1, 2]),
            );
            append_section(
                &mut bytes,
                &mut sections,
                SectionKind::VocabularyTokenBlob,
                2,
                1,
                |bytes| bytes.extend_from_slice(b"ab"),
            );
            append_section(
                &mut bytes,
                &mut sections,
                SectionKind::SequenceOffsets,
                3,
                8,
                |bytes| append_u64s(bytes, [0, 2, 3]),
            );
            append_section(
                &mut bytes,
                &mut sections,
                SectionKind::CorpusSymbols,
                3,
                4,
                |bytes| append_u32s(bytes, [0, 1, 1]),
            );
            append_section(
                &mut bytes,
                &mut sections,
                SectionKind::PostingOffsets,
                3,
                8,
                |bytes| append_u64s(bytes, [0, 1, 3]),
            );
            append_section(
                &mut bytes,
                &mut sections,
                SectionKind::Postings,
                3,
                4,
                |bytes| append_u32s(bytes, [0, 0, 0, 1, 1, 0]),
            );

            let header = DiskHeader {
                magic: MAGIC,
                format_version: U32::new(FORMAT_VERSION),
                endian_marker: U32::new(ENDIAN_MARKER),
                file_kind: U32::new(FileKind::SearchEngine as u32),
                header_len: U32::new(HEADER_LEN as u32),
                codec_version: U32::new(1),
                section_count: U32::new(section_kinds.len() as u32),
                codec_offset: U64::new(codec_offset as u64),
                codec_len: U64::new(codec.len() as u64),
                section_table_offset: U64::new(table_offset as u64),
                file_len: U64::new(bytes.len() as u64),
            };
            bytes[..HEADER_LEN].copy_from_slice(header.as_bytes());

            let mut entries = Vec::new();
            for (index, (kind, offset, byte_len, count)) in sections.into_iter().enumerate() {
                let entry_offset = table_offset + index * SECTION_ENTRY_LEN;
                let entry = DiskSectionEntry {
                    kind: U32::new(kind as u32),
                    flags: U32::ZERO,
                    offset: U64::new(offset as u64),
                    byte_len: U64::new(byte_len as u64),
                    element_count: U64::new(count),
                };
                bytes[entry_offset..entry_offset + SECTION_ENTRY_LEN]
                    .copy_from_slice(entry.as_bytes());
                entries.push((kind, entry_offset));
            }
            Self { bytes, entries }
        }

        fn entry(&self, kind: SectionKind) -> usize {
            self.entries
                .iter()
                .find_map(|(candidate, offset)| (*candidate == kind).then_some(*offset))
                .unwrap()
        }

        fn section_offset(&self, kind: SectionKind) -> usize {
            let entry = self.entry(kind);
            u64::from_le_bytes(self.bytes[entry + 8..entry + 16].try_into().unwrap()) as usize
        }

        fn parse(&self) -> Result<PersistedFile, Error> {
            PersistedFile::parse(
                test_map(&self.bytes),
                FileKind::SearchEngine,
                "yurine:char:u32le",
                1,
            )
        }
    }

    fn append_section(
        bytes: &mut Vec<u8>,
        sections: &mut Vec<(SectionKind, usize, usize, u64)>,
        kind: SectionKind,
        count: u64,
        alignment: usize,
        append: impl FnOnce(&mut Vec<u8>),
    ) {
        align(bytes, alignment);
        let offset = bytes.len();
        append(bytes);
        sections.push((kind, offset, bytes.len() - offset, count));
    }

    fn align(bytes: &mut Vec<u8>, alignment: usize) {
        let padding = (alignment - bytes.len() % alignment) % alignment;
        bytes.resize(bytes.len() + padding, 0);
    }

    fn append_u32s<const N: usize>(bytes: &mut Vec<u8>, values: [u32; N]) {
        bytes.extend_from_slice(values.map(U32::new).as_bytes());
    }

    fn append_u64s<const N: usize>(bytes: &mut Vec<u8>, values: [u64; N]) {
        bytes.extend_from_slice(values.map(U64::new).as_bytes());
    }

    fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn test_map(bytes: &[u8]) -> Arc<memmap2::Mmap> {
        let mut mmap = MmapOptions::new().len(bytes.len()).map_anon().unwrap();
        mmap.copy_from_slice(bytes);
        Arc::new(mmap.make_read_only().unwrap())
    }

    #[test]
    fn valid_file_passes_structural_validation() {
        let file = Fixture::valid().parse().unwrap();
        assert_eq!(
            file.mapped_slice::<u64>(SectionKind::PostingOffsets)
                .unwrap()
                .as_slice(),
            [0, 1, 3]
        );
    }

    #[test]
    fn open_checks_file_length_before_mapping() {
        let fixture = Fixture::valid();
        let path =
            std::env::temp_dir().join(format!("yurine-format-{}-{}", std::process::id(), line!()));
        fs::write(&path, &fixture.bytes).unwrap();
        PersistedFile::open(&path, FileKind::SearchEngine, &CharCodec).unwrap();
        fs::write(&path, &fixture.bytes[..fixture.bytes.len() - 1]).unwrap();
        let result = PersistedFile::open(&path, FileKind::SearchEngine, &CharCodec);
        fs::remove_file(path).unwrap();
        assert!(matches!(
            result,
            Err(Error::InvalidFile("recorded file length does not match"))
        ));
    }

    #[test]
    fn rejects_corrupt_magic_and_truncated_header() {
        let mut corrupt = Fixture::valid();
        corrupt.bytes[0] ^= 0xff;
        assert!(matches!(corrupt.parse(), Err(Error::InvalidFile(_))));

        let mut truncated = Fixture::valid();
        truncated.bytes.truncate(HEADER_LEN - 1);
        assert!(matches!(truncated.parse(), Err(Error::InvalidFile(_))));
    }

    #[test]
    fn rejects_version_endianness_and_codec_mismatches() {
        let mut version = Fixture::valid();
        write_u32(&mut version.bytes, 8, FORMAT_VERSION + 1);
        assert_eq!(
            version.parse().err(),
            Some(Error::UnsupportedFormatVersion(2))
        );

        let mut endian = Fixture::valid();
        write_u32(&mut endian.bytes, 12, ENDIAN_MARKER.swap_bytes());
        assert_eq!(endian.parse().err(), Some(Error::EndiannessMismatch));

        let fixture = Fixture::valid();
        assert!(matches!(
            PersistedFile::parse(
                test_map(&fixture.bytes),
                FileKind::SearchEngine,
                "another-codec",
                1
            ),
            Err(Error::CodecMismatch { .. })
        ));
        assert!(matches!(
            PersistedFile::parse(
                test_map(&fixture.bytes),
                FileKind::SearchEngine,
                "yurine:char:u32le",
                2
            ),
            Err(Error::CodecVersionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_out_of_range_misaligned_and_overlapping_sections() {
        let mut range = Fixture::valid();
        let entry = range.entry(SectionKind::CorpusSymbols);
        write_u64(&mut range.bytes, entry + 8, u64::MAX - 1);
        assert!(matches!(range.parse(), Err(Error::InvalidFile(_))));

        let mut alignment = Fixture::valid();
        let entry = alignment.entry(SectionKind::SequenceOffsets);
        let offset = alignment.section_offset(SectionKind::SequenceOffsets);
        write_u64(&mut alignment.bytes, entry + 8, (offset + 1) as u64);
        assert!(matches!(alignment.parse(), Err(Error::InvalidFile(_))));

        let mut overlap = Fixture::valid();
        let corpus_entry = overlap.entry(SectionKind::CorpusSymbols);
        let blob_offset = overlap.section_offset(SectionKind::VocabularyTokenBlob);
        write_u64(&mut overlap.bytes, corpus_entry + 8, blob_offset as u64);
        assert!(matches!(overlap.parse(), Err(Error::InvalidFile(_))));
    }

    #[test]
    fn open_rejects_non_monotonic_offsets() {
        for kind in [
            SectionKind::VocabularyTokenOffsets,
            SectionKind::SequenceOffsets,
            SectionKind::PostingOffsets,
        ] {
            let mut fixture = Fixture::valid();
            let offset = fixture.section_offset(kind);
            write_u64(&mut fixture.bytes, offset + 8, u64::MAX);
            assert!(matches!(
                fixture.parse(),
                Err(Error::InvalidFile("offsets are not monotonic"))
            ));
        }
    }

    #[test]
    fn open_rejects_offset_endpoint_and_count_mismatches() {
        let mut endpoint = Fixture::valid();
        let offset = endpoint.section_offset(SectionKind::SequenceOffsets);
        write_u64(&mut endpoint.bytes, offset + 16, 2);
        assert!(matches!(
            endpoint.parse(),
            Err(Error::InvalidFile("offset endpoints do not match payload"))
        ));

        let mut vocabulary_count = Fixture::valid();
        let entry = vocabulary_count.entry(SectionKind::VocabularyTokenBlob);
        write_u64(&mut vocabulary_count.bytes, entry + 24, 1);
        assert!(matches!(
            vocabulary_count.parse(),
            Err(Error::InvalidFile(
                "vocabulary offset count does not match token count"
            ))
        ));

        let mut posting_count = Fixture::valid();
        let entry = posting_count.entry(SectionKind::PostingOffsets);
        let offset = posting_count.section_offset(SectionKind::PostingOffsets);
        write_u64(&mut posting_count.bytes, offset + 8, 3);
        write_u64(&mut posting_count.bytes, entry + 24, 2);
        write_u64(&mut posting_count.bytes, entry + 16, 16);
        assert!(matches!(
            posting_count.parse(),
            Err(Error::InvalidFile(
                "posting offset count does not match vocabulary"
            ))
        ));
    }

    #[test]
    fn open_does_not_validate_payload_semantics() {
        let mut fixture = Fixture::valid();
        let corpus = fixture.section_offset(SectionKind::CorpusSymbols);
        write_u32(&mut fixture.bytes, corpus, u32::MAX);
        let postings = fixture.section_offset(SectionKind::Postings);
        write_u32(&mut fixture.bytes, postings, u32::MAX);

        fixture.parse().unwrap();
    }

    #[test]
    fn rejects_oversized_codec_identifiers() {
        struct LargeCodec(String);
        impl TokenCodec<char> for LargeCodec {
            fn id(&self) -> &str {
                &self.0
            }
            fn encode(&self, _: &char, _: &mut Vec<u8>) -> crate::errors::Result<()> {
                Ok(())
            }
            fn decode(&self, _: &[u8]) -> crate::errors::Result<char> {
                Ok('a')
            }
        }
        let path = Path::new("unused");
        let codec = LargeCodec("x".repeat(MAX_CODEC_ID_LEN + 1));
        assert!(matches!(
            PersistedFile::open(path, FileKind::SearchEngine, &codec),
            Err(Error::CodecIdTooLong { .. })
        ));
    }

    #[test]
    fn disk_types_have_fixed_width_layouts() {
        assert_eq!(size_of::<DiskHeader>(), 64);
        assert_eq!(align_of::<DiskHeader>(), 1);
        assert_eq!(size_of::<DiskSectionEntry>(), 32);
        assert_eq!(align_of::<DiskSectionEntry>(), 1);
        assert_eq!(size_of::<DiskSymbol>(), 4);
        assert_eq!(align_of::<DiskSymbol>(), 4);
        assert_eq!(size_of::<DiskPosting>(), 8);
        assert_eq!(align_of::<DiskPosting>(), 4);
    }
}
