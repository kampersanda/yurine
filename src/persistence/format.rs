use std::collections::BTreeMap;
use std::mem::{align_of, size_of};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use zerocopy::{FromBytes, Immutable, KnownLayout};

use super::TokenCodec;
use super::storage::{MappedSlice, map_file};
use crate::errors::{Error, Result};

pub(crate) const MAGIC: [u8; 8] = *b"YURINE\0\0";
pub(crate) const FORMAT_VERSION: u32 = 1;
pub(crate) const HEADER_LEN: usize = 64;
pub(crate) const SECTION_ENTRY_LEN: usize = 32;
const ENDIAN_MARKER: u32 = 0x0102_0304;

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
    Vocabulary = 1,
    Alphabet = 2,
    SequenceOffsets = 3,
    CorpusSymbols = 4,
    PostingOffsets = 5,
    Postings = 6,
    EmbeddingTokens = 7,
    Embeddings = 8,
    CostMetadata = 9,
}

impl SectionKind {
    fn from_raw(raw: u32) -> Result<Self> {
        match raw {
            1 => Ok(Self::Vocabulary),
            2 => Ok(Self::Alphabet),
            3 => Ok(Self::SequenceOffsets),
            4 => Ok(Self::CorpusSymbols),
            5 => Ok(Self::PostingOffsets),
            6 => Ok(Self::Postings),
            7 => Ok(Self::EmbeddingTokens),
            8 => Ok(Self::Embeddings),
            9 => Ok(Self::CostMetadata),
            _ => Err(Error::InvalidFile("unknown section kind")),
        }
    }

    fn element_layout(self) -> Option<(usize, usize)> {
        match self {
            Self::Vocabulary | Self::EmbeddingTokens | Self::CostMetadata => None,
            Self::Alphabet | Self::CorpusSymbols => {
                Some((size_of::<DiskSymbol>(), align_of::<DiskSymbol>()))
            }
            Self::SequenceOffsets | Self::PostingOffsets => {
                Some((size_of::<u64>(), align_of::<u64>()))
            }
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

impl DiskSymbol {
    #[cfg(test)]
    pub(crate) const fn new(value: u32) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u32 {
        self.0
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, Immutable, KnownLayout)]
pub(crate) struct DiskPosting {
    sequence_id: u32,
    position: u32,
}

impl DiskPosting {
    #[cfg(test)]
    pub(crate) const fn new(sequence_id: u32, position: u32) -> Self {
        Self {
            sequence_id,
            position,
        }
    }

    pub(crate) const fn sequence_id(self) -> u32 {
        self.sequence_id
    }

    pub(crate) const fn position(self) -> u32 {
        self.position
    }
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
        let mmap = map_file(path)?;
        Self::parse(mmap, expected_kind, codec.id(), codec.version())
    }

    fn parse(
        mmap: Arc<Mmap>,
        expected_kind: FileKind,
        expected_codec: &str,
        expected_codec_version: u32,
    ) -> Result<Self> {
        let bytes: &[u8] = &mmap;
        if bytes.len() < HEADER_LEN {
            return Err(Error::InvalidFile("header is truncated"));
        }
        if bytes[..8] != MAGIC {
            return Err(Error::InvalidFile("magic does not match"));
        }
        if read_u32(bytes, 8)? != FORMAT_VERSION {
            return Err(Error::UnsupportedFormatVersion(read_u32(bytes, 8)?));
        }
        if read_u32(bytes, 12)? != ENDIAN_MARKER {
            return Err(Error::EndiannessMismatch);
        }
        if cfg!(target_endian = "big") {
            return Err(Error::UnsupportedHostEndianness);
        }

        let kind = FileKind::from_raw(read_u32(bytes, 16)?)?;
        if kind != expected_kind {
            return Err(Error::InvalidFile("file kind does not match"));
        }
        if read_u32(bytes, 20)? as usize != HEADER_LEN {
            return Err(Error::InvalidFile("header size does not match"));
        }

        let codec_version = read_u32(bytes, 24)?;
        let section_count = u64::from(read_u32(bytes, 28)?);
        let codec_offset = read_u64(bytes, 32)?;
        let codec_len = read_u64(bytes, 40)?;
        let section_table_offset = read_u64(bytes, 48)?;
        let recorded_file_len = read_u64(bytes, 56)?;
        if recorded_file_len != u64::try_from(bytes.len()).unwrap() {
            return Err(Error::InvalidFile("recorded file length does not match"));
        }

        let codec_bytes = checked_range(bytes, codec_offset, codec_len)?;
        let actual_codec = std::str::from_utf8(codec_bytes)
            .map_err(|_| Error::InvalidFile("codec identifier is not UTF-8"))?;
        if actual_codec != expected_codec {
            return Err(Error::CodecMismatch {
                expected: expected_codec.to_owned(),
                actual: actual_codec.to_owned(),
            });
        }
        if codec_version != expected_codec_version {
            return Err(Error::CodecVersionMismatch {
                expected: expected_codec_version,
                actual: codec_version,
            });
        }

        let table_len = section_count
            .checked_mul(SECTION_ENTRY_LEN as u64)
            .ok_or(Error::InvalidFile("section table length overflows"))?;
        let table = checked_range(bytes, section_table_offset, table_len)?;
        let mut sections = BTreeMap::new();
        for entry in table.chunks_exact(SECTION_ENTRY_LEN) {
            let kind = SectionKind::from_raw(read_u32(entry, 0)?)?;
            if read_u32(entry, 4)? != 0 {
                return Err(Error::InvalidFile("section flags are not zero"));
            }
            let descriptor = SectionDescriptor {
                kind,
                offset: read_u64(entry, 8)?,
                byte_len: read_u64(entry, 16)?,
                element_count: read_u64(entry, 24)?,
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
        let vocabulary = self.section(SectionKind::Vocabulary)?;
        let alphabet = self.mapped_slice::<DiskSymbol>(SectionKind::Alphabet)?;
        let sequence_offsets = self.mapped_slice::<u64>(SectionKind::SequenceOffsets)?;
        let corpus = self.section(SectionKind::CorpusSymbols)?;
        let posting_offsets = self.mapped_slice::<u64>(SectionKind::PostingOffsets)?;
        let postings = self.section(SectionKind::Postings)?;

        if sequence_offsets.is_empty() || posting_offsets.is_empty() {
            return Err(Error::InvalidFile("offset section is empty"));
        }
        validate_offset_endpoints(&sequence_offsets, corpus.element_count)?;
        validate_offset_endpoints(&posting_offsets, postings.element_count)?;
        if posting_offsets.len() as u64 != vocabulary.element_count + 1 {
            return Err(Error::InvalidFile(
                "posting offset count does not match vocabulary",
            ));
        }
        validate_alphabet(&alphabet, vocabulary.element_count)?;
        Ok(())
    }

    /// Fully scans large SearchEngine sections for semantic consistency.
    pub(crate) fn verify(&self) -> Result<()> {
        if self.kind != FileKind::SearchEngine {
            return Ok(());
        }
        let vocabulary_len = self.section(SectionKind::Vocabulary)?.element_count;
        let alphabet = self.mapped_slice::<DiskSymbol>(SectionKind::Alphabet)?;
        let sequence_offsets = self.mapped_slice::<u64>(SectionKind::SequenceOffsets)?;
        let corpus = self.mapped_slice::<DiskSymbol>(SectionKind::CorpusSymbols)?;
        let posting_offsets = self.mapped_slice::<u64>(SectionKind::PostingOffsets)?;
        let postings = self.mapped_slice::<DiskPosting>(SectionKind::Postings)?;

        validate_offsets(&sequence_offsets, corpus.len() as u64)?;
        validate_offsets(&posting_offsets, postings.len() as u64)?;
        if corpus
            .iter()
            .any(|symbol| u64::from(symbol.get()) >= vocabulary_len)
        {
            return Err(Error::InvalidFile(
                "corpus symbol is outside the vocabulary",
            ));
        }
        let mut alphabet_seen = vec![false; alphabet.len()];
        for symbol in corpus.iter().map(|symbol| symbol.get()) {
            let index = alphabet
                .binary_search_by_key(&symbol, |candidate| candidate.get())
                .map_err(|_| Error::InvalidFile("corpus symbol is absent from the alphabet"))?;
            alphabet_seen[index] = true;
        }
        if alphabet_seen.iter().any(|seen| !seen) {
            return Err(Error::InvalidFile("alphabet contains an unused symbol"));
        }
        if postings.len() != corpus.len() {
            return Err(Error::InvalidFile(
                "posting count does not match corpus symbol count",
            ));
        }

        for symbol in 0..vocabulary_len {
            let symbol_index = usize::try_from(symbol).map_err(|_| Error::PlatformSizeOverflow)?;
            let start = posting_offsets[symbol_index] as usize;
            let end = posting_offsets[symbol_index + 1] as usize;
            let mut previous = None;
            for posting in &postings[start..end] {
                let key = (posting.sequence_id(), posting.position());
                if previous.is_some_and(|previous| previous >= key) {
                    return Err(Error::InvalidFile("postings are not strictly ordered"));
                }
                previous = Some(key);

                let sequence_id = posting.sequence_id() as usize;
                let Some((&sequence_start, &sequence_end)) = sequence_offsets
                    .get(sequence_id)
                    .zip(sequence_offsets.get(sequence_id + 1))
                else {
                    return Err(Error::InvalidFile("posting has an unknown sequence id"));
                };
                let position = u64::from(posting.position());
                let corpus_index = sequence_start
                    .checked_add(position)
                    .filter(|index| *index < sequence_end)
                    .ok_or(Error::InvalidFile(
                        "posting position is outside its sequence",
                    ))?;
                if u64::from(corpus[corpus_index as usize].get()) != symbol {
                    return Err(Error::InvalidFile("posting does not match the corpus"));
                }
            }
        }
        Ok(())
    }
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

fn validate_offset_endpoints(offsets: &[u64], payload_len: u64) -> Result<()> {
    if offsets.first() != Some(&0) || offsets.last() != Some(&payload_len) {
        return Err(Error::InvalidFile("offset endpoints do not match payload"));
    }
    Ok(())
}

fn validate_offsets(offsets: &[u64], payload_len: u64) -> Result<()> {
    validate_offset_endpoints(offsets, payload_len)?;
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(Error::InvalidFile("offsets are not monotonic"));
    }
    Ok(())
}

fn validate_alphabet(alphabet: &[DiskSymbol], vocabulary_len: u64) -> Result<()> {
    let mut previous = None;
    for symbol in alphabet {
        let value = symbol.get();
        if u64::from(value) >= vocabulary_len {
            return Err(Error::InvalidFile(
                "alphabet symbol is outside the vocabulary",
            ));
        }
        if previous.is_some_and(|previous| previous >= value) {
            return Err(Error::InvalidFile("alphabet is not sorted and unique"));
        }
        previous = Some(value);
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

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or(Error::InvalidFile("fixed-width value is truncated"))?
        .try_into()
        .unwrap();
    Ok(u32::from_le_bytes(raw))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let raw = bytes
        .get(offset..offset + 8)
        .ok_or(Error::InvalidFile("fixed-width value is truncated"))?
        .try_into()
        .unwrap();
    Ok(u64::from_le_bytes(raw))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;

    use memmap2::MmapOptions;

    use super::{
        DiskPosting, DiskSymbol, ENDIAN_MARKER, FORMAT_VERSION, FileKind, HEADER_LEN, MAGIC,
        PersistedFile, SECTION_ENTRY_LEN, SectionKind,
    };
    use crate::errors::Error;
    use crate::persistence::CharCodec;

    struct Fixture {
        bytes: Vec<u8>,
        entries: Vec<(SectionKind, usize)>,
    }

    impl Fixture {
        fn valid() -> Self {
            let codec = b"yurine:char:u32le";
            let section_kinds = [
                SectionKind::Vocabulary,
                SectionKind::Alphabet,
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
                SectionKind::Vocabulary,
                2,
                1,
                |bytes| bytes.extend_from_slice(b"ab"),
            );
            append_section(
                &mut bytes,
                &mut sections,
                SectionKind::Alphabet,
                2,
                4,
                |bytes| append_u32s(bytes, [0, 1]),
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
                |bytes| {
                    append_u32s(bytes, [0, 0, 0, 1, 1, 0]);
                },
            );

            bytes[..8].copy_from_slice(&MAGIC);
            write_u32(&mut bytes, 8, FORMAT_VERSION);
            write_u32(&mut bytes, 12, ENDIAN_MARKER);
            write_u32(&mut bytes, 16, FileKind::SearchEngine as u32);
            write_u32(&mut bytes, 20, HEADER_LEN as u32);
            write_u32(&mut bytes, 24, 1);
            write_u32(&mut bytes, 28, section_kinds.len() as u32);
            write_u64(&mut bytes, 32, codec_offset as u64);
            write_u64(&mut bytes, 40, codec.len() as u64);
            write_u64(&mut bytes, 48, table_offset as u64);
            let file_len = bytes.len() as u64;
            write_u64(&mut bytes, 56, file_len);

            let mut entries = Vec::new();
            for (index, (kind, offset, byte_len, count)) in sections.into_iter().enumerate() {
                let entry = table_offset + index * SECTION_ENTRY_LEN;
                write_u32(&mut bytes, entry, kind as u32);
                write_u64(&mut bytes, entry + 8, offset as u64);
                write_u64(&mut bytes, entry + 16, byte_len as u64);
                write_u64(&mut bytes, entry + 24, count);
                entries.push((kind, entry));
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
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn append_u64s<const N: usize>(bytes: &mut Vec<u8>, values: [u64; N]) {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
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
    fn valid_file_passes_structural_and_complete_verification() {
        let file = Fixture::valid().parse().unwrap();

        assert_eq!(
            file.mapped_slice::<DiskSymbol>(SectionKind::CorpusSymbols)
                .unwrap()
                .iter()
                .map(|symbol| symbol.get())
                .collect::<Vec<_>>(),
            [0, 1, 1]
        );
        assert_eq!(
            file.mapped_slice::<DiskPosting>(SectionKind::Postings)
                .unwrap()[2],
            DiskPosting::new(1, 0)
        );
        file.verify().unwrap();
    }

    #[test]
    fn open_maps_a_file_and_checks_the_codec() {
        let fixture = Fixture::valid();
        let path =
            std::env::temp_dir().join(format!("yurine-format-{}-{}", std::process::id(), line!()));
        fs::write(&path, fixture.bytes).unwrap();
        let file = PersistedFile::open(&path, FileKind::SearchEngine, &CharCodec).unwrap();
        fs::remove_file(path).unwrap();

        file.verify().unwrap();
    }

    #[test]
    fn rejects_corrupt_magic_and_truncated_files() {
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
            Some(Error::UnsupportedFormatVersion(FORMAT_VERSION + 1))
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
                1,
            ),
            Err(Error::CodecMismatch { .. })
        ));
        assert!(matches!(
            PersistedFile::parse(
                test_map(&fixture.bytes),
                FileKind::SearchEngine,
                "yurine:char:u32le",
                2,
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
        let alphabet_offset = overlap.section_offset(SectionKind::Alphabet);
        write_u64(&mut overlap.bytes, corpus_entry + 8, alphabet_offset as u64);
        assert!(matches!(overlap.parse(), Err(Error::InvalidFile(_))));
    }

    #[test]
    fn normal_open_skips_full_payload_but_verify_finds_bad_symbols() {
        let mut fixture = Fixture::valid();
        let corpus_offset = fixture.section_offset(SectionKind::CorpusSymbols);
        write_u32(&mut fixture.bytes, corpus_offset + 8, 99);

        let file = fixture.parse().unwrap();
        assert!(matches!(file.verify(), Err(Error::InvalidFile(_))));
    }

    #[test]
    fn verify_rejects_non_monotonic_offsets_and_inconsistent_postings() {
        let mut offsets = Fixture::valid();
        let sequence_offsets = offsets.section_offset(SectionKind::SequenceOffsets);
        write_u64(&mut offsets.bytes, sequence_offsets + 8, 4);
        let file = offsets.parse().unwrap();
        assert!(matches!(file.verify(), Err(Error::InvalidFile(_))));

        let mut postings = Fixture::valid();
        let posting_offset = postings.section_offset(SectionKind::Postings);
        write_u32(&mut postings.bytes, posting_offset + 4, 1);
        let file = postings.parse().unwrap();
        assert!(matches!(file.verify(), Err(Error::InvalidFile(_))));
    }

    #[test]
    fn disk_types_have_fixed_width_layouts() {
        assert_eq!(size_of::<DiskSymbol>(), 4);
        assert_eq!(align_of::<DiskSymbol>(), 4);
        assert_eq!(DiskSymbol::new(7).get(), 7);
        assert_eq!(size_of::<DiskPosting>(), 8);
        assert_eq!(align_of::<DiskPosting>(), 4);
        assert_eq!(DiskPosting::new(3, 5).sequence_id(), 3);
        assert_eq!(DiskPosting::new(3, 5).position(), 5);
    }
}
