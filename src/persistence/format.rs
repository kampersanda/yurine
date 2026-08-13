//! Reader and structural validator for the version 1 persistence format.
//!
//! A file consists of a fixed-width header, a codec identifier, a section
//! table, and the section payloads. Offsets in the header and section table are
//! absolute byte offsets from the beginning of the file.
//!
//! Opening a file deliberately validates it in three stages:
//!
//! 1. Read the fixed header and compare its recorded file length before mmap.
//! 2. Map the file once, then validate metadata ranges, layouts, and overlap.
//! 3. Scan offset arrays needed for safe slicing, without scanning the large
//!    corpus-symbol and posting payloads.
//!
//! Token decoding and payload-level semantic validation belong to the open and
//! verify APIs introduced in implementation unit 2.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::mem::{align_of, size_of};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use zerocopy::byteorder::little_endian::{U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use super::TokenCodec;
use super::storage::{MappedSlice, map_file};
use crate::errors::{Error, Result};
use crate::types::{Posting, Symbol};

pub(crate) const MAGIC: [u8; 8] = *b"YURINE\0\0";
pub(crate) const FORMAT_VERSION: u32 = 1;
pub(crate) const MAX_CODEC_ID_LEN: usize = 255;
// Detects a damaged or incorrectly encoded header. The v1 payload slices are
// native integer views, so big-endian hosts are rejected separately below.
const ENDIAN_MARKER: u32 = 0x0102_0304;

/// Fixed-size prefix that can be decoded before the file is memory-mapped.
///
/// Explicit little-endian wrappers make the on-disk byte order independent of
/// the host. `Unaligned` keeps parsing safe even when the input byte slice has
/// no stronger alignment than `u8`.
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

/// One entry in the section table.
///
/// `byte_len` describes the occupied file range. `element_count` describes the
/// logical number of fixed-width values, or format-specific records for blobs.
#[repr(C)]
#[derive(Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
struct DiskSectionEntry {
    kind: U32,
    flags: U32,
    offset: U64,
    byte_len: U64,
    element_count: U64,
}

/// Host-native header values retained after the fixed prefix is validated.
///
/// Production `open` validates the header before mmap and passes this value to
/// the mapped parser so the same header checks are not repeated.
#[derive(Debug, Clone, Copy)]
struct ValidatedHeader {
    kind: FileKind,
    codec_version: u32,
    section_count: u32,
    codec_offset: u64,
    codec_len: u64,
    section_table_offset: u64,
    file_len: u64,
}

impl ValidatedHeader {
    fn from_disk(header: &DiskHeader, kind: FileKind) -> Self {
        Self {
            kind,
            codec_version: header.codec_version.get(),
            section_count: header.section_count.get(),
            codec_offset: header.codec_offset.get(),
            codec_len: header.codec_len.get(),
            section_table_offset: header.section_table_offset.get(),
            file_len: header.file_len.get(),
        }
    }
}

pub(crate) const HEADER_LEN: usize = size_of::<DiskHeader>();
pub(crate) const SECTION_ENTRY_LEN: usize = size_of::<DiskSectionEntry>();

/// Identifies which persisted top-level object a file contains.
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

/// Stable identifiers for section-table entries across all persisted objects.
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
        // Blob sections are byte ranges whose logical record count cannot be
        // derived from a fixed element width. Every other section can be
        // checked generically before creating a typed view.
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

/// Validated metadata for one section payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SectionDescriptor {
    pub(crate) kind: SectionKind,
    pub(crate) offset: u64,
    pub(crate) byte_len: u64,
    pub(crate) element_count: u64,
}

/// On-disk symbol representation, kept separate from the domain `Symbol` type.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, Immutable, KnownLayout)]
pub(crate) struct DiskSymbol(u32);

/// On-disk posting representation, kept separate from the domain `Posting`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, Immutable, KnownLayout)]
pub(crate) struct DiskPosting {
    sequence_id: u32,
    position: u32,
}

/// One immutable mmap plus the validated descriptors of its sections.
///
/// All typed section views clone the same `Arc<Mmap>`, so no section is mapped
/// separately and each view keeps the backing bytes alive for its full life.
pub(crate) struct PersistedFile {
    mmap: Arc<Mmap>,
    kind: FileKind,
    sections: BTreeMap<SectionKind, SectionDescriptor>,
}

pub(crate) enum SectionData<'a> {
    Bytes { bytes: &'a [u8], element_count: u64 },
    U64(&'a [u64]),
    Symbols(&'a [Symbol]),
    Postings(&'a [Posting]),
}

impl SectionData<'_> {
    fn byte_len(&self) -> u64 {
        match self {
            Self::Bytes { bytes, .. } => bytes.len() as u64,
            Self::U64(values) => std::mem::size_of_val(*values) as u64,
            Self::Symbols(values) => std::mem::size_of_val(*values) as u64,
            Self::Postings(values) => std::mem::size_of_val(*values) as u64,
        }
    }

    fn element_count(&self) -> u64 {
        match self {
            Self::Bytes { element_count, .. } => *element_count,
            Self::U64(values) => values.len() as u64,
            Self::Symbols(values) => values.len() as u64,
            Self::Postings(values) => values.len() as u64,
        }
    }

    fn write_to(&self, writer: &mut impl Write) -> std::io::Result<()> {
        match self {
            Self::Bytes { bytes, .. } => writer.write_all(bytes),
            Self::U64(values) => values
                .iter()
                .try_for_each(|value| writer.write_all(&value.to_le_bytes())),
            Self::Symbols(values) => values
                .iter()
                .try_for_each(|value| writer.write_all(&value.get().to_le_bytes())),
            Self::Postings(values) => values.iter().try_for_each(|value| {
                writer.write_all(&value.string_id.get().to_le_bytes())?;
                writer.write_all(&value.position.get().to_le_bytes())
            }),
        }
    }
}

/// Writes a complete immutable snapshot and atomically publishes it at `path`.
///
/// Section descriptors are derived from `sections`, so the bytes and metadata
/// cannot disagree. The temporary file and destination share a directory to
/// keep the final rename atomic.
pub(crate) fn write_file<T, C: TokenCodec<T>>(
    path: &Path,
    kind: FileKind,
    codec: &C,
    sections: &[(SectionKind, SectionData<'_>)],
) -> Result<()> {
    validate_codec_id_len(codec.id())?;
    if cfg!(target_endian = "big") {
        return Err(Error::UnsupportedHostEndianness);
    }

    let codec_offset = HEADER_LEN as u64;
    let codec_len = codec.id().len() as u64;
    let section_table_offset = align_up(codec_offset + codec_len, 8)?;
    let table_len = (sections.len() as u64)
        .checked_mul(SECTION_ENTRY_LEN as u64)
        .ok_or(Error::InvalidFile("section table length overflows"))?;
    let mut cursor = section_table_offset + table_len;
    let mut descriptors = Vec::with_capacity(sections.len());
    for (section_kind, data) in sections {
        let alignment = section_kind
            .element_layout()
            .map_or(1, |(_, alignment)| alignment as u64);
        cursor = align_up(cursor, alignment)?;
        let descriptor = SectionDescriptor {
            kind: *section_kind,
            offset: cursor,
            byte_len: data.byte_len(),
            element_count: data.element_count(),
        };
        cursor = cursor
            .checked_add(descriptor.byte_len)
            .ok_or(Error::InvalidFile("file length overflows"))?;
        descriptors.push(descriptor);
    }

    let header = DiskHeader {
        magic: MAGIC,
        format_version: U32::new(FORMAT_VERSION),
        endian_marker: U32::new(ENDIAN_MARKER),
        file_kind: U32::new(kind as u32),
        header_len: U32::new(HEADER_LEN as u32),
        codec_version: U32::new(codec.version()),
        section_count: U32::new(sections.len() as u32),
        codec_offset: U64::new(codec_offset),
        codec_len: U64::new(codec_len),
        section_table_offset: U64::new(section_table_offset),
        file_len: U64::new(cursor),
    };

    // A bare relative filename has an empty parent rather than no parent.
    // Normalize it to `.` so the durability sync opens the current directory.
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).map_err(|error| Error::io(path, error))?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        writer
            .write_all(header.as_bytes())
            .and_then(|_| writer.write_all(codec.id().as_bytes()))
            .map_err(|error| Error::io(path, error))?;
        write_padding(
            &mut writer,
            section_table_offset - codec_offset - codec_len,
            path,
        )?;
        for descriptor in &descriptors {
            let entry = DiskSectionEntry {
                kind: U32::new(descriptor.kind as u32),
                flags: U32::ZERO,
                offset: U64::new(descriptor.offset),
                byte_len: U64::new(descriptor.byte_len),
                element_count: U64::new(descriptor.element_count),
            };
            writer
                .write_all(entry.as_bytes())
                .map_err(|error| Error::io(path, error))?;
        }
        let mut written = section_table_offset + table_len;
        for ((_, data), descriptor) in sections.iter().zip(&descriptors) {
            write_padding(&mut writer, descriptor.offset - written, path)?;
            data.write_to(&mut writer)
                .map_err(|error| Error::io(path, error))?;
            written = descriptor.offset + descriptor.byte_len;
        }
        writer.flush().map_err(|error| Error::io(path, error))?;
    }
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| Error::io(path, error))?;
    temporary
        .persist(path)
        .map_err(|error| Error::io(path, error.error))?;
    sync_parent(parent, path)?;
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    let remainder = value % alignment;
    value
        .checked_add((alignment - remainder) % alignment)
        .ok_or(Error::InvalidFile("file alignment overflows"))
}

fn write_padding(writer: &mut impl Write, len: u64, path: &Path) -> Result<()> {
    const ZEROS: [u8; 64] = [0; 64];
    let mut remaining = len;
    while remaining != 0 {
        let chunk_len = usize::try_from(remaining.min(ZEROS.len() as u64)).unwrap();
        writer
            .write_all(&ZEROS[..chunk_len])
            .map_err(|error| Error::io(path, error))?;
        remaining -= chunk_len as u64;
    }
    Ok(())
}

#[cfg(unix)]
fn sync_parent(parent: &Path, path: &Path) -> Result<()> {
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| Error::io(path, error))
}

#[cfg(not(unix))]
fn sync_parent(_: &Path, _: &Path) -> Result<()> {
    Ok(())
}

impl PersistedFile {
    /// Opens a file while enforcing the mmap safety precondition on file size.
    ///
    /// Reading the fixed header first detects a recorded-length mismatch before
    /// entering the unsafe mmap boundary. It does not protect against callers
    /// modifying or truncating an already mapped snapshot; that remains part of
    /// the persistence API's safety contract.
    pub(crate) fn open<T, C: TokenCodec<T>>(
        path: &Path,
        expected_kind: FileKind,
        codec: &C,
    ) -> Result<Self> {
        validate_codec_id_len(codec.id())?;

        // Keep this `File` open through mapping so the metadata check and mmap
        // refer to the same opened file rather than two path lookups.
        let mut file = File::open(path).map_err(|error| Error::io(path, error))?;
        let mut header_bytes = [0; HEADER_LEN];
        if let Err(error) = file.read_exact(&mut header_bytes) {
            return if error.kind() == std::io::ErrorKind::UnexpectedEof {
                Err(Error::InvalidFile("header is truncated"))
            } else {
                Err(Error::io(path, error))
            };
        }
        let header = DiskHeader::ref_from_bytes(&header_bytes)
            .map_err(|_| Error::InvalidFile("header has an invalid layout"))?;
        let kind = validate_header(header, expected_kind)?;
        let header = ValidatedHeader::from_disk(header, kind);
        let file_len = file
            .metadata()
            .map_err(|error| Error::io(path, error))?
            .len();
        if header.file_len != file_len {
            return Err(Error::InvalidFile("recorded file length does not match"));
        }

        let mmap = map_file(&file, path)?;
        Self::parse_contents(mmap, header, codec.id(), codec.version())
    }

    fn parse(
        mmap: Arc<Mmap>,
        expected_kind: FileKind,
        expected_codec: &str,
        expected_codec_version: u32,
    ) -> Result<Self> {
        // Tests use anonymous mappings and therefore enter after the pre-mmap
        // phase. They still exercise the same header and contents validation.
        validate_codec_id_len(expected_codec)?;
        let bytes: &[u8] = &mmap;
        let header = DiskHeader::ref_from_prefix(bytes)
            .map_err(|_| Error::InvalidFile("header is truncated"))?
            .0;
        let kind = validate_header(header, expected_kind)?;
        if header.file_len.get() != u64::try_from(bytes.len()).unwrap() {
            return Err(Error::InvalidFile("recorded file length does not match"));
        }

        let header = ValidatedHeader::from_disk(header, kind);
        Self::parse_contents(mmap, header, expected_codec, expected_codec_version)
    }

    fn parse_contents(
        mmap: Arc<Mmap>,
        header: ValidatedHeader,
        expected_codec: &str,
        expected_codec_version: u32,
    ) -> Result<Self> {
        let bytes: &[u8] = &mmap;

        // Codec metadata is outside the section table because it is required
        // to decide whether the caller can interpret token blobs at all.
        let codec_len = header.codec_len;
        if codec_len > MAX_CODEC_ID_LEN as u64 {
            return Err(Error::InvalidFile("codec identifier is too long"));
        }
        let codec_offset = header.codec_offset;
        let codec_bytes = checked_range(bytes, codec_offset, codec_len)?;
        let actual_codec = std::str::from_utf8(codec_bytes)
            .map_err(|_| Error::InvalidFile("codec identifier is not UTF-8"))?;
        if actual_codec != expected_codec {
            return Err(Error::CodecMismatch {
                expected: expected_codec.to_owned(),
                actual: actual_codec.to_owned(),
            });
        }
        let codec_version = header.codec_version;
        if codec_version != expected_codec_version {
            return Err(Error::CodecVersionMismatch {
                expected: expected_codec_version,
                actual: codec_version,
            });
        }

        // Interpret the section table only after proving its complete byte
        // range is present. Individual payloads are validated in the loop.
        let section_count = u64::from(header.section_count);
        let table_len = section_count
            .checked_mul(SECTION_ENTRY_LEN as u64)
            .ok_or(Error::InvalidFile("section table length overflows"))?;
        let section_table_offset = header.section_table_offset;
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

        // A valid range alone is insufficient: aliases between metadata and
        // payloads could make one byte sequence acquire conflicting meanings.
        validate_non_overlapping(
            codec_offset,
            codec_len,
            section_table_offset,
            table_len,
            sections.values().copied(),
        )?;

        let file = Self {
            mmap,
            kind: header.kind,
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
        // `MappedSlice::new` is the single boundary that turns validated bytes
        // into a typed slice and keeps the mmap alive.
        let section = self.section(kind)?;
        MappedSlice::new(
            Arc::clone(&self.mmap),
            section.offset,
            section.byte_len,
            section.element_count,
        )
    }

    /// Returns a mapped symbol view after structural section validation.
    ///
    /// Symbol membership in the decoded vocabulary is a semantic property and
    /// is deliberately checked by CorpusStore when the range is accessed.
    pub(crate) fn mapped_symbols(&self, kind: SectionKind) -> Result<MappedSlice<Symbol>> {
        let values = self.mapped_slice::<DiskSymbol>(kind)?;
        // SAFETY: both types are transparent u32 wrappers and all u32 bit
        // patterns are valid. Semantic symbol validation happens lazily.
        Ok(unsafe { values.cast() })
    }

    /// Returns a mapped posting view after structural section validation.
    ///
    /// Sequence IDs, positions, ordering, and correspondence with the corpus
    /// are checked only by SearchEngine::verify.
    pub(crate) fn mapped_postings(&self, kind: SectionKind) -> Result<MappedSlice<Posting>> {
        let values = self.mapped_slice::<DiskPosting>(kind)?;
        // SAFETY: both representations are two consecutive u32 values and all
        // bit patterns are valid for SequenceId and Position.
        Ok(unsafe { values.cast() })
    }

    pub(crate) fn bytes(&self, kind: SectionKind) -> Result<&[u8]> {
        let section = self.section(kind)?;
        checked_range(&self.mmap, section.offset, section.byte_len)
    }

    fn validate_structure(&self) -> Result<()> {
        if self.kind != FileKind::SearchEngine {
            return Ok(());
        }

        // SearchEngine slicing depends only on these three offset arrays. They
        // are small relative to their payloads and are scanned fully at open.
        // Corpus symbols and postings themselves remain demand-paged and are
        // reserved for lazy or explicit semantic validation.
        let vocabulary_offsets = self.mapped_slice::<u64>(SectionKind::VocabularyTokenOffsets)?;
        let vocabulary_blob = self.section(SectionKind::VocabularyTokenBlob)?;
        let sequence_offsets = self.mapped_slice::<u64>(SectionKind::SequenceOffsets)?;
        let corpus = self.section(SectionKind::CorpusSymbols)?;
        let posting_offsets = self.mapped_slice::<u64>(SectionKind::PostingOffsets)?;
        let postings = self.section(SectionKind::Postings)?;

        validate_offsets(&vocabulary_offsets, vocabulary_blob.byte_len)?;
        validate_offsets(&sequence_offsets, corpus.element_count)?;
        validate_offsets(&posting_offsets, postings.element_count)?;

        // For token blobs, `element_count` is the number of encoded tokens,
        // while `byte_len` is the terminal byte offset. Both values are needed
        // because tokens are variable-width.
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
    // Header fields use zerocopy's explicit LE wrappers, while large payload
    // arrays intentionally use native types for direct slice access in v1.
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
    // Validate both the byte range and the typed interpretation advertised by
    // the descriptor. Blob sections only have the former invariant.
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
    // Zero-length regions occupy no bytes and may legally share an offset with
    // the following region, so they are omitted from overlap comparisons.
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
    // These conditions are exactly what later `payload[start..end]` slicing
    // needs: every pair is ordered and the entire array stays in the payload.
    if offsets.first() != Some(&0) || offsets.last() != Some(&payload_len) {
        return Err(Error::InvalidFile("offset endpoints do not match payload"));
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(Error::InvalidFile("offsets are not monotonic"));
    }
    Ok(())
}

fn checked_range(bytes: &[u8], offset: u64, len: u64) -> Result<&[u8]> {
    // Perform arithmetic in the file format's u64 domain before converting to
    // the host's usize. This distinguishes corrupt ranges from platform limits.
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
    use std::io::Write;
    use std::path::Path;
    use std::sync::Arc;

    use memmap2::MmapOptions;
    use tempfile::{NamedTempFile, tempdir};
    use zerocopy::IntoBytes;

    use super::{
        DiskHeader, DiskPosting, DiskSectionEntry, DiskSymbol, ENDIAN_MARKER, FORMAT_VERSION,
        FileKind, HEADER_LEN, MAGIC, MAX_CODEC_ID_LEN, PersistedFile, SECTION_ENTRY_LEN,
        SectionKind, U32, U64, write_padding,
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
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&fixture.bytes).unwrap();
        file.flush().unwrap();
        PersistedFile::open(file.path(), FileKind::SearchEngine, &CharCodec).unwrap();
        file.as_file_mut()
            .set_len((fixture.bytes.len() - 1) as u64)
            .unwrap();
        let result = PersistedFile::open(file.path(), FileKind::SearchEngine, &CharCodec);
        assert!(matches!(
            result,
            Err(Error::InvalidFile("recorded file length does not match"))
        ));
    }

    #[test]
    fn open_io_error_identifies_the_file() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("missing.yurine");

        assert!(matches!(
            PersistedFile::open(&path, FileKind::SearchEngine, &CharCodec),
            Err(Error::Io {
                path: actual,
                kind: std::io::ErrorKind::NotFound,
                ..
            }) if actual == path
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

    #[test]
    fn writes_padding_larger_than_the_internal_zero_buffer() {
        let mut output = Vec::new();

        write_padding(&mut output, 129, Path::new("unused")).unwrap();

        assert_eq!(output, vec![0; 129]);
    }
}
