//! Portable on-disk index writing and complete verification.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::hash::Hash;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use zerocopy::byteorder::little_endian::{U32 as LeU32, U64 as LeU64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::search::SearchEngine;
use crate::tokenization::Tokenizer;
use crate::tokenization::character::CharacterTokenizer;
use crate::tokenization::whitespace::WhitespaceTokenizer;

const FORMAT: &str = "yurine-index";
const VERSION: u32 = 1;
const BYTE_ORDER: &str = "little";
const BYTE_OFFSETS: &str = "u32";

const STRINGS: &str = "strings.utf8";
const STRING_BYTE_OFFSETS: &str = "string_byte_offsets.u64";
const SYMBOLS: &str = "symbols.u32";
const STRING_SYMBOL_OFFSETS: &str = "string_symbol_offsets.u64";
const BYTE_RANGES: &str = "byte_ranges.u32x2";
const POSTINGS: &str = "postings.u32x2";
const POSTING_OFFSETS: &str = "posting_offsets.u64";
const VOCABULARY: &str = "vocabulary.utf8";
const VOCABULARY_OFFSETS: &str = "vocabulary_offsets.u64";

#[derive(Clone, Copy)]
enum ElementCount {
    Bytes,
    StringsPlusOne,
    Tokens,
    VocabularyPlusOne,
}

#[derive(Clone, Copy)]
struct FileLayout {
    name: &'static str,
    element_type: &'static str,
    width: u64,
    elements: ElementCount,
}

const fn file(
    name: &'static str,
    element_type: &'static str,
    width: u64,
    elements: ElementCount,
) -> FileLayout {
    FileLayout {
        name,
        element_type,
        width,
        elements,
    }
}

type LePair = [LeU32; 2];

fn le_pair(first: u32, second: u32) -> LePair {
    [LeU32::new(first), LeU32::new(second)]
}

const LAYOUT: [FileLayout; 9] = [
    file(STRINGS, "u8", 1, ElementCount::Bytes),
    file(STRING_BYTE_OFFSETS, "u64", 8, ElementCount::StringsPlusOne),
    file(SYMBOLS, "u32", 4, ElementCount::Tokens),
    file(
        STRING_SYMBOL_OFFSETS,
        "u64",
        8,
        ElementCount::StringsPlusOne,
    ),
    file(BYTE_RANGES, "u32x2", 8, ElementCount::Tokens),
    file(POSTINGS, "u32x2", 8, ElementCount::Tokens),
    file(POSTING_OFFSETS, "u64", 8, ElementCount::VocabularyPlusOne),
    file(VOCABULARY, "u8", 1, ElementCount::Bytes),
    file(
        VOCABULARY_OFFSETS,
        "u64",
        8,
        ElementCount::VocabularyPlusOne,
    ),
];

/// The versioned metadata stored in `manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexManifest {
    pub format: String,
    pub version: u32,
    pub byte_order: String,
    pub byte_offsets: String,
    pub tokenizer: TokenizerManifest,
    pub strings: u64,
    pub tokens: u64,
    pub vocabulary: u64,
    pub files: BTreeMap<String, FileManifest>,
}

/// Tokenizer identity recorded in an index manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerManifest {
    #[serde(rename = "type")]
    pub r#type: String,
    pub version: u32,
}

/// Metadata for one immutable index data file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileManifest {
    pub element_type: String,
    pub elements: u64,
    pub bytes: u64,
    pub sha256: String,
}

/// Errors produced while writing or verifying an index directory.
#[derive(Debug, Error)]
pub enum IndexError {
    #[error("output index already exists: {0}")]
    OutputExists(PathBuf),
    #[error("failed to access index file {file}: {source}")]
    FileIo {
        file: String,
        #[source]
        source: io::Error,
    },
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),
    #[error("unsupported {field}: {value}")]
    Unsupported { field: &'static str, value: String },
    #[error("invalid index data in {file}: {reason}")]
    InvalidData { file: String, reason: String },
}

impl<C> SearchEngine<CharacterTokenizer, C> {
    /// Writes this character-tokenized index to a new immutable directory.
    pub fn write_index(&self, output: impl AsRef<Path>) -> Result<IndexManifest, IndexError> {
        write_engine(self, output.as_ref(), "character", |token| {
            token.to_string()
        })
    }
}

impl<C> SearchEngine<WhitespaceTokenizer, C> {
    /// Writes this whitespace-tokenized index to a new immutable directory.
    pub fn write_index(&self, output: impl AsRef<Path>) -> Result<IndexManifest, IndexError> {
        write_engine(self, output.as_ref(), "whitespace", Clone::clone)
    }
}

fn write_engine<T, C, F>(
    engine: &SearchEngine<T, C>,
    output: &Path,
    tokenizer: &str,
    token_text: F,
) -> Result<IndexManifest, IndexError>
where
    T: Tokenizer,
    T::Token: Eq + Hash,
    F: Fn(&T::Token) -> String,
{
    if output
        .try_exists()
        .map_err(|source| file_io(output, source))?
    {
        return Err(IndexError::OutputExists(output.to_owned()));
    }

    let (vocabulary, index, store) = engine.index_parts();
    let counts = IndexCounts {
        strings: store.len() as u64,
        tokens: store.symbols().len() as u64,
        vocabulary: vocabulary.len() as u64,
    };
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let staging = tempfile::Builder::new()
        .prefix(".yurine-index-")
        .tempdir_in(parent)
        .map_err(|source| file_io(parent, source))?;

    let mut file_manifests = BTreeMap::new();
    let mut add = |name: &str, result| -> Result<(), IndexError> {
        file_manifests.insert(name.to_owned(), result?);
        Ok(())
    };
    add(
        STRINGS,
        write_data_file(
            staging.path(),
            STRINGS,
            store.strings().len() as u64,
            |writer| writer.write_all(store.strings()),
        ),
    )?;
    add(
        STRING_BYTE_OFFSETS,
        write_records(
            staging.path(),
            STRING_BYTE_OFFSETS,
            store.string_byte_offsets().iter().copied().map(LeU64::new),
        ),
    )?;
    add(
        SYMBOLS,
        write_records(
            staging.path(),
            SYMBOLS,
            store
                .symbols()
                .iter()
                .map(|symbol| LeU32::new(symbol.get())),
        ),
    )?;
    add(
        STRING_SYMBOL_OFFSETS,
        write_records(
            staging.path(),
            STRING_SYMBOL_OFFSETS,
            store.string_offsets().iter().copied().map(LeU64::new),
        ),
    )?;
    add(
        BYTE_RANGES,
        write_records(
            staging.path(),
            BYTE_RANGES,
            store
                .byte_ranges()
                .iter()
                .map(|range| le_pair(range.start().get(), range.end().get())),
        ),
    )?;
    add(
        POSTINGS,
        write_records(
            staging.path(),
            POSTINGS,
            index
                .raw_postings()
                .iter()
                .map(|posting| le_pair(posting.string_id.get(), posting.position.get())),
        ),
    )?;
    add(
        POSTING_OFFSETS,
        write_records(
            staging.path(),
            POSTING_OFFSETS,
            index.posting_offsets().iter().copied().map(LeU64::new),
        ),
    )?;

    add(
        VOCABULARY,
        write_data_file(staging.path(), VOCABULARY, 0, |writer| {
            for token in vocabulary.tokens() {
                let token = token_text(token);
                writer.write_all(token.as_bytes())?;
            }
            Ok(())
        }),
    )?;
    add(
        VOCABULARY_OFFSETS,
        write_data_file(
            staging.path(),
            VOCABULARY_OFFSETS,
            counts.vocabulary + 1,
            |writer| {
                writer.write_all(LeU64::ZERO.as_bytes())?;
                let mut offset = 0u64;
                for token in vocabulary.tokens() {
                    offset += token_text(token).len() as u64;
                    writer.write_all(LeU64::new(offset).as_bytes())?;
                }
                Ok(())
            },
        ),
    )?;

    let manifest = IndexManifest {
        format: FORMAT.to_owned(),
        version: VERSION,
        byte_order: BYTE_ORDER.to_owned(),
        byte_offsets: BYTE_OFFSETS.to_owned(),
        tokenizer: TokenizerManifest {
            r#type: tokenizer.to_owned(),
            version: 1,
        },
        strings: counts.strings,
        tokens: counts.tokens,
        vocabulary: counts.vocabulary,
        files: file_manifests,
    };

    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| IndexError::InvalidManifest(error.to_string()))?;
    manifest_bytes.push(b'\n');
    write_synced_file(&staging.path().join("manifest.json"), &manifest_bytes)?;

    let verified = verify_index(staging.path())?;
    sync_directory(staging.path())?;
    fs::rename(staging.path(), output).map_err(|source| file_io(output, source))?;
    sync_directory(parent)?;
    Ok(verified)
}

/// Completely verifies an index directory and returns its manifest.
pub fn verify_index(directory: impl AsRef<Path>) -> Result<IndexManifest, IndexError> {
    let directory = directory.as_ref();
    let manifest_path = directory.join("manifest.json");
    let manifest_bytes = read_file(&manifest_path)?;
    let manifest: IndexManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| IndexError::InvalidManifest(error.to_string()))?;
    validate_manifest(&manifest)?;

    let mut files = BTreeMap::new();
    for layout in LAYOUT {
        let metadata = &manifest.files[layout.name];
        validate_file_metadata(layout, metadata, counts(&manifest))?;
        let bytes = read_file(&directory.join(layout.name))?;
        let actual_bytes = bytes.len() as u64;
        if actual_bytes != metadata.bytes {
            return invalid(
                layout.name,
                format!("has {actual_bytes} bytes; expected {}", metadata.bytes),
            );
        }
        let actual_checksum = checksum(&bytes);
        if actual_checksum != metadata.sha256 {
            return invalid(
                layout.name,
                format!(
                    "SHA-256 mismatch: expected {}, got {actual_checksum}",
                    metadata.sha256
                ),
            );
        }
        files.insert(layout.name, bytes);
    }

    verify_data(&manifest, &files)?;
    Ok(manifest)
}

fn validate_manifest(manifest: &IndexManifest) -> Result<(), IndexError> {
    if manifest.format != FORMAT {
        return unsupported("index format", &manifest.format);
    }
    if manifest.version != VERSION {
        return unsupported("index version", manifest.version);
    }
    if manifest.byte_order != BYTE_ORDER {
        return unsupported("byte order", &manifest.byte_order);
    }
    if manifest.byte_offsets != BYTE_OFFSETS {
        return unsupported("byte-offset type", &manifest.byte_offsets);
    }
    if !matches!(
        (
            manifest.tokenizer.r#type.as_str(),
            manifest.tokenizer.version,
        ),
        ("character" | "whitespace", 1)
    ) {
        return unsupported(
            "tokenizer",
            format!(
                "{} version {}",
                manifest.tokenizer.r#type, manifest.tokenizer.version
            ),
        );
    }
    for layout in LAYOUT {
        if !manifest.files.contains_key(layout.name) {
            return invalid(
                "manifest.json",
                format!("missing metadata for {}", layout.name),
            );
        }
    }
    if let Some(name) = manifest
        .files
        .keys()
        .find(|name| !LAYOUT.iter().any(|layout| layout.name == name.as_str()))
    {
        return invalid("manifest.json", format!("unexpected metadata for {name}"));
    }
    Ok(())
}

fn validate_file_metadata(
    layout: FileLayout,
    metadata: &FileManifest,
    counts: IndexCounts,
) -> Result<(), IndexError> {
    let elements = match layout.elements {
        ElementCount::Bytes => metadata.bytes,
        ElementCount::StringsPlusOne => counts
            .strings
            .checked_add(1)
            .ok_or_else(|| invalid_error("manifest.json", "string count is too large"))?,
        ElementCount::Tokens => counts.tokens,
        ElementCount::VocabularyPlusOne => counts
            .vocabulary
            .checked_add(1)
            .ok_or_else(|| invalid_error("manifest.json", "vocabulary count is too large"))?,
    };
    if metadata.element_type != layout.element_type {
        return invalid(
            layout.name,
            format!("element type must be {}", layout.element_type),
        );
    }
    if metadata.elements != elements {
        return invalid(
            layout.name,
            format!(
                "element count is {}; expected {elements}",
                metadata.elements
            ),
        );
    }
    let expected_bytes = elements
        .checked_mul(layout.width)
        .ok_or_else(|| invalid_error(layout.name, "byte length overflows u64"))?;
    if metadata.bytes != expected_bytes {
        return invalid(
            layout.name,
            format!(
                "manifest byte length is {}; expected {expected_bytes}",
                metadata.bytes
            ),
        );
    }
    if metadata.sha256.len() != 64 || !metadata.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return invalid(layout.name, "SHA-256 must contain 64 hexadecimal digits");
    }
    Ok(())
}

fn verify_data(
    manifest: &IndexManifest,
    files: &BTreeMap<&str, Vec<u8>>,
) -> Result<(), IndexError> {
    let strings = utf8(STRINGS, &files[STRINGS])?;
    let string_byte_offsets = decode::<LeU64>(STRING_BYTE_OFFSETS, &files[STRING_BYTE_OFFSETS])?;
    let symbols = decode::<LeU32>(SYMBOLS, &files[SYMBOLS])?;
    let string_symbol_offsets =
        decode::<LeU64>(STRING_SYMBOL_OFFSETS, &files[STRING_SYMBOL_OFFSETS])?;
    let byte_ranges = decode::<LePair>(BYTE_RANGES, &files[BYTE_RANGES])?;
    let postings = decode::<LePair>(POSTINGS, &files[POSTINGS])?;
    let posting_offsets = decode::<LeU64>(POSTING_OFFSETS, &files[POSTING_OFFSETS])?;
    let vocabulary = utf8(VOCABULARY, &files[VOCABULARY])?;
    let vocabulary_offsets = decode::<LeU64>(VOCABULARY_OFFSETS, &files[VOCABULARY_OFFSETS])?;

    verify_offsets(STRING_BYTE_OFFSETS, string_byte_offsets, strings.len())?;
    verify_offsets(STRING_SYMBOL_OFFSETS, string_symbol_offsets, symbols.len())?;
    verify_offsets(POSTING_OFFSETS, posting_offsets, postings.len())?;
    verify_offsets(VOCABULARY_OFFSETS, vocabulary_offsets, vocabulary.len())?;

    let vocabulary_count = usize_count("manifest.json", manifest.vocabulary)?;
    for (index, symbol) in symbols.iter().enumerate() {
        if symbol.get() as usize >= vocabulary_count {
            return invalid(
                SYMBOLS,
                format!(
                    "symbol {} at element {index} is outside the vocabulary",
                    symbol.get()
                ),
            );
        }
    }

    let mut unique = HashSet::with_capacity(vocabulary_count);
    for index in 0..vocabulary_count {
        let token = slice(
            VOCABULARY,
            vocabulary,
            vocabulary_offsets[index].get(),
            vocabulary_offsets[index + 1].get(),
        )?;
        if !unique.insert(token) {
            return invalid(VOCABULARY, format!("duplicate token at element {index}"));
        }
    }

    let string_count = usize_count("manifest.json", manifest.strings)?;
    for string_index in 0..string_count {
        let string = slice(
            STRINGS,
            strings,
            string_byte_offsets[string_index].get(),
            string_byte_offsets[string_index + 1].get(),
        )?;
        let symbol_start = offset(
            STRING_SYMBOL_OFFSETS,
            string_symbol_offsets[string_index].get(),
        )?;
        let symbol_end = offset(
            STRING_SYMBOL_OFFSETS,
            string_symbol_offsets[string_index + 1].get(),
        )?;
        for (position, range) in byte_ranges[symbol_start..symbol_end].iter().enumerate() {
            let (start, end) = (range[0].get(), range[1].get());
            if start > end || string.get(start as usize..end as usize).is_none() {
                return invalid(
                    BYTE_RANGES,
                    format!(
                        "range {start}..{end} at element {} is invalid for string {string_index}",
                        symbol_start + position
                    ),
                );
            }
        }
    }

    for symbol in 0..vocabulary_count {
        let start = offset(POSTING_OFFSETS, posting_offsets[symbol].get())?;
        let end = offset(POSTING_OFFSETS, posting_offsets[symbol + 1].get())?;
        let mut previous = None;
        for (posting_index, posting) in postings[start..end].iter().enumerate() {
            let (string_id, position) = (posting[0].get(), posting[1].get());
            let pair = (string_id, position);
            if previous.is_some_and(|value| value >= pair) {
                return invalid(
                    POSTINGS,
                    format!(
                        "postings for symbol {symbol} are not strictly ordered at element {}",
                        start + posting_index
                    ),
                );
            }
            previous = Some(pair);
            let string_id = string_id as usize;
            if string_id >= string_count {
                return invalid(
                    POSTINGS,
                    format!(
                        "string ID {string_id} at element {} is out of range",
                        start + posting_index
                    ),
                );
            }
            let string_start = offset(
                STRING_SYMBOL_OFFSETS,
                string_symbol_offsets[string_id].get(),
            )?;
            let string_end = offset(
                STRING_SYMBOL_OFFSETS,
                string_symbol_offsets[string_id + 1].get(),
            )?;
            let corpus_index = string_start + position as usize;
            if corpus_index >= string_end {
                return invalid(
                    POSTINGS,
                    format!(
                        "position {position} at element {} is out of range for string {string_id}",
                        start + posting_index
                    ),
                );
            }
            if symbols[corpus_index].get() as usize != symbol {
                return invalid(
                    POSTINGS,
                    format!(
                        "element {} points to a different symbol",
                        start + posting_index
                    ),
                );
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct IndexCounts {
    strings: u64,
    tokens: u64,
    vocabulary: u64,
}

fn counts(manifest: &IndexManifest) -> IndexCounts {
    IndexCounts {
        strings: manifest.strings,
        tokens: manifest.tokens,
        vocabulary: manifest.vocabulary,
    }
}

struct ChecksummedWriter {
    inner: BufWriter<File>,
    hasher: Sha256,
    bytes: u64,
}

impl ChecksummedWriter {
    fn create(path: &Path) -> Result<Self, IndexError> {
        let file = File::create(path).map_err(|source| file_io(path, source))?;
        Ok(Self {
            inner: BufWriter::new(file),
            hasher: Sha256::new(),
            bytes: 0,
        })
    }

    fn finish(mut self, path: &Path) -> Result<(u64, String), IndexError> {
        self.inner.flush().map_err(|source| file_io(path, source))?;
        self.inner
            .get_ref()
            .sync_all()
            .map_err(|source| file_io(path, source))?;
        Ok((self.bytes, format!("{:x}", self.hasher.finalize())))
    }
}

impl Write for ChecksummedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(buffer)?;
        self.hasher.update(&buffer[..written]);
        self.bytes += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn write_data_file(
    directory: &Path,
    name: &str,
    elements: u64,
    write: impl FnOnce(&mut ChecksummedWriter) -> io::Result<()>,
) -> Result<FileManifest, IndexError> {
    let layout = find_layout(name);
    let path = directory.join(name);
    let mut writer = ChecksummedWriter::create(&path)?;
    write(&mut writer).map_err(|source| file_io(&path, source))?;
    let (bytes, sha256) = writer.finish(&path)?;
    let elements = match layout.elements {
        ElementCount::Bytes => bytes,
        _ => elements,
    };
    Ok(FileManifest {
        element_type: layout.element_type.to_owned(),
        elements,
        bytes,
        sha256,
    })
}

fn write_records<T, I>(directory: &Path, name: &str, values: I) -> Result<FileManifest, IndexError>
where
    T: IntoBytes + Immutable,
    I: IntoIterator<Item = T>,
    I::IntoIter: ExactSizeIterator,
{
    let values = values.into_iter();
    let elements = values.len() as u64;
    write_data_file(directory, name, elements, |writer| {
        for value in values {
            writer.write_all(value.as_bytes())?;
        }
        Ok(())
    })
}

fn find_layout(name: &str) -> FileLayout {
    LAYOUT
        .iter()
        .copied()
        .find(|layout| layout.name == name)
        .expect("internal index file must have a layout")
}

fn decode<'a, T>(name: &str, bytes: &'a [u8]) -> Result<&'a [T], IndexError>
where
    T: FromBytes + KnownLayout + Immutable,
{
    <[T]>::ref_from_bytes(bytes)
        .map_err(|error| invalid_error(name, format!("invalid record layout: {error}")))
}

fn utf8<'a>(name: &str, bytes: &'a [u8]) -> Result<&'a str, IndexError> {
    std::str::from_utf8(bytes).map_err(|error| invalid_error(name, format!("not UTF-8: {error}")))
}

fn slice<'a>(name: &str, text: &'a str, start: u64, end: u64) -> Result<&'a str, IndexError> {
    let start = offset(name, start)?;
    let end = offset(name, end)?;
    text.get(start..end)
        .ok_or_else(|| invalid_error(name, format!("invalid UTF-8 range {start}..{end}")))
}

fn verify_offsets(name: &str, offsets: &[LeU64], expected_end: usize) -> Result<(), IndexError> {
    if offsets[0].get() != 0 {
        return invalid(name, "first offset must be zero");
    }
    if let Some(index) =
        (1..offsets.len()).find(|&index| offsets[index - 1].get() > offsets[index].get())
    {
        return invalid(name, format!("offsets decrease at element {}", index + 1));
    }
    if offsets[offsets.len() - 1].get() != expected_end as u64 {
        return invalid(name, format!("final offset must be {expected_end}"));
    }
    Ok(())
}

fn usize_count(file: &str, value: u64) -> Result<usize, IndexError> {
    usize::try_from(value).map_err(|_| invalid_error(file, "count does not fit in memory"))
}

fn offset(file: &str, value: u64) -> Result<usize, IndexError> {
    usize::try_from(value).map_err(|_| invalid_error(file, "offset does not fit in memory"))
}

fn checksum(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_file(path: &Path) -> Result<Vec<u8>, IndexError> {
    fs::read(path).map_err(|source| file_io(path, source))
}

fn write_synced_file(path: &Path, bytes: &[u8]) -> Result<(), IndexError> {
    let mut file = File::create(path).map_err(|source| file_io(path, source))?;
    file.write_all(bytes)
        .map_err(|source| file_io(path, source))?;
    file.sync_all().map_err(|source| file_io(path, source))
}

fn sync_directory(path: &Path) -> Result<(), IndexError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| file_io(path, source))
}

fn file_io(path: &Path, source: io::Error) -> IndexError {
    IndexError::FileIo {
        file: path.display().to_string(),
        source,
    }
}

fn invalid<T>(file: &str, reason: impl Into<String>) -> Result<T, IndexError> {
    Err(invalid_error(file, reason))
}

fn unsupported<T>(field: &'static str, value: impl ToString) -> Result<T, IndexError> {
    Err(IndexError::Unsupported {
        field,
        value: value.to_string(),
    })
}

fn invalid_error(file: &str, reason: impl Into<String>) -> IndexError {
    IndexError::InvalidData {
        file: file.to_owned(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::Value;

    use super::{IndexError, IndexManifest, checksum, verify_index};
    use crate::costs::levenshtein::LevenshteinCosts;
    use crate::search::SearchEngineBuilder;
    use crate::tokenization::character::CharacterTokenizer;
    use crate::tokenization::whitespace::WhitespaceTokenizer;

    #[test]
    fn writes_and_verifies_character_index() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("corpus.yurine");
        let mut builder =
            SearchEngineBuilder::new(CharacterTokenizer::new(), LevenshteinCosts::new());
        builder.add_string("a東京a").unwrap();
        builder.add_string("").unwrap();
        let engine = builder.build().unwrap();

        let manifest = engine.write_index(&output).unwrap();

        assert_eq!(manifest.strings, 2);
        assert_eq!(manifest.tokens, 4);
        assert_eq!(manifest.vocabulary, 3);
        assert_eq!(verify_index(&output).unwrap(), manifest);
        assert_eq!(
            fs::read(output.join("strings.utf8")).unwrap(),
            "a東京a".as_bytes()
        );
        assert_eq!(
            fs::read(output.join("string_byte_offsets.u64")).unwrap(),
            [0u64, 8, 8]
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn writes_and_verifies_whitespace_index() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("corpus.yurine");
        let mut builder =
            SearchEngineBuilder::new(WhitespaceTokenizer::new(), LevenshteinCosts::new());
        builder.add_string("東京  京都\n東京").unwrap();
        let engine = builder.build().unwrap();

        let manifest = engine.write_index(&output).unwrap();

        assert_eq!(manifest.tokenizer.r#type, "whitespace");
        assert_eq!(manifest.tokens, 3);
        verify_index(output).unwrap();
    }

    #[test]
    fn writes_and_verifies_empty_index() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("empty.yurine");
        let engine = SearchEngineBuilder::new(CharacterTokenizer::new(), LevenshteinCosts::new())
            .build()
            .unwrap();

        let manifest = engine.write_index(&output).unwrap();

        assert_eq!(
            (manifest.strings, manifest.tokens, manifest.vocabulary),
            (0, 0, 0)
        );
        assert!(fs::read(output.join("strings.utf8")).unwrap().is_empty());
        assert_eq!(
            fs::read(output.join("posting_offsets.u64")).unwrap(),
            0u64.to_le_bytes()
        );
        verify_index(output).unwrap();
    }

    #[test]
    fn rejects_missing_truncated_and_checksum_mismatched_files() {
        let directory = tempfile::tempdir().unwrap();
        let output = write_fixture(directory.path());
        fs::remove_file(output.join("symbols.u32")).unwrap();
        assert!(matches!(
            verify_index(&output),
            Err(IndexError::FileIo { file, source })
                if file.ends_with("symbols.u32") && source.kind() == std::io::ErrorKind::NotFound
        ));

        let output = write_fixture(directory.path());
        fs::write(output.join("symbols.u32"), [0, 0]).unwrap();
        assert!(matches!(
            verify_index(&output),
            Err(IndexError::InvalidData { file, .. }) if file == "symbols.u32"
        ));

        let output = write_fixture(directory.path());
        let mut symbols = fs::read(output.join("symbols.u32")).unwrap();
        symbols[0] ^= 1;
        fs::write(output.join("symbols.u32"), symbols).unwrap();
        assert!(matches!(
            verify_index(&output),
            Err(IndexError::InvalidData { file, .. }) if file == "symbols.u32"
        ));
    }

    #[test]
    fn rejects_unsupported_manifest_values() {
        type Case = (fn(&mut Value), &'static str);

        let directory = tempfile::tempdir().unwrap();
        let cases: [Case; 7] = [
            (
                |value| value["version"] = 2.into(),
                "unsupported index version: 2",
            ),
            (
                |value| value["byte_order"] = "big".into(),
                "unsupported byte order: big",
            ),
            (
                |value| value["format"] = "other".into(),
                "unsupported index format: other",
            ),
            (
                |value| value["byte_offsets"] = "u64".into(),
                "unsupported byte-offset type: u64",
            ),
            (
                |value| value["tokenizer"]["type"] = "custom".into(),
                "unsupported tokenizer: custom version 1",
            ),
            (
                |value| {
                    value["files"]
                        .as_object_mut()
                        .unwrap()
                        .remove("symbols.u32");
                },
                "invalid index data in manifest.json: missing metadata for symbols.u32",
            ),
            (
                |value| value["files"]["extra.u32"] = value["files"]["symbols.u32"].clone(),
                "invalid index data in manifest.json: unexpected metadata for extra.u32",
            ),
        ];

        for (update, expected) in cases {
            let output = write_fixture(directory.path());
            update_manifest(&output, update);
            assert_eq!(verify_index(output).unwrap_err().to_string(), expected);
        }
    }

    #[test]
    fn fully_verifies_all_data_files() {
        type Corruption = (&'static str, fn(&mut [u8]));

        let directory = tempfile::tempdir().unwrap();
        let cases: [Corruption; 9] = [
            ("strings.utf8", |bytes| bytes[0] = 0xff),
            ("string_byte_offsets.u64", |bytes| {
                bytes[8..16].copy_from_slice(&3u64.to_le_bytes())
            }),
            ("symbols.u32", |bytes| {
                bytes[..4].copy_from_slice(&99u32.to_le_bytes())
            }),
            ("string_symbol_offsets.u64", |bytes| {
                bytes[8..16].copy_from_slice(&3u64.to_le_bytes())
            }),
            ("byte_ranges.u32x2", |bytes| {
                bytes[4..8].copy_from_slice(&3u32.to_le_bytes())
            }),
            ("postings.u32x2", |bytes| {
                bytes[4..8].copy_from_slice(&9u32.to_le_bytes())
            }),
            ("posting_offsets.u64", |bytes| {
                bytes[16..24].copy_from_slice(&1u64.to_le_bytes())
            }),
            ("vocabulary.utf8", |bytes| bytes[1] = bytes[0]),
            ("vocabulary_offsets.u64", |bytes| {
                bytes[16..24].copy_from_slice(&1u64.to_le_bytes())
            }),
        ];

        for (file, corrupt) in cases {
            let output = write_fixture(directory.path());
            rewrite_file(&output, file, corrupt);
            assert!(matches!(
                verify_index(output),
                Err(IndexError::InvalidData { file: reported, .. }) if reported == file
            ));
        }
    }

    #[test]
    fn does_not_replace_an_existing_directory() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("corpus.yurine");
        fs::create_dir(&output).unwrap();
        fs::write(output.join("sentinel"), b"keep").unwrap();
        let engine = SearchEngineBuilder::new(CharacterTokenizer::new(), LevenshteinCosts::new())
            .build()
            .unwrap();

        assert!(matches!(
            engine.write_index(&output),
            Err(IndexError::OutputExists(path)) if path == output
        ));
        assert_eq!(fs::read(output.join("sentinel")).unwrap(), b"keep");
    }

    fn write_fixture(parent: &std::path::Path) -> std::path::PathBuf {
        let output = parent.join(format!("{}.yurine", uuid()));
        let mut builder =
            SearchEngineBuilder::new(CharacterTokenizer::new(), LevenshteinCosts::new());
        builder.add_string("ab").unwrap();
        builder.build().unwrap().write_index(&output).unwrap();
        output
    }

    fn update_manifest(output: &std::path::Path, update: impl FnOnce(&mut Value)) {
        let path = output.join("manifest.json");
        let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        update(&mut manifest);
        fs::write(path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    }

    fn rewrite_file(output: &std::path::Path, name: &str, update: impl FnOnce(&mut [u8])) {
        let path = output.join(name);
        let mut bytes = fs::read(&path).unwrap();
        update(&mut bytes);
        fs::write(path, &bytes).unwrap();

        let manifest_path = output.join("manifest.json");
        let mut manifest: IndexManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest.files.get_mut(name).unwrap().sha256 = checksum(&bytes);
        fs::write(manifest_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    }

    fn uuid() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        NEXT.fetch_add(1, Ordering::Relaxed).to_string()
    }
}
