//! Portable on-disk index writing and complete verification.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::hash::Hash;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

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

const FILE_NAMES: [&str; 9] = [
    STRINGS,
    STRING_BYTE_OFFSETS,
    SYMBOLS,
    STRING_SYMBOL_OFFSETS,
    BYTE_RANGES,
    POSTINGS,
    POSTING_OFFSETS,
    VOCABULARY,
    VOCABULARY_OFFSETS,
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
    #[error("unsupported index format: {0}")]
    UnsupportedFormat(String),
    #[error("unsupported index version: {0}")]
    UnsupportedVersion(u32),
    #[error("unsupported byte order: {0}")]
    UnsupportedByteOrder(String),
    #[error("unsupported byte-offset type: {0}")]
    UnsupportedByteOffsets(String),
    #[error("unsupported tokenizer: {kind} version {version}")]
    UnsupportedTokenizer { kind: String, version: u32 },
    #[error("missing metadata for index file {0}")]
    MissingFileMetadata(String),
    #[error("unexpected metadata for index file {0}")]
    UnexpectedFileMetadata(String),
    #[error("index file {file} has {actual} bytes; expected {expected}")]
    FileLength {
        file: String,
        expected: u64,
        actual: u64,
    },
    #[error("SHA-256 mismatch for index file {file}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        file: String,
        expected: String,
        actual: String,
    },
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
    let mut files = BTreeMap::new();
    files.insert(STRINGS.to_owned(), store.strings().to_vec());
    files.insert(
        STRING_BYTE_OFFSETS.to_owned(),
        encode_u64(store.string_byte_offsets().iter().copied()),
    );
    files.insert(
        SYMBOLS.to_owned(),
        encode_u32(store.symbols().iter().map(|symbol| symbol.get())),
    );
    files.insert(
        STRING_SYMBOL_OFFSETS.to_owned(),
        encode_u64(store.string_offsets().iter().copied()),
    );
    files.insert(
        BYTE_RANGES.to_owned(),
        encode_pairs(
            store
                .byte_ranges()
                .iter()
                .map(|range| (range.start().get(), range.end().get())),
        ),
    );
    files.insert(
        POSTINGS.to_owned(),
        encode_pairs(
            index
                .raw_postings()
                .iter()
                .map(|posting| (posting.string_id.get(), posting.position.get())),
        ),
    );
    files.insert(
        POSTING_OFFSETS.to_owned(),
        encode_u64(index.posting_offsets().iter().copied()),
    );

    let mut vocabulary_bytes = Vec::new();
    let mut vocabulary_offsets = vec![0u64];
    for token in vocabulary.tokens() {
        vocabulary_bytes.extend_from_slice(token_text(token).as_bytes());
        vocabulary_offsets.push(vocabulary_bytes.len() as u64);
    }
    files.insert(VOCABULARY.to_owned(), vocabulary_bytes);
    files.insert(
        VOCABULARY_OFFSETS.to_owned(),
        encode_u64(vocabulary_offsets),
    );

    let counts = IndexCounts {
        strings: store.len() as u64,
        tokens: store.symbols().len() as u64,
        vocabulary: vocabulary.len() as u64,
    };
    let mut file_manifests = BTreeMap::new();
    for (name, bytes) in &files {
        file_manifests.insert(name.clone(), describe_file(name, bytes, counts));
    }
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

    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let staging = tempfile::Builder::new()
        .prefix(".yurine-index-")
        .tempdir_in(parent)
        .map_err(|source| file_io(parent, source))?;
    for (name, bytes) in files {
        write_file(&staging.path().join(name), &bytes)?;
    }
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| IndexError::InvalidManifest(error.to_string()))?;
    manifest_bytes.push(b'\n');
    write_file(&staging.path().join("manifest.json"), &manifest_bytes)?;

    let verified = verify_index(staging.path())?;
    fs::rename(staging.path(), output).map_err(|source| file_io(output, source))?;
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
    for name in FILE_NAMES {
        let metadata = &manifest.files[name];
        validate_file_metadata(name, metadata, counts(&manifest))?;
        let bytes = read_file(&directory.join(name))?;
        let actual_bytes = bytes.len() as u64;
        if actual_bytes != metadata.bytes {
            return Err(IndexError::FileLength {
                file: name.to_owned(),
                expected: metadata.bytes,
                actual: actual_bytes,
            });
        }
        let actual_checksum = checksum(&bytes);
        if actual_checksum != metadata.sha256 {
            return Err(IndexError::ChecksumMismatch {
                file: name.to_owned(),
                expected: metadata.sha256.clone(),
                actual: actual_checksum,
            });
        }
        files.insert(name, bytes);
    }

    verify_data(&manifest, &files)?;
    Ok(manifest)
}

fn validate_manifest(manifest: &IndexManifest) -> Result<(), IndexError> {
    if manifest.format != FORMAT {
        return Err(IndexError::UnsupportedFormat(manifest.format.clone()));
    }
    if manifest.version != VERSION {
        return Err(IndexError::UnsupportedVersion(manifest.version));
    }
    if manifest.byte_order != BYTE_ORDER {
        return Err(IndexError::UnsupportedByteOrder(
            manifest.byte_order.clone(),
        ));
    }
    if manifest.byte_offsets != BYTE_OFFSETS {
        return Err(IndexError::UnsupportedByteOffsets(
            manifest.byte_offsets.clone(),
        ));
    }
    if manifest.tokenizer.version != 1
        || !matches!(
            manifest.tokenizer.r#type.as_str(),
            "character" | "whitespace"
        )
    {
        return Err(IndexError::UnsupportedTokenizer {
            kind: manifest.tokenizer.r#type.clone(),
            version: manifest.tokenizer.version,
        });
    }
    for name in FILE_NAMES {
        if !manifest.files.contains_key(name) {
            return Err(IndexError::MissingFileMetadata(name.to_owned()));
        }
    }
    if let Some(name) = manifest
        .files
        .keys()
        .find(|name| !FILE_NAMES.contains(&name.as_str()))
    {
        return Err(IndexError::UnexpectedFileMetadata(name.clone()));
    }
    Ok(())
}

fn validate_file_metadata(
    name: &str,
    metadata: &FileManifest,
    counts: IndexCounts,
) -> Result<(), IndexError> {
    let string_offsets = counts
        .strings
        .checked_add(1)
        .ok_or_else(|| invalid_error("manifest.json", "string count overflows u64"))?;
    let vocabulary_offsets = counts
        .vocabulary
        .checked_add(1)
        .ok_or_else(|| invalid_error("manifest.json", "vocabulary count overflows u64"))?;
    let (element_type, elements, width) = match name {
        STRINGS | VOCABULARY => ("u8", metadata.bytes, 1),
        STRING_BYTE_OFFSETS => ("u64", string_offsets, 8),
        SYMBOLS => ("u32", counts.tokens, 4),
        STRING_SYMBOL_OFFSETS => ("u64", string_offsets, 8),
        BYTE_RANGES => ("u32x2", counts.tokens, 8),
        POSTINGS => ("u32x2", counts.tokens, 8),
        POSTING_OFFSETS => ("u64", vocabulary_offsets, 8),
        VOCABULARY_OFFSETS => ("u64", vocabulary_offsets, 8),
        _ => unreachable!(),
    };
    if metadata.element_type != element_type {
        return invalid(name, format!("element type must be {element_type}"));
    }
    if metadata.elements != elements {
        return invalid(
            name,
            format!(
                "element count is {}; expected {elements}",
                metadata.elements
            ),
        );
    }
    let expected_bytes = elements
        .checked_mul(width)
        .ok_or_else(|| invalid_error(name, "byte length overflows u64"))?;
    if metadata.bytes != expected_bytes {
        return invalid(
            name,
            format!(
                "manifest byte length is {}; expected {expected_bytes}",
                metadata.bytes
            ),
        );
    }
    if metadata.sha256.len() != 64 || !metadata.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return invalid(name, "SHA-256 must contain 64 hexadecimal digits");
    }
    Ok(())
}

fn verify_data(
    manifest: &IndexManifest,
    files: &BTreeMap<&str, Vec<u8>>,
) -> Result<(), IndexError> {
    let strings = std::str::from_utf8(&files[STRINGS])
        .map_err(|error| invalid_error(STRINGS, format!("not UTF-8: {error}")))?;
    let string_byte_offsets = decode_u64(STRING_BYTE_OFFSETS, &files[STRING_BYTE_OFFSETS])?;
    let symbols = decode_u32(SYMBOLS, &files[SYMBOLS])?;
    let string_symbol_offsets = decode_u64(STRING_SYMBOL_OFFSETS, &files[STRING_SYMBOL_OFFSETS])?;
    let byte_ranges = decode_pairs(BYTE_RANGES, &files[BYTE_RANGES])?;
    let postings = decode_pairs(POSTINGS, &files[POSTINGS])?;
    let posting_offsets = decode_u64(POSTING_OFFSETS, &files[POSTING_OFFSETS])?;
    let vocabulary = std::str::from_utf8(&files[VOCABULARY])
        .map_err(|error| invalid_error(VOCABULARY, format!("not UTF-8: {error}")))?;
    let vocabulary_offsets = decode_u64(VOCABULARY_OFFSETS, &files[VOCABULARY_OFFSETS])?;

    verify_offsets(STRING_BYTE_OFFSETS, &string_byte_offsets, strings.len())?;
    verify_offsets(STRING_SYMBOL_OFFSETS, &string_symbol_offsets, symbols.len())?;
    verify_offsets(POSTING_OFFSETS, &posting_offsets, postings.len())?;
    verify_offsets(VOCABULARY_OFFSETS, &vocabulary_offsets, vocabulary.len())?;

    let vocabulary_count = usize_count("manifest.json", manifest.vocabulary)?;
    for (index, &symbol) in symbols.iter().enumerate() {
        if symbol as usize >= vocabulary_count {
            return invalid(
                SYMBOLS,
                format!("symbol {symbol} at element {index} is outside the vocabulary"),
            );
        }
    }

    let mut tokens = Vec::with_capacity(vocabulary_count);
    let mut unique = HashSet::with_capacity(vocabulary_count);
    for index in 0..vocabulary_count {
        let start = offset(VOCABULARY_OFFSETS, vocabulary_offsets[index])?;
        let end = offset(VOCABULARY_OFFSETS, vocabulary_offsets[index + 1])?;
        let token = vocabulary.get(start..end).ok_or_else(|| {
            invalid_error(VOCABULARY, format!("token {index} is not valid UTF-8"))
        })?;
        if !unique.insert(token) {
            return invalid(VOCABULARY, format!("duplicate token at element {index}"));
        }
        let valid_token = match manifest.tokenizer.r#type.as_str() {
            "character" => token.chars().count() == 1,
            "whitespace" => {
                let mut split = token.split_whitespace();
                split.next() == Some(token) && split.next().is_none()
            }
            _ => unreachable!(),
        };
        if !valid_token {
            return invalid(
                VOCABULARY,
                format!(
                    "element {index} is not one {} token",
                    manifest.tokenizer.r#type
                ),
            );
        }
        tokens.push(token);
    }

    let string_count = usize_count("manifest.json", manifest.strings)?;
    for string_index in 0..string_count {
        let byte_start = offset(STRING_BYTE_OFFSETS, string_byte_offsets[string_index])?;
        let byte_end = offset(STRING_BYTE_OFFSETS, string_byte_offsets[string_index + 1])?;
        let string = strings.get(byte_start..byte_end).ok_or_else(|| {
            invalid_error(STRINGS, format!("string {string_index} is not valid UTF-8"))
        })?;
        let symbol_start = offset(STRING_SYMBOL_OFFSETS, string_symbol_offsets[string_index])?;
        let symbol_end = offset(
            STRING_SYMBOL_OFFSETS,
            string_symbol_offsets[string_index + 1],
        )?;
        let expected_tokens: Vec<_> = match manifest.tokenizer.r#type.as_str() {
            "character" => CharacterTokenizer::new()
                .tokenize(string)
                .into_iter()
                .map(|token| (token.value.to_string(), token.byte_range))
                .collect(),
            "whitespace" => WhitespaceTokenizer::new()
                .tokenize(string)
                .into_iter()
                .map(|token| (token.value, token.byte_range))
                .collect(),
            _ => unreachable!(),
        };
        if expected_tokens.len() != symbol_end - symbol_start {
            return invalid(
                STRING_SYMBOL_OFFSETS,
                format!("token count for string {string_index} does not match the tokenizer"),
            );
        }
        for (relative_position, (expected_token, expected_range)) in
            expected_tokens.into_iter().enumerate()
        {
            let position = symbol_start + relative_position;
            let (start, end) = byte_ranges[position];
            let start = start as usize;
            let end = end as usize;
            if (start..end) != expected_range {
                return invalid(
                    BYTE_RANGES,
                    format!(
                        "range {start}..{end} at element {position} does not match the tokenizer"
                    ),
                );
            }
            let token = tokens[symbols[position] as usize];
            if token != expected_token {
                return invalid(
                    SYMBOLS,
                    format!("symbol at element {position} does not match the original string"),
                );
            }
        }
    }

    for symbol in 0..vocabulary_count {
        let start = offset(POSTING_OFFSETS, posting_offsets[symbol])?;
        let end = offset(POSTING_OFFSETS, posting_offsets[symbol + 1])?;
        let mut previous = None;
        for (posting_index, &(string_id, position)) in postings[start..end].iter().enumerate() {
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
            let string_start = offset(STRING_SYMBOL_OFFSETS, string_symbol_offsets[string_id])?;
            let string_end = offset(STRING_SYMBOL_OFFSETS, string_symbol_offsets[string_id + 1])?;
            let corpus_index = string_start
                .checked_add(position as usize)
                .ok_or_else(|| invalid_error(POSTINGS, "posting position overflows usize"))?;
            if corpus_index >= string_end {
                return invalid(
                    POSTINGS,
                    format!(
                        "position {position} at element {} is out of range for string {string_id}",
                        start + posting_index
                    ),
                );
            }
            if symbols[corpus_index] as usize != symbol {
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

fn describe_file(name: &str, bytes: &[u8], counts: IndexCounts) -> FileManifest {
    let (element_type, elements) = match name {
        STRINGS | VOCABULARY => ("u8", bytes.len() as u64),
        STRING_BYTE_OFFSETS | STRING_SYMBOL_OFFSETS => ("u64", counts.strings + 1),
        SYMBOLS | BYTE_RANGES | POSTINGS => {
            (if name == SYMBOLS { "u32" } else { "u32x2" }, counts.tokens)
        }
        POSTING_OFFSETS | VOCABULARY_OFFSETS => ("u64", counts.vocabulary + 1),
        _ => unreachable!(),
    };
    FileManifest {
        element_type: element_type.to_owned(),
        elements,
        bytes: bytes.len() as u64,
        sha256: checksum(bytes),
    }
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

fn encode_u32(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn encode_u64(values: impl IntoIterator<Item = u64>) -> Vec<u8> {
    values.into_iter().flat_map(u64::to_le_bytes).collect()
}

fn encode_pairs(values: impl IntoIterator<Item = (u32, u32)>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|(first, second)| first.to_le_bytes().into_iter().chain(second.to_le_bytes()))
        .collect()
}

fn decode_u32(name: &str, bytes: &[u8]) -> Result<Vec<u32>, IndexError> {
    if !bytes.len().is_multiple_of(4) {
        return invalid(name, "byte length is not divisible by 4");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn decode_u64(name: &str, bytes: &[u8]) -> Result<Vec<u64>, IndexError> {
    if !bytes.len().is_multiple_of(8) {
        return invalid(name, "byte length is not divisible by 8");
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn decode_pairs(name: &str, bytes: &[u8]) -> Result<Vec<(u32, u32)>, IndexError> {
    if !bytes.len().is_multiple_of(8) {
        return invalid(name, "byte length is not divisible by 8");
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            (
                u32::from_le_bytes(chunk[..4].try_into().unwrap()),
                u32::from_le_bytes(chunk[4..].try_into().unwrap()),
            )
        })
        .collect())
}

fn verify_offsets(name: &str, offsets: &[u64], expected_end: usize) -> Result<(), IndexError> {
    if offsets.first() != Some(&0) {
        return invalid(name, "first offset must be zero");
    }
    if let Some((index, _)) = offsets
        .windows(2)
        .enumerate()
        .find(|(_, pair)| pair[0] > pair[1])
    {
        return invalid(name, format!("offsets decrease at element {}", index + 1));
    }
    if offsets.last().copied() != Some(expected_end as u64) {
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

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), IndexError> {
    fs::write(path, bytes).map_err(|source| file_io(path, source))
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
            Err(IndexError::FileLength { file, .. }) if file == "symbols.u32"
        ));

        let output = write_fixture(directory.path());
        let mut symbols = fs::read(output.join("symbols.u32")).unwrap();
        symbols[0] ^= 1;
        fs::write(output.join("symbols.u32"), symbols).unwrap();
        assert!(matches!(
            verify_index(&output),
            Err(IndexError::ChecksumMismatch { file, .. }) if file == "symbols.u32"
        ));
    }

    #[test]
    fn rejects_unsupported_manifest_values() {
        let directory = tempfile::tempdir().unwrap();
        let output = write_fixture(directory.path());
        update_manifest(&output, |manifest| manifest["version"] = 2.into());
        assert!(matches!(
            verify_index(&output),
            Err(IndexError::UnsupportedVersion(2))
        ));

        let output = write_fixture(directory.path());
        update_manifest(&output, |manifest| manifest["byte_order"] = "big".into());
        assert!(matches!(
            verify_index(&output),
            Err(IndexError::UnsupportedByteOrder(order)) if order == "big"
        ));
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
                bytes[4..8].copy_from_slice(&2u32.to_le_bytes())
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
            assert!(
                matches!(verify_index(output), Err(IndexError::InvalidData { .. })),
                "{file} corruption was not rejected as invalid data"
            );
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
