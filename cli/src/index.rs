//! On-disk layout of a prebuilt index directory.
//!
//! An index directory holds the search-engine snapshot next to a copy of the
//! source texts it was built from. The snapshot only stores token sequences,
//! while results report byte offsets into the original text, which tokenization
//! does not preserve. Source texts are therefore stored alongside an offset
//! table so that a search reads only the lines it matched.

use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::tokenization::TokenizerKind;

const FORMAT_VERSION: u32 = 1;
const METADATA_FILE: &str = "metadata.json";
const ENGINE_FILE: &str = "engine.yurine";
const SOURCES_FILE: &str = "sources.txt";
const OFFSETS_FILE: &str = "sources.idx";

/// Description of an index directory, stored as `metadata.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Metadata {
    pub(crate) version: u32,
    pub(crate) tokenizer: TokenizerKind,
    pub(crate) sequence_count: usize,
}

/// Path of the search-engine snapshot inside `directory`.
pub(crate) fn engine_path(directory: &Path) -> PathBuf {
    directory.join(ENGINE_FILE)
}

pub(crate) fn write_metadata(
    directory: &Path,
    tokenizer: TokenizerKind,
    sequence_count: usize,
) -> Result<()> {
    let metadata = Metadata {
        version: FORMAT_VERSION,
        tokenizer,
        sequence_count,
    };
    let path = directory.join(METADATA_FILE);
    let contents = serde_json::to_string_pretty(&metadata)?;
    fs::write(&path, format!("{contents}\n"))
        .with_context(|| format!("failed to write '{}'", path.display()))
}

pub(crate) fn read_metadata(directory: &Path) -> Result<Metadata> {
    let path = directory.join(METADATA_FILE);
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("failed to read index metadata '{}'", path.display()))?;
    let metadata: Metadata = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse index metadata '{}'", path.display()))?;
    if metadata.version != FORMAT_VERSION {
        bail!(
            "index '{}' has unsupported format version {} (expected {FORMAT_VERSION})",
            directory.display(),
            metadata.version
        );
    }
    Ok(metadata)
}

/// Writer of the source-text copy and its line-offset table.
#[derive(Debug)]
pub(crate) struct SourceWriter {
    texts: BufWriter<File>,
    offsets: BufWriter<File>,
    end: u64,
    count: usize,
}

impl SourceWriter {
    /// Creates both files, starting the offset table at the first line.
    pub(crate) fn create(directory: &Path) -> Result<Self> {
        let texts = create(&directory.join(SOURCES_FILE))?;
        let mut offsets = create(&directory.join(OFFSETS_FILE))?;
        offsets.write_all(&0_u64.to_le_bytes())?;
        Ok(Self {
            texts,
            offsets,
            end: 0,
            count: 0,
        })
    }

    /// Appends one source text and records where the next one starts.
    pub(crate) fn push(&mut self, source_text: &str) -> Result<()> {
        writeln!(self.texts, "{source_text}")?;
        self.end += source_text.len() as u64 + 1;
        self.offsets.write_all(&self.end.to_le_bytes())?;
        self.count += 1;
        Ok(())
    }

    /// Flushes both files and returns the number of stored source texts.
    pub(crate) fn finish(mut self) -> Result<usize> {
        self.texts.flush()?;
        self.offsets.flush()?;
        Ok(self.count)
    }
}

fn create(path: &Path) -> Result<BufWriter<File>> {
    let file =
        File::create(path).with_context(|| format!("failed to create '{}'", path.display()))?;
    Ok(BufWriter::new(file))
}

/// Random-access reader of the source texts stored in an index.
#[derive(Debug)]
pub(crate) struct SourceReader {
    texts: File,
    offsets: File,
}

impl SourceReader {
    pub(crate) fn open(directory: &Path, sequence_count: usize) -> Result<Self> {
        let texts = open(&directory.join(SOURCES_FILE))?;
        let path = directory.join(OFFSETS_FILE);
        let offsets = open(&path)?;
        let expected = (sequence_count as u64 + 1) * size_of::<u64>() as u64;
        if offsets.metadata()?.len() != expected {
            bail!(
                "'{}' does not describe {sequence_count} source texts",
                path.display()
            );
        }
        Ok(Self { texts, offsets })
    }

    /// Reads the source text with the given sequence ID.
    pub(crate) fn read(&mut self, sequence_id: usize) -> Result<String> {
        let mut bounds = [0_u8; 2 * size_of::<u64>()];
        self.offsets.seek(SeekFrom::Start(
            sequence_id as u64 * size_of::<u64>() as u64,
        ))?;
        self.offsets.read_exact(&mut bounds)?;
        let start = u64::from_le_bytes(bounds[..size_of::<u64>()].try_into().unwrap());
        let end = u64::from_le_bytes(bounds[size_of::<u64>()..].try_into().unwrap());
        // The stored line ends with a newline that is not part of the text.
        let length = end
            .checked_sub(start + 1)
            .context("index has invalid source offsets")?;

        let mut buffer = vec![0_u8; usize::try_from(length)?];
        self.texts.seek(SeekFrom::Start(start))?;
        self.texts.read_exact(&mut buffer)?;
        String::from_utf8(buffer).context("index has a source text that is not valid UTF-8")
    }
}

fn open(path: &Path) -> Result<File> {
    File::open(path).with_context(|| format!("failed to open '{}'", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{Metadata, SourceReader, SourceWriter, read_metadata, write_metadata};
    use crate::tests::TestDirectory;
    use crate::tokenization::TokenizerKind;

    #[test]
    fn metadata_round_trips() {
        let directory = TestDirectory::new();
        write_metadata(directory.path(), TokenizerKind::Whitespace, 3).unwrap();

        assert_eq!(
            read_metadata(directory.path()).unwrap(),
            Metadata {
                version: 1,
                tokenizer: TokenizerKind::Whitespace,
                sequence_count: 3,
            }
        );
    }

    #[test]
    fn reading_metadata_rejects_an_unsupported_version() {
        let directory = TestDirectory::new();
        directory.write(
            "metadata.json",
            r#"{"version": 2, "tokenizer": "character", "sequence_count": 0}"#,
        );

        let error = read_metadata(directory.path()).unwrap_err();
        assert!(error.to_string().contains("unsupported format version 2"));
    }

    #[test]
    fn reading_metadata_reports_a_missing_index() {
        let directory = TestDirectory::new();

        let error = read_metadata(&directory.path().join("absent")).unwrap_err();
        assert!(error.to_string().contains("failed to read index metadata"));
    }

    #[test]
    fn source_texts_are_read_back_by_sequence_id() {
        let directory = TestDirectory::new();
        let mut writer = SourceWriter::create(directory.path()).unwrap();
        for source_text in ["東京都", "", "a\tb"] {
            writer.push(source_text).unwrap();
        }
        assert_eq!(writer.finish().unwrap(), 3);

        let mut reader = SourceReader::open(directory.path(), 3).unwrap();
        assert_eq!(reader.read(2).unwrap(), "a\tb");
        assert_eq!(reader.read(0).unwrap(), "東京都");
        assert_eq!(reader.read(1).unwrap(), "");
    }

    #[test]
    fn opening_sources_rejects_a_mismatched_sequence_count() {
        let directory = TestDirectory::new();
        let mut writer = SourceWriter::create(directory.path()).unwrap();
        writer.push("東京都").unwrap();
        writer.finish().unwrap();

        let error = SourceReader::open(directory.path(), 2).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not describe 2 source texts")
        );
    }
}
