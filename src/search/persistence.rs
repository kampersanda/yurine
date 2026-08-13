use std::hash::Hash;
use std::path::Path;

use super::SearchEngine;
use crate::errors::{Error, Result};
use crate::persistence::TokenCodec;
use crate::persistence::format::{FileKind, PersistedFile, SectionData, SectionKind, write_file};
use crate::postings::PostingsIndex;
use crate::store::CorpusStore;
use crate::vocabulary::Vocabulary;

impl<T> SearchEngine<T>
where
    T: Clone + Eq + Hash,
{
    /// Saves this immutable search index as a single persistent snapshot.
    ///
    /// The completed file is synchronized and atomically renamed over `path`.
    /// Existing readers keep their previous mapping; callers must never modify
    /// or truncate a published snapshot in place.
    pub fn save_with<C: TokenCodec<T>>(&self, path: impl AsRef<Path>, codec: &C) -> Result<()> {
        self.verify()?;
        let mut token_offsets = Vec::with_capacity(self.vocabulary.len() + 1);
        let mut token_blob = Vec::new();
        token_offsets.push(0);
        for token in self.vocabulary.tokens() {
            let start = token_blob.len();
            codec.encode(token, &mut token_blob)?;
            let decoded = codec.decode(&token_blob[start..])?;
            if &decoded != token {
                return Err(Error::InvalidTokenEncoding(
                    "token codec does not round-trip the vocabulary".into(),
                ));
            }
            token_offsets.push(token_blob.len() as u64);
        }

        let sections = [
            (
                SectionKind::VocabularyTokenOffsets,
                SectionData::U64(&token_offsets),
            ),
            (
                SectionKind::VocabularyTokenBlob,
                SectionData::Bytes {
                    bytes: &token_blob,
                    element_count: self.vocabulary.len() as u64,
                },
            ),
            (
                SectionKind::SequenceOffsets,
                SectionData::U64(self.store.string_offsets()),
            ),
            (
                SectionKind::CorpusSymbols,
                SectionData::Symbols(self.store.symbols()),
            ),
            (
                SectionKind::PostingOffsets,
                SectionData::U64(self.index.posting_offsets()),
            ),
            (
                SectionKind::Postings,
                SectionData::Postings(self.index.postings_slice()),
            ),
        ];
        write_file(path.as_ref(), FileKind::SearchEngine, codec, &sections)
    }

    /// Opens a search index whose large fixed-width arrays remain mmap-backed.
    ///
    /// The file is an immutable snapshot. It must not be modified or truncated
    /// while this engine or any searcher borrowing it remains alive.
    pub fn open_with<C: TokenCodec<T>>(path: impl AsRef<Path>, codec: &C) -> Result<Self> {
        let file = PersistedFile::open(path.as_ref(), FileKind::SearchEngine, codec)?;
        let offsets = file.mapped_slice::<u64>(SectionKind::VocabularyTokenOffsets)?;
        let blob = file.bytes(SectionKind::VocabularyTokenBlob)?;
        let mut tokens = Vec::with_capacity(offsets.len().saturating_sub(1));
        for bounds in offsets.windows(2) {
            let start = usize::try_from(bounds[0]).map_err(|_| Error::PlatformSizeOverflow)?;
            let end = usize::try_from(bounds[1]).map_err(|_| Error::PlatformSizeOverflow)?;
            tokens.push(codec.decode(&blob[start..end])?);
        }
        let vocabulary = Vocabulary::from_tokens(tokens)?;
        let symbol_count = vocabulary.len();
        let store = CorpusStore::from_mapped(
            file.mapped_symbols(SectionKind::CorpusSymbols)?,
            file.mapped_slice::<u64>(SectionKind::SequenceOffsets)?,
            symbol_count,
        );
        let index = PostingsIndex::from_mapped(
            file.mapped_postings(SectionKind::Postings)?,
            file.mapped_slice::<u64>(SectionKind::PostingOffsets)?,
        );
        Self::from_unverified_parts(vocabulary, index, store)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use tempfile::tempdir;

    use super::*;
    use crate::costs::Cost;
    use crate::costs::levenshtein::LevenshteinCosts;
    use crate::persistence::{CharCodec, TokenCodec};
    use crate::search::SearchEngineBuilder;
    use crate::search::range_search::RangeSearchParams;

    fn engine() -> SearchEngine<char> {
        let mut builder = SearchEngineBuilder::new();
        for sequence in ["x東京y", "東京", "東亰", "東京東京", "京都"] {
            builder.add_sequence(sequence.chars()).unwrap();
        }
        builder.build().unwrap()
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

    #[test]
    fn saved_engine_opens_and_matches_owned_results() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("engine.yurine");
        let owned = engine();
        owned.save_with(&path, &CharCodec).unwrap();
        let mapped = SearchEngine::open_with(&path, &CharCodec).unwrap();
        let params = RangeSearchParams::new(Cost::ONE);

        let owned_matches = owned
            .range_searcher(LevenshteinCosts::new())
            .search(&['東', '京'], &params)
            .unwrap();
        let mapped_matches = mapped
            .range_searcher(LevenshteinCosts::new())
            .search(&['東', '京'], &params)
            .unwrap();
        assert_eq!(mapped_matches, owned_matches);
        mapped.verify().unwrap();
    }

    #[test]
    fn empty_engine_round_trips() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("empty.yurine");
        let engine = SearchEngineBuilder::<char>::new().build().unwrap();
        engine.save_with(&path, &CharCodec).unwrap();

        SearchEngine::open_with(&path, &CharCodec)
            .unwrap()
            .verify()
            .unwrap();
    }

    #[test]
    fn saving_the_same_engine_is_byte_deterministic() {
        let directory = tempdir().unwrap();
        let first = directory.path().join("first.yurine");
        let second = directory.path().join("second.yurine");
        let engine = engine();
        engine.save_with(&first, &CharCodec).unwrap();
        engine.save_with(&second, &CharCodec).unwrap();

        assert_eq!(fs::read(first).unwrap(), fs::read(second).unwrap());
    }

    #[test]
    fn mapped_engine_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>(_: &T) {}

        let directory = tempdir().unwrap();
        let path = directory.path().join("engine.yurine");
        engine().save_with(&path, &CharCodec).unwrap();
        let engine = Arc::new(SearchEngine::open_with(&path, &CharCodec).unwrap());
        assert_send_sync(&engine);
        std::thread::spawn(move || engine.verify().unwrap())
            .join()
            .unwrap();
    }

    #[test]
    fn save_rejects_codec_round_trip_violation() {
        struct BrokenCodec;
        impl TokenCodec<char> for BrokenCodec {
            fn id(&self) -> &str {
                "example:broken"
            }
            fn encode(&self, _: &char, output: &mut Vec<u8>) -> Result<()> {
                output.push(b'x');
                Ok(())
            }
            fn decode(&self, _: &[u8]) -> Result<char> {
                Ok('x')
            }
        }

        let directory = tempdir().unwrap();
        let path = directory.path().join("engine.yurine");
        assert!(matches!(
            engine().save_with(path, &BrokenCodec),
            Err(Error::InvalidTokenEncoding(_))
        ));
    }

    #[test]
    fn open_rejects_duplicate_decoded_tokens() {
        struct CollapsingCodec(AtomicBool);
        impl TokenCodec<char> for CollapsingCodec {
            fn id(&self) -> &str {
                "example:collapsing"
            }
            fn encode(&self, token: &char, output: &mut Vec<u8>) -> Result<()> {
                CharCodec.encode(token, output)
            }
            fn decode(&self, bytes: &[u8]) -> Result<char> {
                if self.0.load(Ordering::Relaxed) {
                    Ok('x')
                } else {
                    CharCodec.decode(bytes)
                }
            }
        }

        let directory = tempdir().unwrap();
        let path = directory.path().join("duplicate-tokens.yurine");
        let codec = CollapsingCodec(AtomicBool::new(false));
        engine().save_with(&path, &codec).unwrap();
        codec.0.store(true, Ordering::Relaxed);

        assert!(matches!(
            SearchEngine::open_with(&path, &codec),
            Err(Error::InvalidFile(
                "decoded vocabulary contains duplicate tokens"
            ))
        ));
    }

    #[test]
    fn corrupt_corpus_symbol_is_rejected_lazily_and_by_verify() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("corrupt-corpus.yurine");
        let mut builder = SearchEngineBuilder::new();
        builder.add_sequence(['a']).unwrap();
        builder
            .build()
            .unwrap()
            .save_with(&path, &CharCodec)
            .unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let corpus = section_offset(&bytes, SectionKind::CorpusSymbols as u32);
        bytes[corpus..corpus + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let mapped = SearchEngine::open_with(&path, &CharCodec).unwrap();
        assert!(matches!(
            mapped.verify(),
            Err(Error::UnknownStringSymbol(u32::MAX))
        ));
        assert!(matches!(
            mapped
                .range_searcher(LevenshteinCosts::new())
                .search(&['a'], &RangeSearchParams::new(Cost::ZERO)),
            Err(Error::UnknownStringSymbol(u32::MAX))
        ));
    }

    #[test]
    fn verify_rejects_posting_that_disagrees_with_corpus() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("corrupt-posting.yurine");
        let mut builder = SearchEngineBuilder::new();
        builder.add_sequence(['a']).unwrap();
        builder
            .build()
            .unwrap()
            .save_with(&path, &CharCodec)
            .unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let postings = section_offset(&bytes, SectionKind::Postings as u32);
        bytes[postings + 4..postings + 8].copy_from_slice(&1_u32.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let mapped = SearchEngine::open_with(&path, &CharCodec).unwrap();
        assert!(matches!(
            mapped.verify(),
            Err(Error::InvalidFile("posting does not match the corpus"))
        ));
    }
}
