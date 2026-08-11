//! Construction of consistent search engines from strings.

use std::hash::Hash;

use super::SearchEngine;
use crate::costs::EditCosts;
use crate::errors::{Error, Result};
use crate::postings::PostingsIndexBuilder;
use crate::store::CorpusStoreBuilder;
use crate::tokenization::{Tokenized, Tokenizer};
use crate::types::{Position, Posting, StringId};
use crate::vocabulary::VocabularyBuilder;

/// Builds a [`SearchEngine`] from strings in insertion order.
#[derive(Debug)]
pub struct SearchEngineBuilder<T, C>
where
    T: Tokenizer,
{
    tokenizer: T,
    costs: C,
    strings: Vec<Vec<Tokenized<T::Token>>>,
}

impl<T, C> SearchEngineBuilder<T, C>
where
    T: Tokenizer,
    T::Token: Clone + Eq + Hash,
    C: EditCosts<T::Token>,
{
    /// Creates an empty builder.
    pub fn new(tokenizer: T, costs: C) -> Self {
        Self {
            tokenizer,
            costs,
            strings: Vec::new(),
        }
    }

    /// Tokenizes and adds a string, returning its insertion-ordered ID.
    ///
    /// # Errors
    ///
    /// Returns [`crate::errors::Error::StringIdOverflow`] if the corpus has
    /// too many strings, [`crate::errors::Error::PositionOverflow`] if the
    /// tokenized string is too long, or
    /// [`crate::errors::Error::ByteOffsetOverflow`] if the UTF-8 string is
    /// larger than `u32` bytes.
    pub fn add_string(&mut self, input: &str) -> Result<StringId> {
        let string_id = StringId::from_usize(self.strings.len())?;
        validate_byte_length(input.len())?;
        let tokens = self.tokenizer.tokenize(input);
        Position::from_usize(tokens.len())?;
        self.strings.push(tokens);
        Ok(string_id)
    }

    /// Builds the vocabulary, corpus store, postings index, and search engine.
    ///
    /// # Errors
    ///
    /// Returns [`crate::errors::Error::SymbolOverflow`] if the corpus has too
    /// many distinct tokens, or [`crate::errors::Error::ByteOffsetOverflow`]
    /// if a token byte offset does not fit in `u32`.
    pub fn build(self) -> Result<SearchEngine<T, C>> {
        let Self {
            tokenizer,
            costs,
            strings,
        } = self;

        let mut vocabulary_builder = VocabularyBuilder::new();
        for string in &strings {
            vocabulary_builder.insert_all(string.iter().map(|token| token.value.clone()));
        }
        let vocabulary = vocabulary_builder.build()?;

        let mut index_builder = PostingsIndexBuilder::new();
        let mut store_builder = CorpusStoreBuilder::new();
        for (raw_string_id, string) in strings.into_iter().enumerate() {
            let string_id = StringId::from_usize(raw_string_id)?;
            let (tokens, byte_ranges): (Vec<_>, Vec<_>) = string
                .into_iter()
                .map(|token| (token.value, token.byte_range))
                .unzip();
            let symbols = vocabulary.encode(tokens);
            for (raw_position, symbol) in symbols.iter().copied().enumerate() {
                index_builder.add_posting(
                    symbol,
                    Posting {
                        string_id,
                        position: Position::from_usize(raw_position)?,
                    },
                );
            }
            store_builder.add_string(symbols, byte_ranges)?;
        }

        SearchEngine::from_parts(
            tokenizer,
            vocabulary,
            costs,
            index_builder.build(),
            store_builder.build(),
        )
    }
}

fn validate_byte_length(length: usize) -> Result<()> {
    u32::try_from(length)
        .map(|_| ())
        .map_err(|_| Error::ByteOffsetOverflow)
}

#[cfg(test)]
mod tests {
    use super::{SearchEngineBuilder, validate_byte_length};
    use crate::costs::Cost;
    use crate::costs::levenshtein::LevenshteinCosts;
    use crate::search::Match;
    use crate::search::range_search::RangeSearchParams;
    use crate::tokenization::character::CharacterTokenizer;
    use crate::tokenization::{Tokenized, Tokenizer};
    use crate::types::{Position, StringId};

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_string_byte_lengths_larger_than_u32() {
        let too_large = usize::try_from(u64::from(u32::MAX) + 1).unwrap();

        assert_eq!(
            validate_byte_length(too_large),
            Err(crate::errors::Error::ByteOffsetOverflow)
        );
    }

    #[test]
    fn builds_an_empty_corpus() {
        let engine = SearchEngineBuilder::new(CharacterTokenizer::new(), LevenshteinCosts::new())
            .build()
            .unwrap();

        assert!(
            engine
                .range_search("a", &RangeSearchParams::new(Cost::ZERO))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn preserves_insertion_ordered_ids_for_unicode_strings() {
        let mut builder =
            SearchEngineBuilder::new(CharacterTokenizer::new(), LevenshteinCosts::new());

        assert_eq!(builder.add_string("東京").unwrap(), StringId::new(0));
        assert_eq!(builder.add_string("").unwrap(), StringId::new(1));
        assert_eq!(builder.add_string("京都").unwrap(), StringId::new(2));
        assert_eq!(builder.add_string("東京").unwrap(), StringId::new(3));

        let matches = builder
            .build()
            .unwrap()
            .range_search("東京", &RangeSearchParams::new(Cost::ZERO))
            .unwrap();

        assert_eq!(
            matches,
            [
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(0)..Position::new(2),
                    byte_range: 0..6,
                    distance: Cost::ZERO,
                },
                Match {
                    string_id: StringId::new(3),
                    token_range: Position::new(0)..Position::new(2),
                    byte_range: 0..6,
                    distance: Cost::ZERO,
                },
            ]
        );
    }

    #[test]
    fn indexes_repeated_tokens_at_each_position() {
        let mut builder =
            SearchEngineBuilder::new(CharacterTokenizer::new(), LevenshteinCosts::new());
        builder.add_string("aaa").unwrap();

        let matches = builder
            .build()
            .unwrap()
            .range_search("aa", &RangeSearchParams::new(Cost::ZERO))
            .unwrap();

        assert_eq!(
            matches,
            [
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(0)..Position::new(2),
                    byte_range: 0..2,
                    distance: Cost::ZERO,
                },
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(1)..Position::new(3),
                    byte_range: 1..3,
                    distance: Cost::ZERO,
                },
            ]
        );
    }

    #[test]
    fn returns_utf8_byte_range_in_original_string() {
        let mut builder =
            SearchEngineBuilder::new(CharacterTokenizer::new(), LevenshteinCosts::new());
        builder.add_string("a東京b").unwrap();

        let matches = builder
            .build()
            .unwrap()
            .range_search("東京", &RangeSearchParams::new(Cost::ZERO))
            .unwrap();

        assert_eq!(matches[0].token_range, Position::new(1)..Position::new(3));
        assert_eq!(matches[0].byte_range, 1..7);
    }

    struct WholeStringTokenizer;

    impl Tokenizer for WholeStringTokenizer {
        type Token = String;

        fn tokenize(&self, input: &str) -> Vec<Tokenized<Self::Token>> {
            vec![Tokenized::new(input.to_owned(), 0..input.len())]
        }
    }

    #[test]
    fn uses_the_same_tokenizer_for_corpus_and_query() {
        let mut builder = SearchEngineBuilder::new(WholeStringTokenizer, LevenshteinCosts::new());
        builder.add_string("東京").unwrap();

        let matches = builder
            .build()
            .unwrap()
            .range_search("東京", &RangeSearchParams::new(Cost::ZERO))
            .unwrap();

        assert_eq!(matches[0].token_range, Position::new(0)..Position::new(1));
        assert_eq!(matches[0].byte_range, 0..6);
    }
}
