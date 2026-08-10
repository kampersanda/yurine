//! Construction of consistent search engines from strings.

use std::hash::Hash;

use super::SearchEngine;
use crate::costs::EditCosts;
use crate::errors::Result;
use crate::postings::PostingsIndexBuilder;
use crate::store::CorpusStoreBuilder;
use crate::tokenization::Tokenizer;
use crate::types::{Position, Posting, StringId};
use crate::vocabulary::VocabularyBuilder;

/// Builds a [`SearchEngine`] from strings in insertion order.
#[derive(Debug)]
pub struct SearchEngineBuilder<T, Costs>
where
    T: Tokenizer,
{
    tokenizer: T,
    costs: Costs,
    strings: Vec<Vec<T::Token>>,
}

impl<T, Costs> SearchEngineBuilder<T, Costs>
where
    T: Tokenizer,
    T::Token: Clone + Eq + Hash,
    Costs: EditCosts<T::Token>,
{
    /// Creates an empty builder.
    pub fn new(tokenizer: T, costs: Costs) -> Self {
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
    /// too many strings, or [`crate::errors::Error::PositionOverflow`] if the
    /// tokenized string is too long.
    pub fn add_string(&mut self, input: &str) -> Result<StringId> {
        let string_id = StringId::from_usize(self.strings.len())?;
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
    /// many distinct tokens.
    pub fn build(self) -> Result<SearchEngine<T, Costs>> {
        let Self {
            tokenizer,
            costs,
            strings,
        } = self;

        let mut vocabulary_builder = VocabularyBuilder::new();
        for string in &strings {
            vocabulary_builder.insert_all(string.iter().cloned());
        }
        let vocabulary = vocabulary_builder.build()?;

        let mut index_builder = PostingsIndexBuilder::new();
        let mut store_builder = CorpusStoreBuilder::new();
        for (raw_string_id, string) in strings.into_iter().enumerate() {
            let string_id = StringId::from_usize(raw_string_id)?;
            let symbols = vocabulary.encode(string);
            for (raw_position, symbol) in symbols.iter().copied().enumerate() {
                index_builder.add_posting(
                    symbol,
                    Posting {
                        string_id,
                        position: Position::from_usize(raw_position)?,
                    },
                );
            }
            store_builder.add_string(symbols);
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

#[cfg(test)]
mod tests {
    use super::SearchEngineBuilder;
    use crate::costs::Cost;
    use crate::costs::levenshtein::LevenshteinCosts;
    use crate::search::Match;
    use crate::search::range_search::RangeSearchParams;
    use crate::tokenization::Tokenizer;
    use crate::tokenization::character::CharacterTokenizer;
    use crate::types::{Position, StringId};

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
                    range: Position::new(0)..Position::new(2),
                    distance: Cost::ZERO,
                },
                Match {
                    string_id: StringId::new(3),
                    range: Position::new(0)..Position::new(2),
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
                    range: Position::new(0)..Position::new(2),
                    distance: Cost::ZERO,
                },
                Match {
                    string_id: StringId::new(0),
                    range: Position::new(1)..Position::new(3),
                    distance: Cost::ZERO,
                },
            ]
        );
    }

    struct WholeStringTokenizer;

    impl Tokenizer for WholeStringTokenizer {
        type Token = String;

        fn tokenize(&self, input: &str) -> Vec<Self::Token> {
            vec![input.to_owned()]
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

        assert_eq!(matches[0].range, Position::new(0)..Position::new(1));
    }
}
