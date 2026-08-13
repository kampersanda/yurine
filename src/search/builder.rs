//! Construction of consistent search engines from token sequences.

use std::hash::Hash;

use super::SearchEngine;
use crate::costs::EditCosts;
use crate::errors::Result;
use crate::postings::PostingsIndexBuilder;
use crate::store::CorpusStoreBuilder;
use crate::types::{Position, Posting, StringId};
use crate::vocabulary::VocabularyBuilder;

/// Builds a [`SearchEngine`] from token sequences in insertion order.
#[derive(Debug)]
pub struct SearchEngineBuilder<T, C> {
    costs: C,
    sequences: Vec<Vec<T>>,
}

impl<T, C> SearchEngineBuilder<T, C>
where
    T: Clone + Eq + Hash,
    C: EditCosts<T>,
{
    /// Creates an empty builder.
    pub fn new(costs: C) -> Self {
        Self {
            costs,
            sequences: Vec::new(),
        }
    }

    /// Adds a data sequence and returns the ID reserved for its encoded string.
    ///
    /// # Errors
    ///
    /// Returns [`crate::errors::Error::StringIdOverflow`] if the corpus has
    /// too many data strings or [`crate::errors::Error::PositionOverflow`]
    /// if the sequence is too long.
    pub fn add_sequence<I>(&mut self, sequence: I) -> Result<StringId>
    where
        I: IntoIterator<Item = T>,
    {
        let string_id = StringId::from_usize(self.sequences.len())?;
        let sequence: Vec<_> = sequence.into_iter().collect();
        Position::from_usize(sequence.len())?;
        self.sequences.push(sequence);
        Ok(string_id)
    }

    /// Builds the vocabulary, corpus store, postings index, and search engine.
    ///
    /// # Errors
    ///
    /// Returns [`crate::errors::Error::SymbolOverflow`] if the corpus has too
    /// many distinct tokens. Returns [`crate::errors::Error::UnknownStringSymbol`]
    /// if a string symbol is not present in the vocabulary.
    pub fn build(self) -> Result<SearchEngine<T, C>> {
        let Self { costs, sequences } = self;

        let mut vocabulary_builder = VocabularyBuilder::new();
        for sequence in &sequences {
            vocabulary_builder.insert_all(sequence.iter().cloned());
        }
        let vocabulary = vocabulary_builder.build()?;

        let mut index_builder = PostingsIndexBuilder::new(vocabulary.len());
        let mut store_builder = CorpusStoreBuilder::new();
        for (raw_string_id, sequence) in sequences.into_iter().enumerate() {
            let string_id = StringId::from_usize(raw_string_id)?;
            let string = vocabulary.encode(sequence);
            for (raw_position, symbol) in string.iter().copied().enumerate() {
                index_builder.add_posting(
                    symbol,
                    Posting {
                        string_id,
                        position: Position::from_usize(raw_position)?,
                    },
                )?;
            }
            store_builder.add_string(string);
        }

        SearchEngine::from_parts(
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
    use crate::types::{Position, StringId};

    #[test]
    fn builds_an_empty_corpus() {
        let engine = SearchEngineBuilder::<char, _>::new(LevenshteinCosts::new())
            .build()
            .unwrap();

        assert!(
            engine
                .range_search(&['a'], &RangeSearchParams::new(Cost::ZERO))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn preserves_insertion_ordered_ids_for_sequences() {
        let mut builder = SearchEngineBuilder::new(LevenshteinCosts::new());

        assert_eq!(
            builder.add_sequence(['東', '京']).unwrap(),
            StringId::new(0)
        );
        assert_eq!(builder.add_sequence([]).unwrap(), StringId::new(1));
        assert_eq!(
            builder.add_sequence(['京', '都']).unwrap(),
            StringId::new(2)
        );
        assert_eq!(
            builder.add_sequence(['東', '京']).unwrap(),
            StringId::new(3)
        );

        let matches = builder
            .build()
            .unwrap()
            .range_search(&['東', '京'], &RangeSearchParams::new(Cost::ZERO))
            .unwrap();

        assert_eq!(
            matches,
            [
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(0)..Position::new(2),
                    distance: Cost::ZERO,
                },
                Match {
                    string_id: StringId::new(3),
                    token_range: Position::new(0)..Position::new(2),
                    distance: Cost::ZERO,
                },
            ]
        );
    }

    #[test]
    fn indexes_repeated_tokens_at_each_position() {
        let mut builder = SearchEngineBuilder::new(LevenshteinCosts::new());
        builder.add_sequence(['a', 'a', 'a']).unwrap();

        let matches = builder
            .build()
            .unwrap()
            .range_search(&['a', 'a'], &RangeSearchParams::new(Cost::ZERO))
            .unwrap();

        assert_eq!(
            matches,
            [
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(0)..Position::new(2),
                    distance: Cost::ZERO,
                },
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(1)..Position::new(3),
                    distance: Cost::ZERO,
                },
            ]
        );
    }

    #[test]
    fn accepts_non_text_token_types() {
        let mut builder = SearchEngineBuilder::new(LevenshteinCosts::new());
        builder.add_sequence([10_u16, 20, 30, 40]).unwrap();

        let matches = builder
            .build()
            .unwrap()
            .range_search(&[20_u16, 30], &RangeSearchParams::new(Cost::ZERO))
            .unwrap();

        assert_eq!(matches[0].token_range, Position::new(1)..Position::new(3));
    }
}
