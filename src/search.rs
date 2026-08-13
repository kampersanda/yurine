mod builder;
mod encoding;
mod filtering;
pub mod range_search;
mod verification;

use std::hash::Hash;
use std::ops::Range;

use crate::costs::Cost;
use crate::errors::{Error, Result};
use crate::postings::PostingsIndex;
use crate::store::CorpusStore;
use crate::types::{Position, SequenceId};
use crate::vocabulary::Vocabulary;

use filtering::neighborhood::SubstitutionNeighborhood;

pub use builder::SearchEngineBuilder;

/// A candidate match of a query string in a data string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    string_id: SequenceId,
    string_position: Position,
    query_position: Position,
}

/// A verified data segment satisfying the inclusive distance threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// The identifier returned when the matched data sequence was added.
    pub sequence_id: SequenceId,
    /// The matched zero-based, end-exclusive token range.
    pub token_range: Range<Position>,
    /// The weighted edit distance from the query sequence to the data segment.
    pub distance: Cost,
}

/// A reusable search index built from data sequences.
///
/// The index owns the vocabulary, encoded corpus, and postings needed for
/// filtering and verification. Edit costs are supplied for each search, so
/// one index can be reused with different cost policies.
///
/// Create an index with [`SearchEngineBuilder`].
pub struct SearchEngine<T> {
    vocabulary: Vocabulary<T>,
    index: PostingsIndex,
    store: CorpusStore,
    neighborhood: SubstitutionNeighborhood,
}

impl<T> SearchEngine<T>
where
    T: Clone + Eq + Hash,
{
    pub(crate) fn from_parts(
        vocabulary: Vocabulary<T>,
        index: PostingsIndex,
        store: CorpusStore,
    ) -> Result<Self> {
        for symbol in store.alphabet() {
            if vocabulary.token(*symbol).is_none() {
                return Err(Error::UnknownStringSymbol(*symbol));
            }
        }
        let neighborhood = SubstitutionNeighborhood::new(store.alphabet().iter().copied())?;
        Ok(Self {
            vocabulary,
            index,
            store,
            neighborhood,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SearchEngine;
    use crate::errors::Error;
    use crate::postings::PostingsIndexBuilder;
    use crate::store::CorpusStoreBuilder;
    use crate::types::Symbol;
    use crate::vocabulary::VocabularyBuilder;

    #[test]
    fn rejects_string_symbol_absent_from_vocabulary() {
        let mut vocabulary_builder = VocabularyBuilder::new();
        vocabulary_builder.insert('a');
        let vocabulary = vocabulary_builder.build().unwrap();
        let unknown_symbol = Symbol::new(1);

        let mut store_builder = CorpusStoreBuilder::new();
        store_builder.add_string(vec![unknown_symbol]);

        let result = SearchEngine::from_parts(
            vocabulary,
            PostingsIndexBuilder::new(1).build(),
            store_builder.build(),
        );

        assert!(matches!(
            result,
            Err(Error::UnknownStringSymbol(symbol)) if symbol == unknown_symbol
        ));
    }
}
