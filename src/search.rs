mod builder;
mod encoding;
mod filtering;
pub mod range_search;
mod verification;

use std::hash::Hash;
use std::ops::Range;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::postings::PostingsIndex;
use crate::store::CorpusStore;
use crate::types::{Position, StringId};
use crate::vocabulary::Vocabulary;

use filtering::neighborhood::SubstitutionNeighborhood;

pub use builder::SearchEngineBuilder;

/// A candidate match of a query string in a data string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    string_id: StringId,
    string_position: Position,
    query_position: Position,
}

/// A verified data segment satisfying the inclusive distance threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// The encoded data string corresponding to the matched data sequence.
    pub string_id: StringId,
    /// The matched zero-based, end-exclusive token range.
    pub token_range: Range<Position>,
    /// The weighted edit distance from the query sequence to the data segment.
    pub distance: Cost,
}

/// Coordinates threshold-subsequence filtering and exact verification.
///
/// Create an engine with [`SearchEngineBuilder`].
pub struct SearchEngine<T, C> {
    vocabulary: Vocabulary<T>,
    costs: C,
    index: PostingsIndex,
    store: CorpusStore,
    neighborhood: SubstitutionNeighborhood,
}

impl<T, C> SearchEngine<T, C>
where
    T: Clone + Eq + Hash,
    C: EditCosts<T>,
{
    pub(crate) fn from_parts(
        vocabulary: Vocabulary<T>,
        costs: C,
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
            costs,
            index,
            store,
            neighborhood,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SearchEngine;
    use crate::costs::levenshtein::LevenshteinCosts;
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
            LevenshteinCosts::new(),
            PostingsIndexBuilder::new(1).build(),
            store_builder.build(),
        );

        assert!(matches!(
            result,
            Err(Error::UnknownStringSymbol(symbol)) if symbol == unknown_symbol
        ));
    }
}
