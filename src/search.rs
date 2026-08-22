//! Building and querying reusable sequence indexes.
//!
//! Add data with [`SearchEngineBuilder`], build a [`SearchEngine`], then create
//! a range searcher with [`SearchEngine::range_searcher`]. Search configuration
//! and execution live in [`range_search`].

mod bound;
mod builder;
mod encoding;
mod filtering;
#[cfg(feature = "persist")]
mod persistence;
pub mod range_search;
mod verification;

use std::hash::Hash;
use std::ops::Range;

use crate::errors::Result;
use crate::postings::PostingsIndex;
use crate::store::CorpusStore;
use crate::types::{Position, SequenceId};
use crate::vocabulary::Vocabulary;

use filtering::SubstitutionNeighborhood;

pub use builder::SearchEngineBuilder;
pub use range_search::{RangeSearchMetrics, RangeSearchParams, RangeSearcher};

/// A candidate match of a query string in a data string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    string_id: SequenceId,
    string_position: Position,
    query_position: Position,
}

/// A verified data segment satisfying the inclusive distance threshold.
///
/// One match stands for one part of a data sequence, not for one alignment:
/// segments overlapping it that the threshold also admits are reduced to the
/// closest of them. See [`RangeSearcher::search`].
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// The identifier returned when the matched data sequence was added.
    pub sequence_id: usize,
    /// The matched zero-based, end-exclusive token range.
    pub token_range: Range<usize>,
    /// The weighted edit distance from the query sequence to the data segment.
    pub distance: f32,
}

/// A reusable search index built from data sequences.
///
/// The index owns the vocabulary, encoded corpus, and postings needed for
/// filtering and verification. Edit costs are owned by range searchers, so
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
        // Builder-produced engines pay the complete corpus validation cost once
        // here, after which owned search access does not need to rescan symbols.
        store.verify()?;
        Self::from_unverified_parts(vocabulary, index, store)
    }

    /// Assembles an engine without scanning corpus and posting payloads.
    ///
    /// This is the mmap open path: file metadata and offset arrays have already
    /// passed structural validation, but large payload semantics are checked
    /// lazily during search or completely through [`Self::verify`].
    pub(crate) fn from_unverified_parts(
        vocabulary: Vocabulary<T>,
        index: PostingsIndex,
        store: CorpusStore,
    ) -> Result<Self> {
        let alphabet = index.alphabet();
        let neighborhood = SubstitutionNeighborhood::new(alphabet)?;
        Ok(Self {
            vocabulary,
            index,
            store,
            neighborhood,
        })
    }

    /// Fully validates corpus symbols and postings against each other.
    ///
    /// The ordering matters: corpus validation establishes the precondition
    /// that lets posting validation access each string without rescanning it.
    pub fn verify(&self) -> Result<()> {
        self.store.verify()?;
        self.index.verify(&self.store)
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
            store_builder.build(1),
        );

        assert!(matches!(
            result,
            Err(Error::UnknownStringSymbol(symbol)) if symbol == unknown_symbol.get()
        ));
    }
}
