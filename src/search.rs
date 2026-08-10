mod encoding;
mod filtering;
pub mod range_search;
mod verification;

use std::hash::Hash;
use std::ops::Range;

use crate::costs::{Cost, EditCosts};
use crate::errors::Result;
use crate::postings::PostingsIndex;
use crate::store::CorpusStore;
use crate::tokenization::Tokenizer;
use crate::types::{Position, StringId};
use crate::vocabulary::Vocabulary;

use filtering::neighborhood::SubstitutionNeighborhood;

/// A candidate match of a query in a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Candidate {
    string_id: StringId,
    data_position: Position,
    query_position: Position,
}

/// A verified substring satisfying the inclusive distance threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// The data string containing the match.
    pub string_id: StringId,
    /// The matched zero-based, end-exclusive token range.
    pub range: Range<Position>,
    /// The weighted edit distance from the query to the substring.
    pub distance: Cost,
}

/// Coordinates threshold-subsequence filtering and exact verification.
pub struct SearchEngine<T, Costs>
where
    T: Tokenizer,
{
    tokenizer: T,
    vocabulary: Vocabulary<T::Token>,
    costs: Costs,
    index: PostingsIndex,
    store: CorpusStore,
    neighborhood: SubstitutionNeighborhood,
}

impl<T, Costs> SearchEngine<T, Costs>
where
    T: Tokenizer,
    T::Token: Clone + Eq + Hash,
    Costs: EditCosts<T::Token>,
{
    /// Creates a search engine from prebuilt token and symbol data.
    ///
    /// The index and store must use symbols assigned by `vocabulary`.
    pub fn new(
        tokenizer: T,
        vocabulary: Vocabulary<T::Token>,
        costs: Costs,
        index: PostingsIndex,
        store: CorpusStore,
    ) -> Result<Self> {
        let neighborhood = SubstitutionNeighborhood::new(store.alphabet().iter().copied())?;
        Ok(Self {
            tokenizer,
            vocabulary,
            costs,
            index,
            store,
            neighborhood,
        })
    }
}
