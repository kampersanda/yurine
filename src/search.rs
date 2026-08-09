mod filtering;
pub mod range_search;
mod verification;

use std::hash::Hash;
use std::ops::Range;

use crate::corpus::CorpusStore;
use crate::costs::{Cost, EditCosts};
use crate::errors::Result;
use crate::postings::PostingsIndex;
use crate::types::{Position, StringId};

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
    /// The matched zero-based, end-exclusive symbol range.
    pub range: Range<Position>,
    /// The weighted edit distance from the query to the substring.
    pub distance: Cost,
}

/// Coordinates threshold-subsequence filtering and exact verification.
#[derive(Debug, Clone)]
pub struct SearchEngine<Symbol, Costs, Index, Store> {
    costs: Costs,
    index: Index,
    store: Store,
    neighborhood: SubstitutionNeighborhood<Symbol>,
}

impl<Symbol, Costs, Index, Store> SearchEngine<Symbol, Costs, Index, Store>
where
    Symbol: Eq + Hash + Clone,
    Costs: EditCosts<Symbol>,
    Index: PostingsIndex<Symbol>,
    Store: CorpusStore<Symbol>,
{
    /// Creates a search engine.
    pub fn new(costs: Costs, index: Index, store: Store) -> Result<Self> {
        let neighborhood = SubstitutionNeighborhood::new(store.alphabet())?;
        Ok(Self {
            costs,
            index,
            store,
            neighborhood,
        })
    }
}
