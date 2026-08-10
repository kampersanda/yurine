mod filtering;
pub mod range_search;
mod verification;

use std::ops::Range;

use crate::costs::{Cost, EditCosts};
use crate::errors::Result;
use crate::postings::PostingsIndex;
use crate::store::CorpusStore;
use crate::types::{Position, StringId, Symbol};

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
pub struct SearchEngine<Costs> {
    costs: Costs,
    index: PostingsIndex,
    store: CorpusStore,
    neighborhood: SubstitutionNeighborhood,
}

impl<Costs> SearchEngine<Costs>
where
    Costs: EditCosts<Symbol>,
{
    /// Creates a search engine.
    pub fn new(costs: Costs, index: PostingsIndex, store: CorpusStore) -> Result<Self> {
        let neighborhood = SubstitutionNeighborhood::new(store.alphabet().iter().copied())?;
        Ok(Self {
            costs,
            index,
            store,
            neighborhood,
        })
    }
}
