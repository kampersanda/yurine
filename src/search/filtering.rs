pub mod candidate;
pub mod neighborhood;

use std::collections::HashSet;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::postings::PostingsIndex;
use crate::search::Candidate;
use crate::search::filtering::neighborhood::SubstitutionNeighborhood;
use crate::types::{Position, Symbol};

/// Generates candidate anchors.
///
/// Candidates are returned in selected-position, neighborhood, and postings
/// order. Exact duplicate triples are removed while preserving their first
/// occurrence; anchors with different query positions remain distinct.
///
/// Returns [`Error::InvalidQueryPosition`] if `selected` contains a position
/// outside `query`.
pub(super) fn generate_candidates<C>(
    query: &[Symbol],
    selected: &[Position],
    eta: Cost,
    index: &PostingsIndex,
    costs: &C,
    neighborhood: &SubstitutionNeighborhood,
) -> Result<Vec<Candidate>>
where
    C: EditCosts<Symbol>,
{
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    for selected_position in selected {
        let query_symbol =
            query
                .get(selected_position.as_usize())
                .ok_or(Error::InvalidQueryPosition {
                    position: *selected_position,
                    query_len: query.len(),
                })?;
        for neighbor in neighborhood.neighbors(*query_symbol, eta, costs) {
            for posting in index.postings(neighbor) {
                let candidate = Candidate {
                    string_id: posting.string_id,
                    data_position: posting.position,
                    query_position: *selected_position,
                };
                if seen.insert(candidate) {
                    candidates.push(candidate);
                }
            }
        }
    }
    Ok(candidates)
}
