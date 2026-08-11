pub mod candidate;
pub mod neighborhood;

use std::collections::HashSet;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::postings::PostingsIndex;
use crate::search::Candidate;
use crate::search::filtering::neighborhood::SubstitutionNeighborhood;
use crate::types::{Position, Symbol};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct CandidateGenerationMetrics {
    pub raw_candidates: usize,
    pub unique_candidates: usize,
    pub candidate_vec_capacity: usize,
    pub dedup_set_capacity: usize,
}

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
) -> Result<(Vec<Candidate>, CandidateGenerationMetrics)>
where
    C: EditCosts<Symbol>,
{
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    let mut raw_candidates = 0usize;

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
                raw_candidates += 1;
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
    let metrics = CandidateGenerationMetrics {
        raw_candidates,
        unique_candidates: candidates.len(),
        candidate_vec_capacity: candidates.capacity(),
        dedup_set_capacity: seen.capacity(),
    };
    Ok((candidates, metrics))
}

#[cfg(test)]
mod tests {
    use super::generate_candidates;
    use super::neighborhood::SubstitutionNeighborhood;
    use crate::costs::Cost;
    use crate::costs::levenshtein::LevenshteinCosts;
    use crate::errors::Error;
    use crate::postings::PostingsIndexBuilder;
    use crate::search::Candidate;
    use crate::types::{Position, Posting, StringId, Symbol};

    #[test]
    fn generates_candidates_in_selected_position_and_posting_order() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let mut index = PostingsIndexBuilder::new();
        index.add_posting(
            first,
            Posting {
                string_id: StringId::new(1),
                position: Position::new(2),
            },
        );
        index.add_posting(
            second,
            Posting {
                string_id: StringId::new(0),
                position: Position::new(3),
            },
        );
        let neighborhood = SubstitutionNeighborhood::new([first, second]).unwrap();

        let (candidates, metrics) = generate_candidates(
            &[first, second],
            &[Position::new(1), Position::new(0)],
            Cost::ZERO,
            &index.build(),
            &LevenshteinCosts,
            &neighborhood,
        )
        .unwrap();

        assert_eq!(
            candidates,
            [
                Candidate {
                    string_id: StringId::new(0),
                    data_position: Position::new(3),
                    query_position: Position::new(1),
                },
                Candidate {
                    string_id: StringId::new(1),
                    data_position: Position::new(2),
                    query_position: Position::new(0),
                },
            ]
        );
        assert_eq!(metrics.raw_candidates, 2);
        assert_eq!(metrics.unique_candidates, 2);
    }

    #[test]
    fn rejects_selected_position_outside_the_query() {
        let result = generate_candidates(
            &[Symbol::new(0)],
            &[Position::new(1)],
            Cost::ZERO,
            &PostingsIndexBuilder::new().build(),
            &LevenshteinCosts,
            &SubstitutionNeighborhood::new([]).unwrap(),
        );

        assert_eq!(
            result,
            Err(Error::InvalidQueryPosition {
                position: Position::new(1),
                query_len: 1,
            })
        );
    }
}
