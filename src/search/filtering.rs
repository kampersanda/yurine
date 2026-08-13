pub mod candidate;
pub mod neighborhood;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::postings::PostingsIndex;
use crate::search::Candidate;
use crate::search::filtering::neighborhood::SubstitutionNeighborhood;
use crate::types::{Position, Symbol};

/// Generates candidate anchors.
///
/// Candidates are returned in selected-position, neighborhood, and postings
/// order. For engines created by [`crate::search::SearchEngineBuilder`], they
/// are unique because selected positions, neighborhood symbols, and each
/// symbol's postings are unique, and postings for distinct symbols do not
/// overlap. If a future index can generate duplicate anchors, verification
/// still consolidates identical result substrings; duplicates would add work
/// but would not change search results.
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
                candidates.push(Candidate {
                    string_id: posting.string_id,
                    string_position: posting.position,
                    query_position: *selected_position,
                });
            }
        }
    }
    Ok(candidates)
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
        let mut index = PostingsIndexBuilder::new(2);
        index
            .add_posting(
                first,
                Posting {
                    string_id: StringId::new(1),
                    position: Position::new(2),
                },
            )
            .unwrap();
        index
            .add_posting(
                second,
                Posting {
                    string_id: StringId::new(0),
                    position: Position::new(3),
                },
            )
            .unwrap();
        let neighborhood = SubstitutionNeighborhood::new([first, second]).unwrap();

        let candidates = generate_candidates(
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
                    string_position: Position::new(3),
                    query_position: Position::new(1),
                },
                Candidate {
                    string_id: StringId::new(1),
                    string_position: Position::new(2),
                    query_position: Position::new(0),
                },
            ]
        );
    }

    #[test]
    fn rejects_selected_position_outside_the_query() {
        let result = generate_candidates(
            &[Symbol::new(0)],
            &[Position::new(1)],
            Cost::ZERO,
            &PostingsIndexBuilder::new(0).build(),
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
