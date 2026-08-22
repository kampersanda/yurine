mod candidate;
mod eta;
mod neighborhood;

pub(in crate::search) use candidate::{MinCandidateSelector, SelectedPosition};
pub(in crate::search) use eta::{any_radius_can_select, wider_radii};
pub(in crate::search) use neighborhood::SubstitutionNeighborhood;

use crate::postings::PostingsIndex;
use crate::search::Candidate;

/// Generates candidate anchors from the neighborhoods that selection computed.
///
/// Candidates are returned in selected-position, neighborhood, and postings
/// order. For engines created by [`crate::search::SearchEngineBuilder`], they
/// are unique because selected positions, neighborhood symbols, and each
/// symbol's postings are unique, and postings for distinct symbols do not
/// overlap. If a future index can generate duplicate anchors, verification
/// still consolidates identical result substrings; duplicates would add work
/// but would not change search results.
///
/// The selected positions are consumed so each neighborhood is released once
/// expanded, rather than staying alive beside the candidates through
/// verification.
pub(super) fn generate_candidates(
    selected: Vec<SelectedPosition>,
    index: &PostingsIndex,
) -> Vec<Candidate> {
    let mut candidates = Vec::new();

    for selected_position in selected {
        for neighbor in selected_position.neighbors {
            for posting in index.postings(neighbor) {
                candidates.push(Candidate {
                    string_id: posting.string_id,
                    string_position: posting.position,
                    query_position: selected_position.position,
                });
            }
        }
    }
    candidates
}

/// Returns how many candidates [`generate_candidates`] would generate from
/// `selected`.
///
/// Each candidate is one posting of one neighborhood symbol, so this is the
/// sum of the selected positions' neighborhood posting counts. Posting-list
/// lengths are read in constant time, so this scans the neighborhoods alone
/// and never touches the postings. It is an exact count rather than an
/// estimate, and it lives beside generation so the two cannot disagree about
/// what a candidate is.
pub(super) fn count_candidates(selected: &[SelectedPosition], index: &PostingsIndex) -> usize {
    selected
        .iter()
        .flat_map(|selected_position| selected_position.neighbors.iter())
        .map(|neighbor| index.frequency(*neighbor))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{SelectedPosition, count_candidates, generate_candidates};
    use crate::postings::PostingsIndexBuilder;
    use crate::search::Candidate;
    use crate::types::{Position, Posting, SequenceId, Symbol};

    #[test]
    fn generates_candidates_in_selected_position_and_posting_order() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let mut index = PostingsIndexBuilder::new(2);
        index
            .add_posting(
                first,
                Posting {
                    string_id: SequenceId::new(1),
                    position: Position::new(2),
                },
            )
            .unwrap();
        index
            .add_posting(
                second,
                Posting {
                    string_id: SequenceId::new(0),
                    position: Position::new(3),
                },
            )
            .unwrap();

        let candidates = generate_candidates(
            vec![
                SelectedPosition {
                    position: Position::new(1),
                    neighbors: vec![second],
                },
                SelectedPosition {
                    position: Position::new(0),
                    neighbors: vec![first],
                },
            ],
            &index.build(),
        );

        assert_eq!(
            candidates,
            [
                Candidate {
                    string_id: SequenceId::new(0),
                    string_position: Position::new(3),
                    query_position: Position::new(1),
                },
                Candidate {
                    string_id: SequenceId::new(1),
                    string_position: Position::new(2),
                    query_position: Position::new(0),
                },
            ]
        );
    }

    #[test]
    fn counts_the_candidates_generation_would_produce() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let mut index = PostingsIndexBuilder::new(2);
        for (symbol, string_id, position) in [(first, 0, 0), (first, 1, 2), (second, 1, 3)] {
            index
                .add_posting(
                    symbol,
                    Posting {
                        string_id: SequenceId::new(string_id),
                        position: Position::new(position),
                    },
                )
                .unwrap();
        }
        let index = index.build();
        let selected = vec![
            SelectedPosition {
                position: Position::new(0),
                neighbors: vec![first, second],
            },
            SelectedPosition {
                position: Position::new(1),
                neighbors: vec![second],
            },
        ];

        assert_eq!(
            count_candidates(&selected, &index),
            generate_candidates(selected.clone(), &index).len()
        );
    }

    #[test]
    fn generates_candidates_in_neighborhood_order_within_a_position() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let mut index = PostingsIndexBuilder::new(2);
        for (symbol, string_id) in [(first, 1), (second, 0)] {
            index
                .add_posting(
                    symbol,
                    Posting {
                        string_id: SequenceId::new(string_id),
                        position: Position::new(0),
                    },
                )
                .unwrap();
        }

        let candidates = generate_candidates(
            vec![SelectedPosition {
                position: Position::new(0),
                neighbors: vec![first, second],
            }],
            &index.build(),
        );

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.string_id)
                .collect::<Vec<_>>(),
            [SequenceId::new(1), SequenceId::new(0)]
        );
    }
}
