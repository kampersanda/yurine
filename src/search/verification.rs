//! Verification of candidates against a distance threshold.

pub mod bidirectional_trie;
pub mod smith_waterman;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::search::{Candidate, Match};
use crate::store::CorpusStore;
use crate::types::{Position, StringId, Symbol};

/// Verification algorithm used to check filtering candidates.
pub(in crate::search) enum Verifier {
    BidirectionalTrie,
    SmithWaterman,
}

impl Verifier {
    /// Returns exactly the non-empty substrings whose distance is at most
    /// `threshold`.
    ///
    /// Each interval must occur exactly once. Results must be ordered by
    /// string ID, then range start, then range end.
    pub(in crate::search) fn verify<C>(
        &self,
        query: &[Symbol],
        candidates: &[Candidate],
        corpus: &CorpusStore,
        threshold: Cost,
        costs: &C,
    ) -> Result<Vec<Match>>
    where
        C: EditCosts<Symbol>,
    {
        match self {
            Self::BidirectionalTrie => {
                bidirectional_trie::verify(query, candidates, corpus, threshold, costs)
            }
            Self::SmithWaterman => {
                smith_waterman::verify(query, candidates, corpus, threshold, costs)
            }
        }
    }
}

fn create_match(
    corpus: &CorpusStore,
    string_id: StringId,
    start: usize,
    end: usize,
    distance: Cost,
) -> Result<Match> {
    let start_position = Position::from_usize(start)?;
    let end_position = Position::from_usize(end - 1)?;
    let start_byte_range = corpus
        .byte_range(string_id, start_position)?
        .ok_or(Error::UnknownString(string_id))?;
    let end_byte_range = corpus
        .byte_range(string_id, end_position)?
        .ok_or(Error::UnknownString(string_id))?;
    Ok(Match {
        string_id,
        token_range: start_position..Position::from_usize(end)?,
        byte_range: start_byte_range.start..end_byte_range.end,
        distance,
    })
}

/// Validates that a candidate's string ID and positions are within bounds.
fn validated_candidate_data<'a>(
    query: &[Symbol],
    candidate: &Candidate,
    corpus: &'a CorpusStore,
) -> Result<&'a [Symbol]> {
    let data = corpus
        .string(candidate.string_id)?
        .ok_or(Error::UnknownString(candidate.string_id))?;
    let data_slice = data;
    let data_position = candidate.data_position.as_usize();
    if data_position >= data_slice.len() {
        return Err(Error::InvalidDataPosition {
            position: candidate.data_position,
            data_len: data_slice.len(),
        });
    }
    let query_position = candidate.query_position.as_usize();
    if query_position >= query.len() {
        return Err(Error::InvalidQueryPosition {
            position: candidate.query_position,
            query_len: query.len(),
        });
    }
    Ok(data)
}

/// Adds two non-negative DP distances, mapping an exact sum above
/// [`f32::MAX`] to infinity.
///
/// IEEE-754 addition can round `f32::MAX +` a sufficiently small positive
/// value back to `f32::MAX`. The error-free transformation below recovers that
/// positive residual so an unrepresentable distance cannot qualify at a
/// finite strict search threshold.
fn add_distance(left: f32, right: f32) -> f32 {
    let sum = left + right;
    if !sum.is_finite() {
        return f32::INFINITY;
    }
    if sum != f32::MAX {
        return sum;
    }
    let right_rounded = sum - left;
    let residual = (left - (sum - right_rounded)) + (right - right_rounded);
    if residual > 0.0 { f32::INFINITY } else { sum }
}

/// Initializes the weighted-edit-distance column for an empty data prefix.
///
/// Internal DP cells use `f32` so accumulation above [`Cost::MAX`] becomes
/// infinity instead of being confused with an exact, representable maximum.
fn root_column<C>(query: &[Symbol], costs: &C) -> Vec<f32>
where
    C: EditCosts<Symbol>,
{
    // `column[r]` is wed(query[..r], empty). Reaching the empty data prefix
    // requires deleting every symbol in the query prefix.
    let mut column = Vec::with_capacity(query.len() + 1);
    column.push(0.0);
    for query_symbol in query {
        column.push(add_distance(
            column.last().copied().unwrap_or(0.0),
            costs.deletion(query_symbol).get(),
        ));
    }
    column
}

/// Advances a weighted-edit-distance column by one data symbol.
fn step_dp<C>(query: &[Symbol], data_symbol: Symbol, previous: &[f32], costs: &C) -> Vec<f32>
where
    C: EditCosts<Symbol>,
{
    debug_assert_eq!(previous.len(), query.len() + 1);

    // If `previous[r]` describes a processed data prefix P, `current[r]`
    // describes P followed by `data_symbol`. Row zero therefore inserts the
    // new data symbol into an empty query.
    let mut current = Vec::with_capacity(query.len() + 1);
    current.push(add_distance(
        previous[0],
        costs.insertion(&data_symbol).get(),
    ));
    for (query_index, query_symbol) in query.iter().enumerate() {
        // The three predecessors consume both symbols, only the data symbol,
        // or only the query symbol, respectively. This fixes the direction as
        // wed(query, data prefix).
        let substitution = add_distance(
            previous[query_index],
            costs.substitution(query_symbol, &data_symbol).get(),
        );
        let insertion = add_distance(
            previous[query_index + 1],
            costs.insertion(&data_symbol).get(),
        );
        let deletion = add_distance(current[query_index], costs.deletion(query_symbol).get());
        current.push(substitution.min(insertion).min(deletion));
    }
    current
}

#[cfg(test)]
mod tests {
    use super::{Verifier, add_distance};
    use crate::costs::{Cost, EditCosts};
    use crate::search::{Candidate, Match};
    use crate::store::{CorpusStore, CorpusStoreBuilder};
    use crate::types::{Position, StringId, Symbol};

    struct UnitCosts;

    impl EditCosts<Symbol> for UnitCosts {
        fn substitution(&self, from: &Symbol, to: &Symbol) -> Cost {
            if from == to { Cost::ZERO } else { Cost::ONE }
        }

        fn deletion(&self, _symbol: &Symbol) -> Cost {
            Cost::ONE
        }

        fn insertion(&self, _symbol: &Symbol) -> Cost {
            Cost::ONE
        }
    }

    fn corpus(symbols: Vec<Symbol>) -> CorpusStore {
        let mut builder = CorpusStoreBuilder::new();
        let original = "x".repeat(symbols.len());
        let byte_ranges = (0..symbols.len())
            .map(|position| position..position + 1)
            .collect();
        builder.add_string(original, symbols, byte_ranges).unwrap();
        builder.build()
    }

    #[test]
    fn bidirectional_trie_returns_each_anchored_interval_once_in_range_order() {
        let a = Symbol::new(0);
        let b = Symbol::new(1);
        let x = Symbol::new(2);
        let corpus = corpus(vec![a, x, b]);
        let candidates: Vec<_> = (0..3)
            .flat_map(|data_position| {
                (0..2).map(move |query_position| Candidate {
                    string_id: StringId::new(0),
                    data_position: Position::new(data_position),
                    query_position: Position::new(query_position),
                })
            })
            .collect();

        let matches = Verifier::BidirectionalTrie
            .verify(&[a, b], &candidates, &corpus, Cost::ONE, &UnitCosts)
            .unwrap();

        assert_eq!(
            matches,
            [
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(0)..Position::new(1),
                    byte_range: 0..1,
                    distance: Cost::ONE,
                },
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(0)..Position::new(2),
                    byte_range: 0..2,
                    distance: Cost::ONE,
                },
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(0)..Position::new(3),
                    byte_range: 0..3,
                    distance: Cost::ONE,
                },
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(1)..Position::new(3),
                    byte_range: 1..3,
                    distance: Cost::ONE,
                },
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(2)..Position::new(3),
                    byte_range: 2..3,
                    distance: Cost::ONE,
                },
            ]
        );
    }

    #[test]
    fn smith_waterman_exhaustively_checks_each_candidate_string_once() {
        let a = Symbol::new(0);
        let corpus = corpus(vec![a, a]);
        let duplicate_candidates = [
            Candidate {
                string_id: StringId::new(0),
                data_position: Position::new(0),
                query_position: Position::new(0),
            },
            Candidate {
                string_id: StringId::new(0),
                data_position: Position::new(1),
                query_position: Position::new(0),
            },
        ];

        let matches = Verifier::SmithWaterman
            .verify(&[a], &duplicate_candidates, &corpus, Cost::ZERO, &UnitCosts)
            .unwrap();

        assert_eq!(
            matches,
            [
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(0)..Position::new(1),
                    byte_range: 0..1,
                    distance: Cost::ZERO,
                },
                Match {
                    string_id: StringId::new(0),
                    token_range: Position::new(1)..Position::new(2),
                    byte_range: 1..2,
                    distance: Cost::ZERO,
                },
            ]
        );
    }

    #[test]
    fn verification_rejects_unknown_string_before_returning_matches() {
        let symbol = Symbol::new(0);
        let corpus = corpus(vec![symbol]);
        let candidate = Candidate {
            string_id: StringId::new(1),
            data_position: Position::new(0),
            query_position: Position::new(0),
        };

        let result = Verifier::BidirectionalTrie.verify(
            &[symbol],
            &[candidate],
            &corpus,
            Cost::ZERO,
            &UnitCosts,
        );

        assert_eq!(
            result,
            Err(crate::errors::Error::UnknownString(StringId::new(1)))
        );
    }

    #[test]
    fn verification_rejects_out_of_bounds_data_position() {
        let symbol = Symbol::new(0);
        let corpus = corpus(vec![symbol]);
        let candidate = Candidate {
            string_id: StringId::new(0),
            data_position: Position::new(1),
            query_position: Position::new(0),
        };

        let result = Verifier::BidirectionalTrie.verify(
            &[symbol],
            &[candidate],
            &corpus,
            Cost::ZERO,
            &UnitCosts,
        );

        assert_eq!(
            result,
            Err(crate::errors::Error::InvalidDataPosition {
                position: Position::new(1),
                data_len: 1,
            })
        );
    }

    #[test]
    fn verification_rejects_out_of_bounds_query_position() {
        let symbol = Symbol::new(0);
        let corpus = corpus(vec![symbol]);
        let candidate = Candidate {
            string_id: StringId::new(0),
            data_position: Position::new(0),
            query_position: Position::new(1),
        };

        let result = Verifier::BidirectionalTrie.verify(
            &[symbol],
            &[candidate],
            &corpus,
            Cost::ZERO,
            &UnitCosts,
        );

        assert_eq!(
            result,
            Err(crate::errors::Error::InvalidQueryPosition {
                position: Position::new(1),
                query_len: 1,
            })
        );
    }

    #[test]
    fn distance_addition_does_not_saturate_at_maximum() {
        assert_eq!(add_distance(f32::MAX, 0.0), f32::MAX);
        assert_eq!(add_distance(f32::MAX, 1.0), f32::INFINITY);
    }
}
