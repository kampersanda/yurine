//! Verification of candidates against a distance threshold.

mod bidirectional_trie;
mod smith_waterman;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::search::{Candidate, Match};
use crate::store::CorpusStore;
use crate::types::{Position, SequenceId, Symbol};

/// Verification algorithm used to check filtering candidates.
pub(in crate::search) enum Verifier {
    BidirectionalTrie,
    SmithWaterman,
}

impl Verifier {
    /// Returns exactly the non-empty substrings whose distance is at most
    /// `threshold`.
    ///
    /// Each substring must occur exactly once. Results must be ordered by data
    /// string ID, then symbol-range start, then symbol-range end.
    pub(in crate::search) fn verify<C>(
        &self,
        query_string: &[Symbol],
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
                bidirectional_trie::verify(query_string, candidates, corpus, threshold, costs)
            }
            Self::SmithWaterman => {
                smith_waterman::verify(query_string, candidates, corpus, threshold, costs)
            }
        }
    }
}

fn create_match(
    string_id: SequenceId,
    symbol_start: usize,
    symbol_end: usize,
    distance: Cost,
) -> Result<Match> {
    Ok(Match {
        sequence_id: string_id,
        token_range: Position::from_usize(symbol_start)?..Position::from_usize(symbol_end)?,
        distance,
    })
}

/// Validates that a candidate's string ID and positions are within bounds.
fn validated_candidate_string<'a>(
    query_string: &[Symbol],
    candidate: &Candidate,
    corpus: &'a CorpusStore,
) -> Result<&'a [Symbol]> {
    let string = corpus
        .string(candidate.string_id)?
        .ok_or(Error::UnknownString(candidate.string_id))?;
    let string_position = candidate.string_position.as_usize();
    if string_position >= string.len() {
        return Err(Error::InvalidStringPosition {
            position: candidate.string_position,
            string_len: string.len(),
        });
    }
    let query_position = candidate.query_position.as_usize();
    if query_position >= query_string.len() {
        return Err(Error::InvalidQueryPosition {
            position: candidate.query_position,
            query_len: query_string.len(),
        });
    }
    Ok(string)
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
fn root_column<C>(query_string: &[Symbol], costs: &C) -> Vec<f32>
where
    C: EditCosts<Symbol>,
{
    // `column[r]` is wed(query_string[..r], empty). Reaching the empty data
    // prefix requires deleting every symbol in the query-string prefix.
    let mut column = Vec::with_capacity(query_string.len() + 1);
    column.push(0.0);
    for query_symbol in query_string {
        column.push(add_distance(
            column.last().copied().unwrap_or(0.0),
            costs.deletion(query_symbol).get(),
        ));
    }
    column
}

/// Advances a weighted-edit-distance column by one data symbol.
fn step_dp<C>(
    query_string: &[Symbol],
    string_symbol: Symbol,
    previous: &[f32],
    costs: &C,
) -> Vec<f32>
where
    C: EditCosts<Symbol>,
{
    debug_assert_eq!(previous.len(), query_string.len() + 1);

    // If `previous[r]` describes a processed data prefix P, `current[r]`
    // describes P followed by `string_symbol`. Row zero therefore inserts the
    // new data symbol into an empty query string.
    let mut current = Vec::with_capacity(query_string.len() + 1);
    current.push(add_distance(
        previous[0],
        costs.insertion(&string_symbol).get(),
    ));
    for (query_index, query_symbol) in query_string.iter().enumerate() {
        // The three predecessors consume both symbols, only the string symbol,
        // or only the query symbol, respectively. This fixes the direction as
        // wed(query_string, string prefix).
        let substitution = add_distance(
            previous[query_index],
            costs.substitution(query_symbol, &string_symbol).get(),
        );
        let insertion = add_distance(
            previous[query_index + 1],
            costs.insertion(&string_symbol).get(),
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
    use crate::types::{Position, SequenceId, Symbol};

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
        builder.add_string(symbols);
        builder.build(8)
    }

    #[test]
    fn bidirectional_trie_returns_each_anchored_substring_once_in_symbol_range_order() {
        let a = Symbol::new(0);
        let b = Symbol::new(1);
        let x = Symbol::new(2);
        let corpus = corpus(vec![a, x, b]);
        let candidates: Vec<_> = (0..3)
            .flat_map(|string_position| {
                (0..2).map(move |query_position| Candidate {
                    string_id: SequenceId::new(0),
                    string_position: Position::new(string_position),
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
                    sequence_id: SequenceId::new(0),
                    token_range: Position::new(0)..Position::new(1),
                    distance: Cost::ONE,
                },
                Match {
                    sequence_id: SequenceId::new(0),
                    token_range: Position::new(0)..Position::new(2),
                    distance: Cost::ONE,
                },
                Match {
                    sequence_id: SequenceId::new(0),
                    token_range: Position::new(0)..Position::new(3),
                    distance: Cost::ONE,
                },
                Match {
                    sequence_id: SequenceId::new(0),
                    token_range: Position::new(1)..Position::new(3),
                    distance: Cost::ONE,
                },
                Match {
                    sequence_id: SequenceId::new(0),
                    token_range: Position::new(2)..Position::new(3),
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
                string_id: SequenceId::new(0),
                string_position: Position::new(0),
                query_position: Position::new(0),
            },
            Candidate {
                string_id: SequenceId::new(0),
                string_position: Position::new(1),
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
                    sequence_id: SequenceId::new(0),
                    token_range: Position::new(0)..Position::new(1),
                    distance: Cost::ZERO,
                },
                Match {
                    sequence_id: SequenceId::new(0),
                    token_range: Position::new(1)..Position::new(2),
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
            string_id: SequenceId::new(1),
            string_position: Position::new(0),
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
            Err(crate::errors::Error::UnknownString(SequenceId::new(1)))
        );
    }

    #[test]
    fn verification_rejects_out_of_bounds_string_position() {
        let symbol = Symbol::new(0);
        let corpus = corpus(vec![symbol]);
        let candidate = Candidate {
            string_id: SequenceId::new(0),
            string_position: Position::new(1),
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
            Err(crate::errors::Error::InvalidStringPosition {
                position: Position::new(1),
                string_len: 1,
            })
        );
    }

    #[test]
    fn verification_rejects_out_of_bounds_query_position() {
        let symbol = Symbol::new(0);
        let corpus = corpus(vec![symbol]);
        let candidate = Candidate {
            string_id: SequenceId::new(0),
            string_position: Position::new(0),
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
