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
    let byte_ranges = corpus
        .byte_ranges(string_id)?
        .ok_or(Error::UnknownString(string_id))?;
    Ok(Match {
        string_id,
        token_range: Position::from_usize(start)?..Position::from_usize(end)?,
        byte_range: byte_ranges[start].start..byte_ranges[end - 1].end,
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
    use super::Verifier;
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

    fn fixture() -> ([Symbol; 1], [Candidate; 1], CorpusStore) {
        let symbol = Symbol::new(0);
        let mut builder = CorpusStoreBuilder::new();
        builder.add_string(vec![symbol, symbol], vec![0..1, 1..2]);
        let candidate = Candidate {
            string_id: StringId::new(0),
            data_position: Position::new(0),
            query_position: Position::new(0),
        };
        ([symbol], [candidate], builder.build())
    }

    #[test]
    fn bidirectional_trie_dispatches_to_anchor_local_verification() {
        let (query, candidates, corpus) = fixture();

        let matches = Verifier::BidirectionalTrie
            .verify(&query, &candidates, &corpus, Cost::ZERO, &UnitCosts)
            .unwrap();

        assert_eq!(
            matches,
            [Match {
                string_id: StringId::new(0),
                token_range: Position::new(0)..Position::new(1),
                byte_range: 0..1,
                distance: Cost::ZERO,
            }]
        );
    }

    #[test]
    fn smith_waterman_dispatches_to_exhaustive_verification() {
        let (query, candidates, corpus) = fixture();

        let matches = Verifier::SmithWaterman
            .verify(&query, &candidates, &corpus, Cost::ZERO, &UnitCosts)
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
}
