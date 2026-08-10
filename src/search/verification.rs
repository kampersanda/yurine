//! Verification of candidates against a distance threshold.

pub mod bidirectional_trie;
pub mod smith_waterman;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::search::{Candidate, Match};
use crate::store::CorpusStore;

/// Verifies filtering candidates against an inclusive distance threshold.
///
/// Verification takes `&self`, so one verifier can serve concurrent searches.
/// Implementations must keep any working state local to a single call.
pub trait Verifier<Symbol> {
    /// Returns exactly the non-empty substrings whose distance is at most
    /// `threshold`.
    ///
    /// Each interval must occur exactly once. Results must be ordered by
    /// string ID, then range start, then range end.
    fn verify<Costs>(
        &self,
        query: &[Symbol],
        candidates: &[Candidate],
        corpus: &CorpusStore<Symbol>,
        threshold: Cost,
        costs: &Costs,
    ) -> Result<Vec<Match>>
    where
        Costs: EditCosts<Symbol>;
}

/// Validates that a candidate's string ID and positions are within bounds.
fn validated_candidate_data<'a, Symbol: 'a>(
    query: &[Symbol],
    candidate: &Candidate,
    corpus: &'a CorpusStore<Symbol>,
) -> Result<&'a [Symbol]> {
    let data = corpus
        .sequence(candidate.string_id)?
        .ok_or(Error::UnknownString(candidate.string_id))?;
    let data_slice = data.as_ref();
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
fn root_column<Symbol, Costs>(query: &[&Symbol], costs: &Costs) -> Vec<f32>
where
    Costs: EditCosts<Symbol>,
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
fn step_dp<Symbol, Costs>(
    query: &[&Symbol],
    data_symbol: &Symbol,
    previous: &[f32],
    costs: &Costs,
) -> Vec<f32>
where
    Costs: EditCosts<Symbol>,
{
    debug_assert_eq!(previous.len(), query.len() + 1);

    // If `previous[r]` describes a processed data prefix P, `current[r]`
    // describes P followed by `data_symbol`. Row zero therefore inserts the
    // new data symbol into an empty query.
    let mut current = Vec::with_capacity(query.len() + 1);
    current.push(add_distance(
        previous[0],
        costs.insertion(data_symbol).get(),
    ));
    for (query_index, query_symbol) in query.iter().enumerate() {
        // The three predecessors consume both symbols, only the data symbol,
        // or only the query symbol, respectively. This fixes the direction as
        // wed(query, data prefix).
        let substitution = add_distance(
            previous[query_index],
            costs.substitution(query_symbol, data_symbol).get(),
        );
        let insertion = add_distance(
            previous[query_index + 1],
            costs.insertion(data_symbol).get(),
        );
        let deletion = add_distance(current[query_index], costs.deletion(query_symbol).get());
        current.push(substitution.min(insertion).min(deletion));
    }
    current
}
