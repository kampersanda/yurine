//! MinCand threshold-subsequence selection.

use std::hash::Hash;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::postings::PostingsIndex;
use crate::search::filtering::neighborhood::SubstitutionNeighborhood;
use crate::types::Position;

/// The two-approximation MinCand selector described in the design material.
#[derive(Debug, Clone, Copy, Default)]
pub struct MinCandidateSelector;

impl MinCandidateSelector {
    /// Selects a subsequence complete for matches with distance at most
    /// `threshold`.
    pub fn select<Symbol, Index, Costs>(
        &self,
        query: &[Symbol],
        threshold: Cost,
        eta: Cost,
        index: &Index,
        costs: &Costs,
        neighborhood: &SubstitutionNeighborhood<Symbol>,
    ) -> Result<Vec<Position>>
    where
        Symbol: Eq + Hash + Clone,
        Index: PostingsIndex<Symbol>,
        Costs: EditCosts<Symbol>,
    {
        let threshold = threshold.next_up()?;
        struct Item {
            contribution: f32,
            candidate_count: f32,
            paid: f32,
            selected: bool,
        }

        let mut items = Vec::with_capacity(query.len());
        for symbol in query {
            let contribution = neighborhood.minimum_outside_cost(symbol, eta, costs).get();

            let mut candidate_count = 0usize;
            for neighbor in neighborhood.neighbors(symbol, eta, costs) {
                candidate_count = candidate_count
                    .checked_add(index.frequency(&neighbor)?)
                    .ok_or(Error::ThresholdSubsequenceUnavailable)?;
            }

            items.push(Item {
                contribution,
                candidate_count: candidate_count as f32,
                paid: 0.0,
                selected: false,
            });
        }

        let mut selected = Vec::new();
        let mut selected_contribution = 0.0;

        while selected_contribution < threshold {
            let residual = threshold - selected_contribution;
            let mut best: Option<(usize, f32)> = None;

            for (position, item) in items.iter().enumerate() {
                if item.selected {
                    continue;
                }

                let gain = item.contribution.min(residual);
                if gain <= 0.0 {
                    continue;
                }

                let unpaid = (item.candidate_count - item.paid).max(0.0);
                let value = unpaid / gain;
                if best.is_none_or(|(_, best_value)| value < best_value) {
                    best = Some((position, value));
                }
            }

            let (best_position, best_value) = best.ok_or(Error::ThresholdSubsequenceUnavailable)?;

            for item in items.iter_mut().filter(|item| !item.selected) {
                let gain = item.contribution.min(residual);
                if gain > 0.0 {
                    item.paid += gain * best_value;
                }
            }

            items[best_position].selected = true;
            selected_contribution += items[best_position].contribution;
            selected.push(Position::from_usize(best_position)?);
        }

        Ok(selected)
    }
}
