//! MinCand threshold-subsequence selection.

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::postings::PostingsIndex;
use crate::search::filtering::neighborhood::SubstitutionNeighborhood;
use crate::types::{Position, Symbol};

/// The two-approximation MinCand selector described in the design material.
#[derive(Debug, Clone, Copy, Default)]
pub struct MinCandidateSelector;

impl MinCandidateSelector {
    /// Selects a subsequence complete for matches with distance at most
    /// `threshold`.
    pub fn select<C>(
        &self,
        query: &[Symbol],
        threshold: Cost,
        eta: Cost,
        index: &PostingsIndex,
        costs: &C,
        neighborhood: &SubstitutionNeighborhood,
    ) -> Result<Vec<Position>>
    where
        C: EditCosts<Symbol>,
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
            let contribution = neighborhood.minimum_outside_cost(*symbol, eta, costs).get();

            let mut candidate_count = 0usize;
            for neighbor in neighborhood.neighbors(*symbol, eta, costs) {
                candidate_count = candidate_count
                    .checked_add(index.frequency(neighbor))
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

#[cfg(test)]
mod tests {
    use super::MinCandidateSelector;
    use crate::costs::Cost;
    use crate::costs::levenshtein::LevenshteinCosts;
    use crate::errors::Error;
    use crate::postings::PostingsIndexBuilder;
    use crate::search::filtering::neighborhood::SubstitutionNeighborhood;
    use crate::types::{Position, Posting, StringId, Symbol};

    fn add_occurrences(builder: &mut PostingsIndexBuilder, symbol: Symbol, count: u32) {
        for position in 0..count {
            builder
                .add_posting(
                    symbol,
                    Posting {
                        string_id: StringId::new(0),
                        position: Position::new(position),
                    },
                )
                .unwrap();
        }
    }

    #[test]
    fn selects_the_query_position_with_fewer_candidates() {
        let common = Symbol::new(0);
        let rare = Symbol::new(1);
        let mut index = PostingsIndexBuilder::new(2);
        add_occurrences(&mut index, common, 3);
        add_occurrences(&mut index, rare, 1);
        let neighborhood = SubstitutionNeighborhood::new([common, rare]).unwrap();

        let selected = MinCandidateSelector
            .select(
                &[common, rare],
                Cost::new_const(0.5),
                Cost::ZERO,
                &index.build(),
                &LevenshteinCosts,
                &neighborhood,
            )
            .unwrap();

        assert_eq!(selected, [Position::new(1)]);
    }

    #[test]
    fn selects_enough_positions_to_exceed_the_inclusive_threshold() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let neighborhood = SubstitutionNeighborhood::new([first, second]).unwrap();

        let selected = MinCandidateSelector
            .select(
                &[first, second],
                Cost::ONE,
                Cost::ZERO,
                &PostingsIndexBuilder::new(0).build(),
                &LevenshteinCosts,
                &neighborhood,
            )
            .unwrap();

        assert_eq!(selected, [Position::new(0), Position::new(1)]);
    }

    #[test]
    fn reports_when_total_contribution_cannot_exceed_threshold() {
        let symbol = Symbol::new(0);
        let neighborhood = SubstitutionNeighborhood::new([symbol]).unwrap();

        let result = MinCandidateSelector.select(
            &[symbol],
            Cost::ONE,
            Cost::ZERO,
            &PostingsIndexBuilder::new(0).build(),
            &LevenshteinCosts,
            &neighborhood,
        );

        assert_eq!(result, Err(Error::ThresholdSubsequenceUnavailable));
    }
}
