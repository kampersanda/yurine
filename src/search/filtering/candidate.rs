//! MinCand threshold-subsequence selection.

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::postings::PostingsIndex;
use crate::search::bound::StrictBound;
use crate::search::filtering::neighborhood::SubstitutionNeighborhood;
use crate::types::{Position, Symbol};

/// A selected query position together with the neighborhood it was scored on.
///
/// Candidate generation reuses `neighbors` instead of scanning the alphabet
/// again, so the two phases cannot disagree about the neighborhood.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::search) struct SelectedPosition {
    pub(in crate::search) position: Position,
    pub(in crate::search) neighbors: Vec<Symbol>,
}

/// The two-approximation MinCand selector described in the design material.
#[derive(Debug, Clone, Copy, Default)]
pub(in crate::search) struct MinCandidateSelector;

impl MinCandidateSelector {
    /// Selects a subsequence complete for the matches `bound` admits.
    ///
    /// Positions are returned in selection order, each carrying its
    /// substitution neighborhood.
    ///
    /// Selection succeeds exactly when the positions' contributions together
    /// leave `bound`, and the contributions are added in query position order
    /// so that a caller weighing the same sum reaches the same answer.
    ///
    /// For query-string length `m` and alphabet size `a`, scoring the query
    /// positions scans the alphabet once each and reads a posting-list length
    /// per neighbor, and selection then makes at most `m` rounds over the
    /// positions. This takes `O(m * a + m^2)` time and holds `O(m * a)`
    /// neighborhood symbols. Corpus size enters only through posting-list
    /// lengths, which are read in constant time.
    pub(in crate::search) fn select<C>(
        &self,
        query_string: &[Symbol],
        bound: StrictBound,
        eta: Cost,
        index: &PostingsIndex,
        costs: &C,
        neighborhood: &SubstitutionNeighborhood,
    ) -> Result<Vec<SelectedPosition>>
    where
        C: EditCosts<Symbol>,
    {
        struct Item {
            neighbors: Vec<Symbol>,
            contribution: f32,
            candidate_count: f32,
            paid: f32,
            selected: bool,
        }

        let mut items = Vec::with_capacity(query_string.len());
        for symbol in query_string {
            let (neighbors, outside_cost) = neighborhood.scan(*symbol, eta, costs);

            let mut candidate_count = 0usize;
            for neighbor in &neighbors {
                candidate_count = candidate_count
                    .checked_add(index.frequency(*neighbor))
                    .ok_or(Error::ThresholdSubsequenceUnavailable)?;
            }

            items.push(Item {
                neighbors,
                contribution: outside_cost.get(),
                candidate_count: candidate_count as f32,
                paid: 0.0,
                selected: false,
            });
        }

        let mut selected = Vec::new();
        let mut selected_contribution = 0.0;

        // Selection continues until the contributions together leave the
        // bound, so a match the bound admits cannot avoid every selected
        // position.
        while bound.admits(selected_contribution) {
            let residual = bound.residual(selected_contribution);
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

            let best = &mut items[best_position];
            best.selected = true;
            selected.push(SelectedPosition {
                position: Position::from_usize(best_position)?,
                // Handing the neighborhood over also releases it from `items`,
                // so only the selected positions keep their lists alive.
                neighbors: std::mem::take(&mut best.neighbors),
            });

            // Adding the contributions in query position order, rather than in
            // the order they were selected, keeps the total here equal to the
            // one the caller checked the query against before searching. The
            // two sums hold the same terms, and in `f32` the order they are
            // added in decides the result: a term far below the running total
            // vanishes into it, and whether the sum ends above the bound can
            // turn on that. Once every position is selected the two agree, so
            // a query the caller admitted cannot be reported unselectable.
            selected_contribution = items
                .iter()
                .filter(|item| item.selected)
                .map(|item| item.contribution)
                .sum();
        }

        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::{MinCandidateSelector, SelectedPosition};
    use crate::costs::levenshtein::LevenshteinCosts;
    use crate::costs::{Cost, EditCosts};
    use crate::errors::Error;
    use crate::postings::PostingsIndexBuilder;
    use crate::search::bound::StrictBound;
    use crate::search::filtering::neighborhood::SubstitutionNeighborhood;
    use crate::types::{Position, Posting, SequenceId, Symbol};

    /// The bound admitting exactly the distances at most `threshold`.
    fn bound(threshold: f32) -> StrictBound {
        StrictBound::from_inclusive(Cost::new(threshold).unwrap()).unwrap()
    }

    fn add_occurrences(builder: &mut PostingsIndexBuilder, symbol: Symbol, count: u32) {
        for position in 0..count {
            builder
                .add_posting(
                    symbol,
                    Posting {
                        string_id: SequenceId::new(0),
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
                bound(0.5),
                Cost::ZERO,
                &index.build(),
                &LevenshteinCosts,
                &neighborhood,
            )
            .unwrap();

        // The selected position carries the neighborhood it was scored on, so
        // candidate generation does not scan the alphabet again.
        assert_eq!(
            selected,
            [SelectedPosition {
                position: Position::new(1),
                neighbors: vec![rare],
            }]
        );
    }

    #[test]
    fn selects_enough_positions_to_exceed_the_inclusive_threshold() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let neighborhood = SubstitutionNeighborhood::new([first, second]).unwrap();

        let selected = MinCandidateSelector
            .select(
                &[first, second],
                bound(1.0),
                Cost::ZERO,
                &PostingsIndexBuilder::new(0).build(),
                &LevenshteinCosts,
                &neighborhood,
            )
            .unwrap();

        // Every selection round moves its neighborhood out of the scored
        // items, so a later round must still hand over its own list.
        assert_eq!(
            selected,
            [
                SelectedPosition {
                    position: Position::new(0),
                    neighbors: vec![first],
                },
                SelectedPosition {
                    position: Position::new(1),
                    neighbors: vec![second],
                },
            ]
        );
    }

    /// Deletion costs that only reach the threshold when added in query order.
    ///
    /// Two of them are half an ulp of the threshold, so each vanishes into a
    /// running total that already holds the third.
    struct VanishingCosts;

    impl EditCosts<Symbol> for VanishingCosts {
        fn substitution(&self, from: &Symbol, to: &Symbol) -> Cost {
            if from == to { Cost::ZERO } else { Cost::ONE }
        }

        fn deletion(&self, symbol: &Symbol) -> Cost {
            if symbol.get() == 2 {
                Cost::ONE
            } else {
                Cost::new_const(f32::from_bits(0x3380_0000))
            }
        }

        fn insertion(&self, _symbol: &Symbol) -> Cost {
            Cost::ONE
        }
    }

    #[test]
    fn selects_a_query_whose_contributions_only_reach_the_bound_in_order() {
        let (first, second, third) = (Symbol::new(0), Symbol::new(1), Symbol::new(2));
        let mut index = PostingsIndexBuilder::new(3);
        // The dearest position is the cheapest to generate candidates from, so
        // selection takes it first and the two small ones follow.
        add_occurrences(&mut index, first, 4);
        add_occurrences(&mut index, second, 4);
        add_occurrences(&mut index, third, 1);
        let neighborhood = SubstitutionNeighborhood::new([first, second, third]).unwrap();

        let selected = MinCandidateSelector
            .select(
                &[first, second, third],
                bound(1.0),
                Cost::ZERO,
                &index.build(),
                &VanishingCosts,
                &neighborhood,
            )
            .unwrap();

        // Range search admits this query by adding the same deletion costs in
        // the same order, so selection has to agree rather than report the
        // subsequence unavailable.
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn reports_when_total_contribution_cannot_exceed_threshold() {
        let symbol = Symbol::new(0);
        let neighborhood = SubstitutionNeighborhood::new([symbol]).unwrap();

        let result = MinCandidateSelector.select(
            &[symbol],
            bound(1.0),
            Cost::ZERO,
            &PostingsIndexBuilder::new(0).build(),
            &LevenshteinCosts,
            &neighborhood,
        );

        assert_eq!(result, Err(Error::ThresholdSubsequenceUnavailable));
    }
}
