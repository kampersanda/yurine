//! Widening of the substitution-neighborhood radius.

use crate::costs::{Cost, EditCosts};
use crate::search::bound::StrictBound;
use crate::search::filtering::neighborhood::SubstitutionNeighborhood;
use crate::types::Symbol;

/// Returns whether any eta could let selection construct a threshold
/// subsequence.
///
/// A query position contributes the cheapest way out of its neighborhood, so
/// widening eta grows that contribution until no substitution is left outside
/// and deletion is all that remains. Deleting the whole query sequence is
/// therefore the most the positions can ever contribute together, and a bound
/// admitting that sum admits every radius. Answering this before looking at
/// the alphabet keeps a search no radius can help from enumerating one radius
/// per query position and vocabulary symbol on its way to the same conclusion.
pub(in crate::search) fn any_radius_can_select<C>(
    query_string: &[Symbol],
    bound: StrictBound,
    costs: &C,
) -> bool
where
    C: EditCosts<Symbol>,
{
    let deletions: f32 = query_string
        .iter()
        .map(|symbol| costs.deletion(symbol).get())
        .sum();
    !bound.admits(deletions)
}

/// Returns the radii above `eta` that selection can behave differently at, in
/// increasing order.
///
/// A neighborhood only changes where eta crosses one of its substitution
/// costs, so those costs are the only radii worth trying.
pub(in crate::search) fn wider_radii<C>(
    query_string: &[Symbol],
    eta: Cost,
    costs: &C,
    neighborhood: &SubstitutionNeighborhood,
) -> Vec<Cost>
where
    C: EditCosts<Symbol>,
{
    let mut radii: Vec<Cost> = query_string
        .iter()
        .flat_map(|symbol| {
            neighborhood
                .alphabet()
                .iter()
                .map(move |candidate| costs.substitution(symbol, candidate))
        })
        .filter(|radius| *radius > eta)
        .collect();
    radii.sort_unstable();
    radii.dedup();
    radii
}

#[cfg(test)]
mod tests {
    use super::{any_radius_can_select, wider_radii};
    use crate::costs::{Cost, EditCosts};
    use crate::search::bound::StrictBound;
    use crate::search::filtering::neighborhood::SubstitutionNeighborhood;
    use crate::types::Symbol;

    /// Substitution costs rise with the target symbol, so widening eta walks a
    /// known ladder of radii.
    struct LadderCosts {
        deletion: Cost,
    }

    impl EditCosts<Symbol> for LadderCosts {
        fn substitution(&self, from: &Symbol, to: &Symbol) -> Cost {
            if from == to {
                Cost::ZERO
            } else {
                Cost::new_const(to.get() as f32 / 10.0)
            }
        }

        fn deletion(&self, _symbol: &Symbol) -> Cost {
            self.deletion
        }

        fn insertion(&self, _symbol: &Symbol) -> Cost {
            Cost::ONE
        }
    }

    fn bound(threshold: f32) -> StrictBound {
        StrictBound::from_inclusive(Cost::new(threshold).unwrap()).unwrap()
    }

    fn alphabet() -> SubstitutionNeighborhood {
        SubstitutionNeighborhood::new([Symbol::new(1), Symbol::new(2), Symbol::new(3)]).unwrap()
    }

    #[test]
    fn deleting_the_query_for_less_than_the_bound_rules_out_every_radius() {
        // Contributions stop growing at the deletion cost, so two positions
        // can never contribute more than 0.2 against a threshold of 0.5.
        let costs = LadderCosts {
            deletion: Cost::new_const(0.1),
        };

        assert!(!any_radius_can_select(
            &[Symbol::new(0), Symbol::new(0)],
            bound(0.5),
            &costs
        ));
    }

    #[test]
    fn deleting_the_query_for_more_than_the_bound_leaves_a_radius() {
        let costs = LadderCosts {
            deletion: Cost::new_const(0.4),
        };

        assert!(any_radius_can_select(
            &[Symbol::new(0), Symbol::new(0)],
            bound(0.5),
            &costs
        ));
    }

    #[test]
    fn collects_the_substitution_costs_as_radii() {
        let costs = LadderCosts {
            deletion: Cost::ONE,
        };

        // Both positions share the same ladder, so the radii are the distinct
        // substitution costs of one of them.
        assert_eq!(
            wider_radii(
                &[Symbol::new(0), Symbol::new(0)],
                Cost::ZERO,
                &costs,
                &alphabet()
            ),
            [
                Cost::new_const(0.1),
                Cost::new_const(0.2),
                Cost::new_const(0.3)
            ]
        );
    }

    #[test]
    fn offers_only_radii_above_the_configured_one() {
        let costs = LadderCosts {
            deletion: Cost::ONE,
        };

        // Selection already ran at the configured radius, so a narrower one
        // has nothing left to offer.
        assert_eq!(
            wider_radii(&[Symbol::new(0)], Cost::new_const(0.1), &costs, &alphabet()),
            [Cost::new_const(0.2), Cost::new_const(0.3)]
        );
    }
}
