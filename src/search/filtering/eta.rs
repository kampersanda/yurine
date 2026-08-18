//! Re-tuning of the substitution-neighborhood radius.

use crate::costs::{Cost, EditCosts};
use crate::search::bound::StrictBound;
use crate::search::filtering::neighborhood::SubstitutionNeighborhood;
use crate::types::Symbol;

/// Returns the smallest eta above `eta` that selection can work with, or
/// `None` when no eta can.
///
/// Selection needs the query positions' contributions to together leave the
/// bound. A contribution is the cheapest way out of its neighborhood, so it
/// grows as eta widens the neighborhood, and it stops growing at the
/// position's deletion cost once no substitution is left outside. No eta can
/// therefore help a query whose deletion costs together stay within the bound,
/// which is what makes exhaustive verification the last resort rather than the
/// first answer.
///
/// The neighborhoods only change where eta crosses a substitution cost, so the
/// smallest sufficient radius is one of those costs. Finding it binary searches
/// them, which costs a few alphabet scans against the exhaustive verification
/// it avoids.
pub(in crate::search) fn smallest_selectable_eta<C>(
    query_string: &[Symbol],
    bound: StrictBound,
    eta: Cost,
    costs: &C,
    neighborhood: &SubstitutionNeighborhood,
) -> Option<Cost>
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

    let selectable =
        radii.partition_point(|radius| !selects(query_string, bound, *radius, costs, neighborhood));
    radii.get(selectable).copied()
}

/// Returns whether the contributions at `eta` together leave the bound.
fn selects<C>(
    query_string: &[Symbol],
    bound: StrictBound,
    eta: Cost,
    costs: &C,
    neighborhood: &SubstitutionNeighborhood,
) -> bool
where
    C: EditCosts<Symbol>,
{
    let contributions: f32 = query_string
        .iter()
        .map(|symbol| neighborhood.outside_cost(*symbol, eta, costs).get())
        .sum();
    !bound.admits(contributions)
}

#[cfg(test)]
mod tests {
    use super::smallest_selectable_eta;
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
    fn finds_the_smallest_radius_that_leaves_the_bound() {
        // At eta zero the cheapest excluded substitution costs 0.1, which the
        // threshold 0.25 admits. Raising eta past 0.1 and then 0.2 leaves 0.3
        // as the cheapest way out, which the threshold no longer admits.
        let costs = LadderCosts {
            deletion: Cost::ONE,
        };

        assert_eq!(
            smallest_selectable_eta(
                &[Symbol::new(0)],
                bound(0.25),
                Cost::ZERO,
                &costs,
                &alphabet()
            ),
            Some(Cost::new_const(0.2))
        );
    }

    #[test]
    fn reports_no_radius_when_deleting_the_query_stays_within_the_bound() {
        // Contributions stop growing at the deletion cost, so two positions
        // can never contribute more than 0.2 against a threshold of 0.5.
        let costs = LadderCosts {
            deletion: Cost::new_const(0.1),
        };

        assert_eq!(
            smallest_selectable_eta(
                &[Symbol::new(0), Symbol::new(0)],
                bound(0.5),
                Cost::ZERO,
                &costs,
                &alphabet()
            ),
            None
        );
    }

    #[test]
    fn widens_past_every_substitution_when_only_deletion_leaves_the_bound() {
        // Excluding the dearest substitution leaves 0.3 per position, which the
        // threshold 1.0 still admits. Only deleting both positions, at 2.0
        // each, leaves the bound, and that needs an eta excluding nothing.
        let costs = LadderCosts {
            deletion: Cost::new_const(2.0),
        };

        assert_eq!(
            smallest_selectable_eta(
                &[Symbol::new(0), Symbol::new(0)],
                bound(1.0),
                Cost::ZERO,
                &costs,
                &alphabet()
            ),
            Some(Cost::new_const(0.3))
        );
    }

    #[test]
    fn only_considers_radii_above_the_configured_one() {
        let costs = LadderCosts {
            deletion: Cost::ONE,
        };

        // The threshold admits nothing, so 0.1 would select too. Re-tuning
        // answers a radius that could not select, and widening it is the only
        // move that keeps the caller's radius meaningful.
        let radius = smallest_selectable_eta(
            &[Symbol::new(0)],
            bound(0.0),
            Cost::new_const(0.2),
            &costs,
            &alphabet(),
        );

        assert_eq!(radius, Some(Cost::new_const(0.3)));
    }
}
