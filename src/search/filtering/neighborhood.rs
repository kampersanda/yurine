//! Substitution neighborhoods.

use hashbrown::HashSet;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::types::Symbol;

/// A substitution neighborhood enumerated from a finite alphabet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::search) struct SubstitutionNeighborhood {
    alphabet: Vec<Symbol>,
}

impl SubstitutionNeighborhood {
    /// Creates a neighborhood over `alphabet`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DuplicateAlphabetSymbol`]
    /// if a symbol occurs more than once.
    pub(in crate::search) fn new<I>(alphabet: I) -> Result<Self>
    where
        I: IntoIterator<Item = Symbol>,
    {
        let alphabet: Vec<_> = alphabet.into_iter().collect();
        {
            let mut unique = HashSet::with_capacity(alphabet.len());
            for symbol in &alphabet {
                if !unique.insert(symbol) {
                    return Err(Error::DuplicateAlphabetSymbol);
                }
            }
        }
        Ok(Self { alphabet })
    }

    /// Returns the alphabet the neighborhood is enumerated from.
    pub(in crate::search) fn alphabet(&self) -> &[Symbol] {
        &self.alphabet
    }

    /// Splits the alphabet at eta in a single pass over the substitution costs.
    ///
    /// Returns the symbols whose substitution cost from `symbol` is at most
    /// eta, in alphabet order, together with the minimum cost of deleting
    /// `symbol` or substituting it with a symbol outside that neighborhood.
    /// The returned symbols are unique because the alphabet is. The supplied
    /// edit-cost policy is the same policy that verification uses.
    pub(in crate::search) fn scan<C>(
        &self,
        symbol: Symbol,
        eta: Cost,
        costs: &C,
    ) -> (Vec<Symbol>, Cost)
    where
        C: EditCosts<Symbol>,
    {
        let mut neighbors = Vec::new();
        let mut outside_cost = costs.deletion(&symbol);
        for candidate in self.alphabet.iter().copied() {
            let substitution = costs.substitution(&symbol, &candidate);
            if substitution <= eta {
                neighbors.push(candidate);
            } else {
                outside_cost = outside_cost.min(substitution);
            }
        }
        (neighbors, outside_cost)
    }
}

#[cfg(test)]
mod tests {
    use super::SubstitutionNeighborhood;
    use crate::costs::{Cost, EditCosts};
    use crate::errors::Error;
    use crate::types::Symbol;

    struct RankedCosts;

    impl EditCosts<Symbol> for RankedCosts {
        fn substitution(&self, from: &Symbol, to: &Symbol) -> Cost {
            if from == to {
                Cost::ZERO
            } else {
                Cost::new_const(to.get() as f32 / 4.0)
            }
        }

        fn deletion(&self, _symbol: &Symbol) -> Cost {
            Cost::new_const(0.9)
        }

        fn insertion(&self, _symbol: &Symbol) -> Cost {
            Cost::ONE
        }
    }

    #[test]
    fn rejects_duplicate_alphabet_symbols() {
        let symbol = Symbol::new(1);

        assert_eq!(
            SubstitutionNeighborhood::new([symbol, symbol]),
            Err(Error::DuplicateAlphabetSymbol)
        );
    }

    #[test]
    fn scan_collects_neighbors_up_to_eta_in_alphabet_order() {
        let neighborhood =
            SubstitutionNeighborhood::new([Symbol::new(3), Symbol::new(1), Symbol::new(2)])
                .unwrap();

        let (neighbors, _) = neighborhood.scan(Symbol::new(0), Cost::new_const(0.5), &RankedCosts);

        assert_eq!(neighbors, [Symbol::new(1), Symbol::new(2)]);
    }

    #[test]
    fn scan_returns_deletion_or_the_cheapest_excluded_substitution() {
        let neighborhood =
            SubstitutionNeighborhood::new([Symbol::new(1), Symbol::new(2), Symbol::new(3)])
                .unwrap();

        let (_, excluded) = neighborhood.scan(Symbol::new(0), Cost::new_const(0.5), &RankedCosts);
        let (_, deletion) = neighborhood.scan(Symbol::new(0), Cost::ONE, &RankedCosts);

        assert_eq!(excluded, Cost::new_const(0.75));
        assert_eq!(deletion, Cost::new_const(0.9));
    }

    #[test]
    fn scan_reports_an_empty_neighborhood_with_the_deletion_cost() {
        // Both substitutions exceed an eta of zero, so the neighborhood is
        // empty, and both cost more than deletion, so deletion remains the
        // cheapest way out.
        let neighborhood = SubstitutionNeighborhood::new([Symbol::new(4), Symbol::new(8)]).unwrap();

        let (neighbors, outside_cost) = neighborhood.scan(Symbol::new(0), Cost::ZERO, &RankedCosts);

        assert!(neighbors.is_empty());
        assert_eq!(outside_cost, Cost::new_const(0.9));
    }
}
