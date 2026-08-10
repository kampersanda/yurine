//! Substitution neighborhoods.

use std::collections::HashSet;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::types::Symbol;

/// A substitution neighborhood enumerated from a finite alphabet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubstitutionNeighborhood {
    alphabet: Vec<Symbol>,
}

impl SubstitutionNeighborhood {
    /// Creates a neighborhood over `alphabet`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DuplicateAlphabetSymbol`]
    /// if a symbol occurs more than once.
    pub fn new<I>(alphabet: I) -> Result<Self>
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

    /// Visits symbols whose substitution cost from `symbol` is at most eta.
    ///
    /// The returned symbols must be unique. The supplied edit-cost policy is
    /// the same policy that verification uses.
    pub fn neighbors<C>(&self, symbol: Symbol, eta: Cost, costs: &C) -> Vec<Symbol>
    where
        C: EditCosts<Symbol>,
    {
        self.alphabet
            .iter()
            .copied()
            .filter(|candidate| costs.substitution(&symbol, candidate) <= eta)
            .collect()
    }

    /// Returns the minimum cost of deleting `symbol` or substituting it with a
    /// symbol outside its neighborhood.
    pub fn minimum_outside_cost<C>(&self, symbol: Symbol, eta: Cost, costs: &C) -> Cost
    where
        C: EditCosts<Symbol>,
    {
        self.alphabet
            .iter()
            .copied()
            .map(|candidate| costs.substitution(&symbol, &candidate))
            .filter(|substitution| *substitution > eta)
            .fold(costs.deletion(&symbol), Cost::min)
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
    fn neighbors_include_the_eta_boundary_in_alphabet_order() {
        let neighborhood =
            SubstitutionNeighborhood::new([Symbol::new(3), Symbol::new(1), Symbol::new(2)])
                .unwrap();

        assert_eq!(
            neighborhood.neighbors(Symbol::new(0), Cost::new_const(0.5), &RankedCosts),
            [Symbol::new(1), Symbol::new(2)]
        );
    }

    #[test]
    fn minimum_outside_cost_uses_deletion_or_the_cheapest_excluded_substitution() {
        let neighborhood =
            SubstitutionNeighborhood::new([Symbol::new(1), Symbol::new(2), Symbol::new(3)])
                .unwrap();

        assert_eq!(
            neighborhood.minimum_outside_cost(Symbol::new(0), Cost::new_const(0.5), &RankedCosts,),
            Cost::new_const(0.75)
        );
        assert_eq!(
            neighborhood.minimum_outside_cost(Symbol::new(0), Cost::ONE, &RankedCosts),
            Cost::new_const(0.9)
        );
    }
}
