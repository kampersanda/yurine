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
    pub fn new<Alphabet>(alphabet: Alphabet) -> Result<Self>
    where
        Alphabet: IntoIterator<Item = Symbol>,
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
    pub fn neighbors<Costs>(&self, symbol: Symbol, eta: Cost, costs: &Costs) -> Vec<Symbol>
    where
        Costs: EditCosts<Symbol>,
    {
        self.alphabet
            .iter()
            .copied()
            .filter(|candidate| costs.substitution(&symbol, candidate) <= eta)
            .collect()
    }

    /// Returns the minimum cost of deleting `symbol` or substituting it with a
    /// symbol outside its neighborhood.
    pub fn minimum_outside_cost<Costs>(&self, symbol: Symbol, eta: Cost, costs: &Costs) -> Cost
    where
        Costs: EditCosts<Symbol>,
    {
        self.alphabet
            .iter()
            .copied()
            .map(|candidate| costs.substitution(&symbol, &candidate))
            .filter(|substitution| *substitution > eta)
            .fold(costs.deletion(&symbol), Cost::min)
    }
}
