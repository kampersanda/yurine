//! Levenshtein edit costs.

use super::{Cost, EditCosts};
use crate::types::Symbol;

/// Unit costs for the Levenshtein distance.
#[derive(Debug, Clone, Copy, Default)]
pub struct LevenshteinCosts;

impl LevenshteinCosts {
    /// Creates Levenshtein edit costs.
    pub const fn new() -> Self {
        Self
    }
}

impl EditCosts for LevenshteinCosts {
    fn substitution(&self, from: Symbol, to: Symbol) -> Cost {
        if from == to { Cost::ZERO } else { Cost::ONE }
    }

    fn deletion(&self, _symbol: Symbol) -> Cost {
        Cost::ONE
    }

    fn insertion(&self, _symbol: Symbol) -> Cost {
        Cost::ONE
    }
}

#[cfg(test)]
mod tests {
    use super::LevenshteinCosts;
    use crate::costs::{Cost, EditCosts};
    use crate::types::Symbol;

    #[test]
    fn substitution_is_zero_for_equal_symbols_and_one_otherwise() {
        let costs = LevenshteinCosts::new();
        let first = Symbol::new(0);
        let second = Symbol::new(1);

        assert_eq!(costs.substitution(first, first), Cost::ZERO);
        assert_eq!(costs.substitution(first, second), Cost::ONE);
    }

    #[test]
    fn deletion_and_insertion_have_unit_cost() {
        let costs = LevenshteinCosts::new();
        let symbol = Symbol::new(0);

        assert_eq!(costs.deletion(symbol), Cost::ONE);
        assert_eq!(costs.insertion(symbol), Cost::ONE);
    }
}
