//! Levenshtein edit costs.

use super::{Cost, EditCosts};

#[cfg(feature = "persist")]
mod persistence;

/// Unit costs for the Levenshtein distance.
#[derive(Debug, Clone, Copy, Default)]
pub struct LevenshteinCosts;

impl LevenshteinCosts {
    /// Creates Levenshtein edit costs.
    pub const fn new() -> Self {
        Self
    }
}

impl<T> EditCosts<T> for LevenshteinCosts
where
    T: Eq,
{
    fn substitution(&self, from: &T, to: &T) -> Cost {
        if from == to { Cost::ZERO } else { Cost::ONE }
    }

    fn deletion(&self, _token: &T) -> Cost {
        Cost::ONE
    }

    fn insertion(&self, _token: &T) -> Cost {
        Cost::ONE
    }
}

#[cfg(test)]
mod tests {
    use super::LevenshteinCosts;
    use crate::costs::{Cost, EditCosts};

    #[test]
    fn substitution_is_zero_for_equal_tokens_and_one_otherwise() {
        let costs = LevenshteinCosts::new();
        let tokyo = String::from("東京");
        let another_tokyo = String::from("東京");
        let kyoto = String::from("京都");

        assert_eq!(costs.substitution(&tokyo, &another_tokyo), Cost::ZERO);
        assert_eq!(costs.substitution(&tokyo, &kyoto), Cost::ONE);
    }

    #[test]
    fn deletion_and_insertion_have_unit_cost() {
        let costs = LevenshteinCosts::new();

        assert_eq!(costs.deletion(&'a'), Cost::ONE);
        assert_eq!(costs.insertion(&'a'), Cost::ONE);
    }
}
