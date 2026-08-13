//! Customizable edit costs.

use std::collections::HashMap;
use std::hash::Hash;

use super::{Cost, EditCosts};

#[cfg(feature = "persist")]
mod persistence;

/// Editable costs for weighted edit distance.
///
/// Each operation initially uses its default cost. Token-specific costs can be
/// added or replaced with the setter methods.
/// [`Default`] uses unit costs equivalent to the Levenshtein distance.
///
/// Rules are directional because edits transform the query into a data
/// segment. For example, a substitution from `'a'` to `'A'` does not also set
/// the cost from `'A'` to `'a'`.
///
/// ```
/// use yurine::costs::{Cost, EditCosts, custom::CustomCosts};
///
/// let mut costs = CustomCosts::default();
/// costs.set_substitution('a', 'A', Cost::new_const(0.25));
/// costs.set_deletion('.', Cost::new_const(0.1));
///
/// assert_eq!(costs.substitution(&'a', &'A'), Cost::new_const(0.25));
/// assert_eq!(costs.substitution(&'A', &'a'), Cost::ONE);
/// assert_eq!(costs.deletion(&'.'), Cost::new_const(0.1));
/// ```
#[derive(Debug, Clone)]
pub struct CustomCosts<T> {
    default_substitution: Cost,
    default_deletion: Cost,
    default_insertion: Cost,
    substitutions: HashMap<T, HashMap<T, Cost>>,
    deletions: HashMap<T, Cost>,
    insertions: HashMap<T, Cost>,
}

impl<T> CustomCosts<T> {
    /// Creates editable costs with the supplied operation defaults.
    ///
    /// The substitution default applies only to different tokens. Replacing a
    /// token with itself always costs zero.
    pub fn new(substitution: Cost, deletion: Cost, insertion: Cost) -> Self {
        Self {
            default_substitution: substitution,
            default_deletion: deletion,
            default_insertion: insertion,
            substitutions: HashMap::new(),
            deletions: HashMap::new(),
            insertions: HashMap::new(),
        }
    }

    /// Changes the default substitution cost.
    pub fn set_default_substitution(&mut self, cost: Cost) {
        self.default_substitution = cost;
    }

    /// Changes the default deletion cost.
    pub fn set_default_deletion(&mut self, cost: Cost) {
        self.default_deletion = cost;
    }

    /// Changes the default insertion cost.
    pub fn set_default_insertion(&mut self, cost: Cost) {
        self.default_insertion = cost;
    }
}

impl<T> CustomCosts<T>
where
    T: Eq + Hash,
{
    /// Sets the cost of replacing `from` with `to`.
    ///
    /// An override for equal tokens is ignored because equal-token
    /// substitution always costs zero.
    pub fn set_substitution(&mut self, from: T, to: T, cost: Cost) {
        if from != to {
            self.substitutions.entry(from).or_default().insert(to, cost);
        }
    }

    /// Sets the cost of deleting `token`.
    pub fn set_deletion(&mut self, token: T, cost: Cost) {
        self.deletions.insert(token, cost);
    }

    /// Sets the cost of inserting `token`.
    pub fn set_insertion(&mut self, token: T, cost: Cost) {
        self.insertions.insert(token, cost);
    }
}

impl<T> Default for CustomCosts<T> {
    fn default() -> Self {
        Self::new(Cost::ONE, Cost::ONE, Cost::ONE)
    }
}

impl<T> EditCosts<T> for CustomCosts<T>
where
    T: Eq + Hash,
{
    fn substitution(&self, from: &T, to: &T) -> Cost {
        if from == to {
            return Cost::ZERO;
        }

        self.substitutions
            .get(from)
            .and_then(|costs| costs.get(to))
            .copied()
            .unwrap_or(self.default_substitution)
    }

    fn deletion(&self, token: &T) -> Cost {
        self.deletions
            .get(token)
            .copied()
            .unwrap_or(self.default_deletion)
    }

    fn insertion(&self, token: &T) -> Cost {
        self.insertions
            .get(token)
            .copied()
            .unwrap_or(self.default_insertion)
    }
}

#[cfg(test)]
mod tests {
    use super::CustomCosts;
    use crate::costs::{Cost, EditCosts};

    #[test]
    fn uses_operation_defaults() {
        let costs = CustomCosts::<char>::new(
            Cost::new_const(2.0),
            Cost::new_const(3.0),
            Cost::new_const(4.0),
        );

        assert_eq!(costs.substitution(&'a', &'a'), Cost::ZERO);
        assert_eq!(costs.substitution(&'a', &'b'), Cost::new_const(2.0));
        assert_eq!(costs.deletion(&'a'), Cost::new_const(3.0));
        assert_eq!(costs.insertion(&'a'), Cost::new_const(4.0));
    }

    #[test]
    fn edits_default_costs() {
        let mut costs = CustomCosts::<char>::default();

        costs.set_default_substitution(Cost::new_const(2.0));
        costs.set_default_deletion(Cost::new_const(3.0));
        costs.set_default_insertion(Cost::new_const(4.0));

        assert_eq!(costs.substitution(&'a', &'b'), Cost::new_const(2.0));
        assert_eq!(costs.deletion(&'a'), Cost::new_const(3.0));
        assert_eq!(costs.insertion(&'a'), Cost::new_const(4.0));
    }

    #[test]
    fn edits_token_specific_costs() {
        let mut costs = CustomCosts::<char>::default();

        costs.set_substitution('a', 'A', Cost::new_const(0.25));
        costs.set_deletion('.', Cost::new_const(0.5));
        costs.set_insertion('-', Cost::new_const(0.75));

        assert_eq!(costs.substitution(&'a', &'A'), Cost::new_const(0.25));
        assert_eq!(costs.substitution(&'A', &'a'), Cost::ONE);
        assert_eq!(costs.deletion(&'.'), Cost::new_const(0.5));
        assert_eq!(costs.deletion(&'a'), Cost::ONE);
        assert_eq!(costs.insertion(&'-'), Cost::new_const(0.75));
        assert_eq!(costs.insertion(&'a'), Cost::ONE);
    }

    #[test]
    fn equal_token_substitution_always_costs_zero() {
        let mut costs = CustomCosts::<char>::default();

        costs.set_substitution('a', 'a', Cost::ONE);

        assert_eq!(costs.substitution(&'a', &'a'), Cost::ZERO);
    }
}
