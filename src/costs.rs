//! Weighted edit-operation costs used during search.
//!
//! Start with [`LevenshteinCosts`] for ordinary edit distance.
//! Use [`CustomCosts`] for token-specific rules, or implement
//! [`EditCosts`] when costs must be computed dynamically.

pub mod custom;
pub mod embedding;
pub mod levenshtein;
#[cfg(feature = "persist")]
pub(crate) mod persistence;

pub use custom::CustomCosts;
pub use embedding::{CosineEmbeddingCosts, EmbeddingStore, EmbeddingStoreBuilder};
pub use levenshtein::LevenshteinCosts;

use std::cmp::Ordering;
use std::fmt;
use std::ops::Sub;

use crate::errors::{Error, Result};
/// Supplies the costs that define a weighted edit distance.
///
/// The search direction is always from the query sequence to a data segment.
/// Thus, deletion consumes a query token and insertion consumes a
/// data token.
///
/// # Substitution against deletion and insertion
///
/// The three operations need not be related in any particular way, and results
/// are exact whatever their costs. The relationship does decide how a search
/// is answered, and therefore how long it takes.
///
/// Filtering yields data segments anchored on a token pair, so verifying them
/// only inspects alignments that pair at least one query token with one data
/// token. Deleting a query token and inserting a data token can be replaced by
/// substituting one for the other without increasing the distance exactly when
/// `substitution(from, to) <= deletion(from) + insertion(to)`, which is what
/// makes an anchored alignment optimal.
///
/// Where that inequality does not hold, the cheapest alignment of a segment
/// may pair no tokens at all. Such an alignment deletes every query token, so
/// it costs at least the sum of those deletions, and a search whose threshold
/// reaches that sum is answered by exhaustive verification instead of by
/// filtering. Costs that make deletion and insertion much cheaper than
/// substitution therefore reach that slower path more often; see
/// [`RangeSearcher::search`](crate::search::RangeSearcher::search).
pub trait EditCosts<T> {
    /// Returns the cost of replacing `from` with `to`.
    ///
    /// If `from` and `to` are the same token, this must return zero.
    fn substitution(&self, from: &T, to: &T) -> Cost;

    /// Returns the cost of deleting a query token.
    fn deletion(&self, token: &T) -> Cost;

    /// Returns the cost of inserting a data token.
    fn insertion(&self, token: &T) -> Cost;
}

/// A finite, non-negative, single-precision edit cost or search threshold.
///
/// Use [`Cost::new`] for runtime values and [`Cost::new_const`] for constants.
/// Both reject negative and non-finite values.
///
/// ```
/// use yurine::costs::Cost;
///
/// let threshold = Cost::new(0.75)?;
/// assert_eq!(threshold.get(), 0.75);
/// assert!(Cost::new(f32::NAN).is_err());
/// # Ok::<(), yurine::errors::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost(f32);

impl Cost {
    /// Zero cost.
    pub const ZERO: Self = Self(0.0);

    /// Unit cost.
    pub const ONE: Self = Self(1.0);

    /// The largest representable cost.
    pub const MAX: Self = Self(f32::MAX);

    /// Creates a finite, non-negative cost.
    pub const fn new(value: f32) -> Result<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::InvalidCost(value));
        }
        Ok(if value == 0.0 {
            Self::ZERO
        } else {
            Self(value)
        })
    }

    /// Creates a cost in a constant context.
    ///
    /// # Panics
    ///
    /// Panics if `value` is negative, infinite, or NaN.
    pub const fn new_const(value: f32) -> Self {
        if !value.is_finite() || value < 0.0 {
            panic!("cost must be finite and non-negative");
        }
        if value == 0.0 {
            Self::ZERO
        } else {
            Self(value)
        }
    }

    /// Returns the underlying finite, non-negative value.
    pub const fn get(self) -> f32 {
        self.0
    }

    /// Returns the smaller of two costs.
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// Returns the larger of two costs.
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// Compares two costs in total order.
    pub fn total_cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }

    /// Returns the next representable cost above this one.
    ///
    /// # Errors
    ///
    /// Returns an error if this cost is [`Cost::MAX`].
    pub fn next_up(self) -> Result<Self> {
        if self == Self::MAX {
            return Err(Error::InvalidCost(self.0));
        }
        Ok(Self(self.0.next_up()))
    }
}

impl Eq for Cost {}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

impl fmt::Display for Cost {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl TryFrom<f32> for Cost {
    type Error = Error;

    fn try_from(value: f32) -> Result<Self> {
        Self::new(value)
    }
}

impl From<Cost> for f32 {
    fn from(cost: Cost) -> Self {
        cost.get()
    }
}

impl PartialEq<f32> for Cost {
    fn eq(&self, other: &f32) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Cost> for f32 {
    fn eq(&self, other: &Cost) -> bool {
        *self == other.0
    }
}

impl PartialOrd<f32> for Cost {
    fn partial_cmp(&self, other: &f32) -> Option<Ordering> {
        self.0.partial_cmp(other)
    }
}

impl PartialOrd<Cost> for f32 {
    fn partial_cmp(&self, other: &Cost) -> Option<Ordering> {
        self.partial_cmp(&other.0)
    }
}

impl Sub<f32> for Cost {
    type Output = f32;

    fn sub(self, other: f32) -> f32 {
        self.0 - other
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::{Cost, EditCosts};
    use crate::errors::Error;

    struct CharacterCosts;

    impl EditCosts<char> for CharacterCosts {
        fn substitution(&self, from: &char, to: &char) -> Cost {
            if from == to {
                Cost::ZERO
            } else if from.eq_ignore_ascii_case(to) {
                Cost::new_const(0.5)
            } else {
                Cost::ONE
            }
        }

        fn deletion(&self, token: &char) -> Cost {
            if token.is_ascii_punctuation() {
                Cost::new_const(0.25)
            } else {
                Cost::ONE
            }
        }

        fn insertion(&self, token: &char) -> Cost {
            self.deletion(token)
        }
    }

    #[test]
    fn edit_costs_can_depend_on_tokens() {
        let costs = CharacterCosts;

        assert_eq!(costs.substitution(&'a', &'A'), 0.5);
        assert_eq!(costs.deletion(&'.'), 0.25);
        assert_eq!(costs.insertion(&'a'), Cost::ONE);
    }

    #[test]
    fn accepts_finite_non_negative_costs_and_normalizes_negative_zero() {
        assert_eq!(Cost::new(1.25).unwrap().get(), 1.25);
        assert_eq!(Cost::new(-0.0).unwrap(), Cost::ZERO);
        assert_eq!(Cost::new(-0.0).unwrap().get().to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn rejects_negative_and_non_finite_costs() {
        for value in [-0.1, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert!(matches!(Cost::new(value), Err(Error::InvalidCost(_))));
        }
    }

    #[test]
    fn compares_costs_in_numeric_order() {
        let low = Cost::new_const(0.25);
        let high = Cost::new_const(0.75);

        assert_eq!(low.cmp(&high), Ordering::Less);
        assert_eq!(low.min(high), low);
        assert_eq!(low.max(high), high);
        assert!(low < 0.5);
        assert!(0.5 < high);
        assert_eq!(high - 0.25, 0.5);
    }

    #[test]
    fn next_up_returns_the_smallest_larger_cost() {
        let next = Cost::ONE.next_up().unwrap();

        assert!(next > Cost::ONE);
        assert_eq!(next.get().to_bits(), Cost::ONE.get().to_bits() + 1);
        assert!(matches!(Cost::MAX.next_up(), Err(Error::InvalidCost(value)) if value == f32::MAX));
    }
}
