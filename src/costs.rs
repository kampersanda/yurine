//! Weighted edit-operation cost abstractions.

pub mod levenshtein;

use std::cmp::Ordering;
use std::fmt;
use std::ops::Sub;

use crate::errors::{Error, Result};
use crate::types::Symbol;

/// Supplies the costs that define a weighted edit distance.
///
/// The search direction is always from the query to a data substring. Thus,
/// deletion consumes a query symbol and insertion consumes a data symbol.
pub trait EditCosts {
    /// Returns the cost of replacing `from` with `to`.
    ///
    /// If `from` and `to` are the same symbol, this must return zero.
    fn substitution(&self, from: Symbol, to: Symbol) -> Cost;

    /// Returns the cost of deleting a query symbol.
    fn deletion(&self, symbol: Symbol) -> Cost;

    /// Returns the cost of inserting a data symbol.
    fn insertion(&self, symbol: Symbol) -> Cost;
}

/// A finite, non-negative, single-precision edit cost or search threshold.
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
