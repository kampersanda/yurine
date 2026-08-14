//! The strict distance bound the core search algorithms work with.

use crate::costs::Cost;
use crate::errors::{Error, Result};

/// A strict upper bound on distances: a distance qualifies when it is less
/// than the bound.
///
/// The public API returns every segment whose distance is at most the search
/// threshold, while the core algorithms follow the original definitions and
/// search for distances strictly below a bound. [`Self::from_inclusive`]
/// applies the single step that relates the two, and is the only way to build
/// this type. Everything below the public boundary passes the bound around, so
/// the conversion cannot be skipped, repeated, or confused with an ordinary
/// [`Cost`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::search) struct StrictBound(Cost);

impl StrictBound {
    /// Returns the bound admitting exactly the distances at most `threshold`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidThreshold`] if `threshold` is [`Cost::MAX`],
    /// which has no representable cost above it to bound against.
    pub(in crate::search) fn from_inclusive(threshold: Cost) -> Result<Self> {
        threshold
            .next_up()
            .map(Self)
            .ok_or(Error::InvalidThreshold(threshold.get()))
    }

    /// Returns whether `distance` satisfies the bound.
    pub(in crate::search) fn admits(self, distance: f32) -> bool {
        distance < self.0.get()
    }

    /// Returns the budget that remains once `paid` has been spent.
    ///
    /// A distance satisfies the bound exactly when the part beyond `paid` is
    /// less than the returned budget. The budget stays positive as long as
    /// `paid` itself is admitted, including where `paid` equals the inclusive
    /// threshold this bound was built from. Subtracting from that threshold
    /// instead would reach zero there and report the budget as spent while a
    /// result satisfying the threshold is still reachable.
    pub(in crate::search) fn residual(self, paid: f32) -> f32 {
        self.0.get() - paid
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::StrictBound;
    use crate::costs::Cost;
    use crate::errors::Error;

    #[rstest]
    #[case(0.0)]
    #[case(0.25)]
    #[case(1.0)]
    #[case(3.5)]
    fn admits_exactly_the_distances_at_most_the_inclusive_threshold(#[case] threshold: f32) {
        let inclusive = Cost::new(threshold).unwrap();
        let bound = StrictBound::from_inclusive(inclusive).unwrap();

        assert!(bound.admits(threshold));
        assert!(!bound.admits(inclusive.next_up().unwrap().get()));
    }

    #[test]
    fn residual_stays_positive_at_the_inclusive_threshold() {
        let bound = StrictBound::from_inclusive(Cost::ONE).unwrap();

        assert!(bound.residual(1.0) > 0.0);
        assert!(!bound.admits(1.0 + bound.residual(1.0)));
    }

    #[test]
    fn rejects_a_threshold_without_a_strict_upper_bound() {
        assert_eq!(
            StrictBound::from_inclusive(Cost::MAX),
            Err(Error::InvalidThreshold(f32::MAX))
        );
    }
}
