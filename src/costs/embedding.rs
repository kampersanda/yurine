//! Static token embeddings for edit-cost policies.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroUsize;

use super::{Cost, EditCosts};
use crate::errors::{Error, Result};

/// Stores fixed-dimensional, L2-normalized embeddings by token.
///
/// Embeddings are validated and normalized when inserted, so values returned
/// by [`EmbeddingStore::get`] are always finite, non-zero unit vectors with the
/// store's configured dimension.
#[derive(Debug, Clone)]
pub struct EmbeddingStore<T> {
    dimension: NonZeroUsize,
    embeddings: HashMap<T, Box<[f32]>>,
}

impl<T> EmbeddingStore<T>
where
    T: Eq + Hash,
{
    /// Creates an empty store for embeddings of `dimension` elements.
    pub fn new(dimension: NonZeroUsize) -> Self {
        Self {
            dimension,
            embeddings: HashMap::new(),
        }
    }

    /// Inserts and L2-normalizes an embedding for `token`.
    ///
    /// If the token was already present, its previous normalized embedding is
    /// returned.
    ///
    /// # Errors
    ///
    /// Returns an error if the supplied vector has the wrong dimension,
    /// contains a non-finite value, or has zero L2 norm. The store is unchanged
    /// when validation fails.
    pub fn insert(
        &mut self,
        token: T,
        embedding: impl Into<Vec<f32>>,
    ) -> Result<Option<Box<[f32]>>> {
        let mut embedding = embedding.into();
        if embedding.len() != self.dimension.get() {
            return Err(Error::InvalidEmbeddingDimension {
                expected: self.dimension.get(),
                actual: embedding.len(),
            });
        }

        for (index, value) in embedding.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(Error::InvalidEmbeddingValue { index, value });
            }
        }

        let squared_norm: f64 = embedding
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum();
        if squared_norm == 0.0 {
            return Err(Error::ZeroNormEmbedding);
        }
        let norm = squared_norm.sqrt();
        for value in &mut embedding {
            *value = (f64::from(*value) / norm) as f32;
        }

        Ok(self.embeddings.insert(token, embedding.into_boxed_slice()))
    }

    /// Returns the normalized embedding registered for `token`.
    pub fn get<Q>(&self, token: &Q) -> Option<&[f32]>
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.embeddings.get(token).map(Box::as_ref)
    }

    /// Returns the required number of elements in every embedding.
    pub const fn dimension(&self) -> NonZeroUsize {
        self.dimension
    }

    /// Returns the number of registered tokens.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Returns whether the store contains no embeddings.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

/// Edit costs derived from cosine distances between static token embeddings.
///
/// Substitution between equal tokens always costs zero. For different tokens
/// with embeddings, the cost is `clamp(1 - cosine, 0, 1)`. If either embedding
/// is absent, the configured missing-embedding cost is used instead. Deletion
/// and insertion use configurable constant costs.
#[derive(Debug, Clone)]
pub struct CosineEmbeddingCosts<T> {
    embeddings: EmbeddingStore<T>,
    deletion: Cost,
    insertion: Cost,
    missing_substitution: Cost,
}

impl<T> CosineEmbeddingCosts<T> {
    /// Creates cosine embedding costs with unit operation costs.
    pub fn new(embeddings: EmbeddingStore<T>) -> Self {
        Self {
            embeddings,
            deletion: Cost::ONE,
            insertion: Cost::ONE,
            missing_substitution: Cost::ONE,
        }
    }

    /// Uses `cost` for deletion.
    pub fn with_deletion_cost(mut self, cost: Cost) -> Self {
        self.deletion = cost;
        self
    }

    /// Uses `cost` for insertion.
    pub fn with_insertion_cost(mut self, cost: Cost) -> Self {
        self.insertion = cost;
        self
    }

    /// Uses `cost` when either token lacks an embedding.
    pub fn with_missing_substitution_cost(mut self, cost: Cost) -> Self {
        self.missing_substitution = cost;
        self
    }

    /// Returns the static embeddings used by this cost policy.
    pub const fn embeddings(&self) -> &EmbeddingStore<T> {
        &self.embeddings
    }
}

impl<T> EditCosts<T> for CosineEmbeddingCosts<T>
where
    T: Eq + Hash,
{
    fn substitution(&self, from: &T, to: &T) -> Cost {
        if from == to {
            return Cost::ZERO;
        }

        let (Some(from), Some(to)) = (self.embeddings.get(from), self.embeddings.get(to)) else {
            return self.missing_substitution;
        };
        let similarity: f64 = from
            .iter()
            .zip(to)
            .map(|(from, to)| f64::from(*from) * f64::from(*to))
            .sum();
        let distance = (1.0 - similarity).clamp(0.0, 1.0) as f32;
        Cost::new_const(distance)
    }

    fn deletion(&self, _token: &T) -> Cost {
        self.deletion
    }

    fn insertion(&self, _token: &T) -> Cost {
        self.insertion
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{CosineEmbeddingCosts, EmbeddingStore};
    use crate::costs::{Cost, EditCosts};
    use crate::errors::Error;

    fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    fn assert_slice_approx_eq(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    fn assert_cost_approx_eq(actual: Cost, expected: f32) {
        assert!(
            (actual.get() - expected).abs() < 1e-6,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn stores_normalized_embeddings_and_reports_metadata() {
        let mut store = EmbeddingStore::new(nonzero(2));
        assert_eq!(store.dimension(), nonzero(2));
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());

        assert!(store.insert('a', vec![3.0, 4.0]).unwrap().is_none());

        assert_slice_approx_eq(store.get(&'a').unwrap(), &[0.6, 0.8]);
        assert_eq!(store.get(&'b'), None);
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn replacing_a_token_returns_its_previous_normalized_embedding() {
        let mut store = EmbeddingStore::new(nonzero(2));
        store.insert("東京".to_owned(), vec![3.0, 4.0]).unwrap();

        let previous = store
            .insert("東京".to_owned(), vec![0.0, 2.0])
            .unwrap()
            .unwrap();

        assert_slice_approx_eq(&previous, &[0.6, 0.8]);
        assert_slice_approx_eq(store.get("東京").unwrap(), &[0.0, 1.0]);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn rejects_wrong_dimension_without_changing_the_store() {
        let mut store = EmbeddingStore::new(nonzero(2));
        store.insert('a', vec![1.0, 0.0]).unwrap();

        assert_eq!(
            store.insert('a', vec![1.0]),
            Err(Error::InvalidEmbeddingDimension {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(store.get(&'a'), Some([1.0, 0.0].as_slice()));
    }

    #[test]
    fn rejects_each_non_finite_value() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut store = EmbeddingStore::new(nonzero(2));
            assert!(matches!(
                store.insert('a', vec![1.0, value]),
                Err(Error::InvalidEmbeddingValue { index: 1, value: actual })
                    if actual.to_bits() == value.to_bits()
            ));
            assert!(store.is_empty());
        }
    }

    #[test]
    fn rejects_zero_norm_embedding() {
        let mut store = EmbeddingStore::new(nonzero(2));

        assert_eq!(
            store.insert('a', vec![0.0, -0.0]),
            Err(Error::ZeroNormEmbedding)
        );
        assert!(store.is_empty());
    }

    #[test]
    fn normalizes_large_finite_values_without_overflow() {
        let mut store = EmbeddingStore::new(nonzero(2));
        store.insert('a', vec![f32::MAX, f32::MAX]).unwrap();

        let expected = 1.0 / 2.0_f32.sqrt();
        assert_slice_approx_eq(store.get(&'a').unwrap(), &[expected, expected]);
    }

    #[test]
    fn cosine_costs_use_unit_defaults() {
        let mut store = EmbeddingStore::new(nonzero(2));
        store.insert('a', vec![1.0, 0.0]).unwrap();
        store.insert('b', vec![0.0, 1.0]).unwrap();
        let costs = CosineEmbeddingCosts::new(store);

        assert_eq!(costs.substitution(&'a', &'b'), Cost::ONE);
        assert_eq!(costs.substitution(&'a', &'x'), Cost::ONE);
        assert_eq!(costs.deletion(&'a'), Cost::ONE);
        assert_eq!(costs.insertion(&'a'), Cost::ONE);
    }

    #[test]
    fn equal_tokens_cost_zero_even_without_an_embedding() {
        let costs = CosineEmbeddingCosts::new(EmbeddingStore::<char>::new(nonzero(2)))
            .with_missing_substitution_cost(Cost::new_const(0.75));

        assert_eq!(costs.substitution(&'x', &'x'), Cost::ZERO);
    }

    #[test]
    fn cosine_distance_controls_substitution_cost() {
        let mut store = EmbeddingStore::new(nonzero(2));
        store.insert('a', vec![1.0, 0.0]).unwrap();
        store.insert('p', vec![2.0, 0.0]).unwrap();
        store.insert('s', vec![0.6, 0.8]).unwrap();
        store.insert('o', vec![0.0, 1.0]).unwrap();
        store.insert('n', vec![-1.0, 0.0]).unwrap();
        let costs = CosineEmbeddingCosts::new(store);

        assert_cost_approx_eq(costs.substitution(&'a', &'p'), 0.0);
        assert_cost_approx_eq(costs.substitution(&'a', &'s'), 0.4);
        assert_cost_approx_eq(costs.substitution(&'a', &'o'), 1.0);
        assert_cost_approx_eq(costs.substitution(&'a', &'n'), 1.0);
    }

    #[test]
    fn missing_embedding_cost_applies_if_either_embedding_is_absent() {
        let mut store = EmbeddingStore::new(nonzero(2));
        store.insert('a', vec![1.0, 0.0]).unwrap();
        let costs =
            CosineEmbeddingCosts::new(store).with_missing_substitution_cost(Cost::new_const(0.25));

        assert_eq!(costs.substitution(&'a', &'x'), Cost::new_const(0.25));
        assert_eq!(costs.substitution(&'x', &'y'), Cost::new_const(0.25));
    }

    #[test]
    fn configures_deletion_and_insertion_independently() {
        let costs = CosineEmbeddingCosts::new(EmbeddingStore::<char>::new(nonzero(2)))
            .with_deletion_cost(Cost::new_const(0.25))
            .with_insertion_cost(Cost::new_const(0.75));

        assert_eq!(costs.deletion(&'a'), Cost::new_const(0.25));
        assert_eq!(costs.insertion(&'a'), Cost::new_const(0.75));
    }

    #[test]
    fn exposes_its_embedding_store() {
        let mut store = EmbeddingStore::new(nonzero(2));
        store.insert('a', vec![1.0, 0.0]).unwrap();
        let costs = CosineEmbeddingCosts::new(store);

        assert_eq!(costs.embeddings().dimension(), nonzero(2));
        assert_eq!(costs.embeddings().len(), 1);
    }
}
