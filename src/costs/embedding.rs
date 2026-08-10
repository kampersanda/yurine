//! Static token embeddings for edit-cost policies.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

use crate::errors::{Error, Result};

/// Stores fixed-dimensional, L2-normalized embeddings by token.
///
/// Embeddings are validated and normalized when inserted, so values returned
/// by [`EmbeddingStore::get`] are always finite, non-zero unit vectors with the
/// store's configured dimension.
#[derive(Debug, Clone)]
pub struct EmbeddingStore<T> {
    dimension: usize,
    embeddings: HashMap<T, Box<[f32]>>,
}

impl<T> EmbeddingStore<T>
where
    T: Eq + Hash,
{
    /// Creates an empty store for embeddings of `dimension` elements.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ZeroEmbeddingDimension`] if `dimension` is zero.
    pub fn new(dimension: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(Error::ZeroEmbeddingDimension);
        }
        Ok(Self {
            dimension,
            embeddings: HashMap::new(),
        })
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
        if embedding.len() != self.dimension {
            return Err(Error::InvalidEmbeddingDimension {
                expected: self.dimension,
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
    pub const fn dimension(&self) -> usize {
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

#[cfg(test)]
mod tests {
    use super::EmbeddingStore;
    use crate::errors::Error;

    fn assert_slice_approx_eq(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn rejects_zero_dimension() {
        assert!(matches!(
            EmbeddingStore::<char>::new(0),
            Err(Error::ZeroEmbeddingDimension)
        ));
    }

    #[test]
    fn stores_normalized_embeddings_and_reports_metadata() {
        let mut store = EmbeddingStore::new(2).unwrap();
        assert_eq!(store.dimension(), 2);
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
        let mut store = EmbeddingStore::new(2).unwrap();
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
        let mut store = EmbeddingStore::new(2).unwrap();
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
            let mut store = EmbeddingStore::new(2).unwrap();
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
        let mut store = EmbeddingStore::new(2).unwrap();

        assert_eq!(
            store.insert('a', vec![0.0, -0.0]),
            Err(Error::ZeroNormEmbedding)
        );
        assert!(store.is_empty());
    }

    #[test]
    fn normalizes_large_finite_values_without_overflow() {
        let mut store = EmbeddingStore::new(2).unwrap();
        store.insert('a', vec![f32::MAX, f32::MAX]).unwrap();

        let expected = 1.0 / 2.0_f32.sqrt();
        assert_slice_approx_eq(store.get(&'a').unwrap(), &[expected, expected]);
    }
}
