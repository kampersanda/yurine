//! Static token embeddings for edit-cost policies.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::OnceLock;

use super::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::storage::Storage;

#[cfg(feature = "persist")]
mod costs_persistence;
#[cfg(feature = "persist")]
mod persistence;

/// Builds a fixed-dimensional store of token embeddings.
///
/// Embeddings are stored consecutively in insertion order. They are validated
/// and normalized when inserted. Call [`EmbeddingStoreBuilder::build`] to make
/// the store immutable and ready for searching or persistence.
///
/// ```
/// use std::num::NonZeroUsize;
/// use yurine::costs::EmbeddingStoreBuilder;
///
/// let mut builder = EmbeddingStoreBuilder::new(NonZeroUsize::new(2).unwrap());
/// builder.insert("cat", [3.0, 4.0])?;
/// let store = builder.build();
///
/// assert_eq!(store.get("cat"), Some([0.6, 0.8].as_slice()));
/// # Ok::<(), yurine::errors::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddingStoreBuilder<T> {
    dimension: NonZeroUsize,
    embedding_indices: HashMap<T, u32>,
    embeddings: Vec<f32>,
}

impl<T> EmbeddingStoreBuilder<T>
where
    T: Eq + Hash,
{
    /// Creates an empty builder for embeddings of `dimension` elements.
    pub fn new(dimension: NonZeroUsize) -> Self {
        Self {
            dimension,
            embedding_indices: HashMap::new(),
            embeddings: Vec::new(),
        }
    }

    /// Inserts and L2-normalizes an embedding for `token`.
    ///
    /// If the token was already present, its previous normalized embedding is
    /// returned.
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot represent another embedding index,
    /// or if the supplied vector has the wrong dimension, contains a non-finite
    /// value, or has zero L2 norm. The store is unchanged when validation
    /// fails.
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

        let dimension = self.dimension.get();
        match self.embedding_indices.entry(token) {
            Entry::Occupied(entry) => {
                let start = *entry.get() as usize * dimension;
                let stored = &mut self.embeddings[start..start + dimension];
                let previous = stored.to_vec().into_boxed_slice();
                stored.copy_from_slice(&embedding);
                Ok(Some(previous))
            }
            Entry::Vacant(entry) => {
                let index = u32::try_from(self.embeddings.len() / dimension)
                    .map_err(|_| Error::EmbeddingIndexOverflow)?;
                entry.insert(index);
                self.embeddings.extend(embedding);
                Ok(None)
            }
        }
    }

    /// Returns the normalized embedding registered for `token`.
    pub fn get<Q>(&self, token: &Q) -> Option<&[f32]>
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let index = *self.embedding_indices.get(token)? as usize;
        let start = index * self.dimension.get();
        Some(&self.embeddings[start..start + self.dimension.get()])
    }

    /// Returns the required number of elements in every embedding.
    pub const fn dimension(&self) -> NonZeroUsize {
        self.dimension
    }

    /// Returns the number of registered tokens.
    pub fn len(&self) -> usize {
        self.embedding_indices.len()
    }

    /// Returns whether the store contains no embeddings.
    pub fn is_empty(&self) -> bool {
        self.embedding_indices.is_empty()
    }

    /// Finishes this builder as an immutable embedding store.
    pub fn build(self) -> EmbeddingStore<T> {
        EmbeddingStore {
            dimension: self.dimension,
            embedding_indices: self.embedding_indices,
            embeddings: Storage::Owned(self.embeddings.into_boxed_slice()),
            validated_rows: None,
        }
    }
}

/// An immutable store of fixed-dimensional, L2-normalized token embeddings.
///
/// In-memory builders own their vector matrix. Stores opened from a persisted
/// snapshot keep the matrix memory-mapped and rebuild only the token index on
/// the heap.
#[derive(Debug, Clone)]
pub struct EmbeddingStore<T> {
    dimension: NonZeroUsize,
    embedding_indices: HashMap<T, u32>,
    embeddings: Storage<f32>,
    validated_rows: Option<Box<[OnceLock<bool>]>>,
}

impl<T> EmbeddingStore<T>
where
    T: Eq + Hash,
{
    /// Returns the normalized embedding registered for `token`.
    ///
    /// A semantically invalid row from a corrupted mapped file is treated as
    /// absent. [`EmbeddingStore::verify`] reports the precise validation error.
    pub fn get<Q>(&self, token: &Q) -> Option<&[f32]>
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let index = *self.embedding_indices.get(token)? as usize;
        let start = index.checked_mul(self.dimension.get())?;
        let end = start.checked_add(self.dimension.get())?;
        let row = self.embeddings.get(start..end)?;
        if let Some(validated_rows) = &self.validated_rows {
            let valid = validated_rows
                .get(index)?
                .get_or_init(|| validate_embedding(row).is_ok());
            if !valid {
                return None;
            }
        }
        Some(row)
    }

    /// Returns the required number of elements in every embedding.
    pub const fn dimension(&self) -> NonZeroUsize {
        self.dimension
    }

    /// Returns the number of registered tokens.
    pub fn len(&self) -> usize {
        self.embedding_indices.len()
    }

    /// Returns whether the store contains no embeddings.
    pub fn is_empty(&self) -> bool {
        self.embedding_indices.is_empty()
    }

    /// Fully validates every embedding row.
    pub fn verify(&self) -> Result<()> {
        let expected_len = self
            .len()
            .checked_mul(self.dimension.get())
            .ok_or(Error::InvalidFile("embedding matrix length overflows"))?;
        if self.embeddings.len() != expected_len {
            return Err(Error::InvalidFile(
                "embedding matrix length does not match its shape",
            ));
        }
        for (index, row) in self
            .embeddings
            .chunks_exact(self.dimension.get())
            .enumerate()
        {
            let result = validate_embedding(row);
            if let Some(validated_rows) = &self.validated_rows {
                let _ = validated_rows[index].set(result.is_ok());
            }
            result?;
        }
        Ok(())
    }
}

fn validate_embedding(embedding: &[f32]) -> Result<()> {
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
    if (squared_norm - 1.0).abs() > 1e-5 {
        return Err(Error::InvalidFile("embedding is not L2-normalized"));
    }
    Ok(())
}

/// Returns the dot product of two L2-normalized embeddings.
///
/// This is the innermost loop of a search: filtering evaluates it once per
/// alphabet symbol per query position. Two details keep it vectorized. Splitting
/// the sum across independent accumulators breaks the serial dependency chain
/// that forbids reassociating floating-point addition, and iterating with
/// `chunks_exact` gives each lane a statically known length, so no bounds check
/// survives to block the vectorizer. Indexing the two slices directly is enough
/// to lose both.
///
/// The lanes accumulate in `f32`, which is what makes them worth having: `f64`
/// accumulators are half as wide and cost about a quarter of the speedup. The
/// price is that rounding compounds once per term, so the error grows with the
/// dimension rather than staying under any fixed constant. Against a sequential
/// `f64` sum it measures around `2e-7` from 64 through 16384 dimensions, then
/// `7e-7` at 65536 and `2e-6` at 262144. The classical bound for a dot product
/// of unit vectors is `dimension * EPSILON`, which
/// `substitution_cost_matches_a_sequential_sum_at_any_dimension` asserts.
///
/// That error is well clear of the decisions it feeds. Filtering keeps a symbol
/// when its cost is at most eta, and cosine costs are continuous, so they land
/// near eta only by coincidence: over 664,000 comparisons per eta on clustered
/// embeddings, the closest any cost came to a nonzero eta was `1.5e-5`, and no
/// symbol changed neighborhoods.
///
/// Eta of zero is the exception, because costs cluster exactly on it rather than
/// spreading around it, and there the same sweep moved 41 of 664,000 symbols.
/// Only distinct tokens with near-identical embeddings are affected: equal
/// tokens never reach this function, so a threshold of zero still finds every
/// exact match. Rows are stored as `f32`, so their norms are not exactly one and
/// costs for such pairs are already approximate whatever this sum does.
///
/// Sixteen lanes rather than eight measured faster for both accumulator widths.
fn dot_product(from: &[f32], to: &[f32]) -> f32 {
    const LANES: usize = 16;

    let mut accumulators = [0.0; LANES];
    let mut from_chunks = from.chunks_exact(LANES);
    let mut to_chunks = to.chunks_exact(LANES);
    for (from, to) in from_chunks.by_ref().zip(to_chunks.by_ref()) {
        for lane in 0..LANES {
            accumulators[lane] += from[lane] * to[lane];
        }
    }

    let mut product: f32 = accumulators.iter().sum();
    for (from, to) in from_chunks.remainder().iter().zip(to_chunks.remainder()) {
        product += from * to;
    }
    product
}

/// Edit costs derived from cosine distances between static token embeddings.
///
/// Substitution between equal tokens always costs zero. For different tokens
/// with embeddings, the cost is `clamp(1 - cosine, 0, 1)`. If either embedding
/// is absent, the configured missing-embedding cost is used instead. Deletion
/// and insertion use configurable constant costs.
///
/// ```
/// use std::num::NonZeroUsize;
/// use yurine::costs::{Cost, CosineEmbeddingCosts, EditCosts, EmbeddingStoreBuilder};
///
/// let mut builder = EmbeddingStoreBuilder::new(NonZeroUsize::new(2).unwrap());
/// builder.insert("literature", [1.0, 0.0])?;
/// builder.insert("books", [0.8, 0.6])?;
/// let costs = CosineEmbeddingCosts::new(builder.build());
///
/// let distance = costs.substitution(&"literature", &"books").get();
/// assert!((distance - 0.2).abs() < 1e-6);
/// assert_eq!(costs.substitution(&"literature", &"unknown"), Cost::ONE);
/// # Ok::<(), yurine::errors::Error>(())
/// ```
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
        let distance = (1.0 - dot_product(from, to)).clamp(0.0, 1.0);
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

    use approx::assert_abs_diff_eq;

    use super::{CosineEmbeddingCosts, EmbeddingStoreBuilder};
    use crate::costs::{Cost, EditCosts};
    use crate::errors::Error;

    fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    #[test]
    fn stores_normalized_embeddings_and_reports_metadata() {
        let mut store = EmbeddingStoreBuilder::new(nonzero(2));
        assert_eq!(store.dimension(), nonzero(2));
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());

        assert!(store.insert('a', vec![3.0, 4.0]).unwrap().is_none());

        assert_abs_diff_eq!(store.get(&'a').unwrap(), [0.6, 0.8].as_slice());
        assert_eq!(store.get(&'b'), None);
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn stores_embeddings_in_a_flat_array() {
        let mut store = EmbeddingStoreBuilder::new(nonzero(2));
        store.insert('a', vec![1.0, 0.0]).unwrap();
        store.insert('b', vec![0.0, 1.0]).unwrap();

        assert_eq!(store.embeddings, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(store.embedding_indices[&'a'], 0);
        assert_eq!(store.embedding_indices[&'b'], 1);

        store.insert('a', vec![0.0, 1.0]).unwrap();

        assert_eq!(store.embeddings, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(store.embedding_indices[&'a'], 0);
    }

    #[test]
    fn replacing_a_token_returns_its_previous_normalized_embedding() {
        let mut store = EmbeddingStoreBuilder::new(nonzero(2));
        store.insert("東京".to_owned(), vec![3.0, 4.0]).unwrap();

        let previous = store
            .insert("東京".to_owned(), vec![0.0, 2.0])
            .unwrap()
            .unwrap();

        assert_abs_diff_eq!(&*previous, [0.6, 0.8].as_slice());
        assert_abs_diff_eq!(store.get("東京").unwrap(), [0.0, 1.0].as_slice());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn rejects_wrong_dimension_without_changing_the_store() {
        let mut store = EmbeddingStoreBuilder::new(nonzero(2));
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
            let mut store = EmbeddingStoreBuilder::new(nonzero(2));
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
        let mut store = EmbeddingStoreBuilder::new(nonzero(2));

        assert_eq!(
            store.insert('a', vec![0.0, -0.0]),
            Err(Error::ZeroNormEmbedding)
        );
        assert!(store.is_empty());
    }

    #[test]
    fn normalizes_large_finite_values_without_overflow() {
        let mut store = EmbeddingStoreBuilder::new(nonzero(2));
        store.insert('a', vec![f32::MAX, f32::MAX]).unwrap();

        let expected = 1.0 / 2.0_f32.sqrt();
        assert_abs_diff_eq!(store.get(&'a').unwrap(), [expected, expected].as_slice());
    }

    #[test]
    fn cosine_costs_use_unit_defaults() {
        let mut store = EmbeddingStoreBuilder::new(nonzero(2));
        store.insert('a', vec![1.0, 0.0]).unwrap();
        store.insert('b', vec![0.0, 1.0]).unwrap();
        let costs = CosineEmbeddingCosts::new(store.build());

        assert_eq!(costs.substitution(&'a', &'b'), Cost::ONE);
        assert_eq!(costs.substitution(&'a', &'x'), Cost::ONE);
        assert_eq!(costs.deletion(&'a'), Cost::ONE);
        assert_eq!(costs.insertion(&'a'), Cost::ONE);
    }

    #[test]
    fn equal_tokens_cost_zero_even_without_an_embedding() {
        let costs =
            CosineEmbeddingCosts::new(EmbeddingStoreBuilder::<char>::new(nonzero(2)).build())
                .with_missing_substitution_cost(Cost::new_const(0.75));

        assert_eq!(costs.substitution(&'x', &'x'), Cost::ZERO);
    }

    #[test]
    fn cosine_distance_controls_substitution_cost() {
        let mut store = EmbeddingStoreBuilder::new(nonzero(2));
        store.insert('a', vec![1.0, 0.0]).unwrap();
        store.insert('p', vec![2.0, 0.0]).unwrap();
        store.insert('s', vec![0.6, 0.8]).unwrap();
        store.insert('o', vec![0.0, 1.0]).unwrap();
        store.insert('n', vec![-1.0, 0.0]).unwrap();
        let costs = CosineEmbeddingCosts::new(store.build());

        assert_abs_diff_eq!(costs.substitution(&'a', &'p').get(), 0.0);
        assert_abs_diff_eq!(costs.substitution(&'a', &'s').get(), 0.4);
        assert_abs_diff_eq!(costs.substitution(&'a', &'o').get(), 1.0);
        assert_abs_diff_eq!(costs.substitution(&'a', &'n').get(), 1.0);
    }

    /// The dot product consumes a fixed number of elements at a time, so a
    /// dimension that is not a whole multiple of that must still account for
    /// its trailing elements.
    #[test]
    fn substitution_cost_includes_elements_outside_the_whole_chunks() {
        let dimension = 300;
        let mut first = vec![0.0; dimension];
        first[0] = 1.0;
        first[298] = 1.0;
        let mut second = vec![0.0; dimension];
        second[298] = 1.0;
        let mut store = EmbeddingStoreBuilder::new(nonzero(dimension));
        store.insert('a', first).unwrap();
        store.insert('b', second).unwrap();
        let costs = CosineEmbeddingCosts::new(store.build());

        // Dropping the trailing elements would leave the two embeddings
        // orthogonal, costing a full unit instead.
        assert_abs_diff_eq!(
            costs.substitution(&'a', &'b').get(),
            1.0 - 1.0 / 2.0_f32.sqrt(),
            epsilon = 1e-6
        );
    }

    /// Deterministic pseudo-random values in `-0.5..0.5`.
    fn pseudo_random_embedding(state: &mut u64, dimension: usize) -> Vec<f32> {
        (0..dimension)
            .map(|_| {
                *state ^= *state << 13;
                *state ^= *state >> 7;
                *state ^= *state << 17;
                *state as u32 as f32 / u32::MAX as f32 - 0.5
            })
            .collect()
    }

    /// Splitting the sum across lanes reorders the additions and accumulates in
    /// `f32`, so the cost drifts from a sequential `f64` sum by an amount that
    /// grows with the dimension.
    ///
    /// The tolerance is the classical `dimension * EPSILON` bound for a
    /// floating-point dot product of unit vectors, rather than a constant read
    /// off one measurement. It is loose enough to admit honest rounding at every
    /// dimension and tight enough to catch a sum that drops terms: leaving out
    /// the trailing elements, or all but one lane, exceeds it by orders of
    /// magnitude.
    #[test]
    fn substitution_cost_matches_a_sequential_sum_at_any_dimension() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for dimension in [64, 300, 1024, 4099] {
            // Two independent vectors of this many dimensions are
            // near-orthogonal, which clamps the cost to one and hides whatever
            // the sum reports. Perturbing the first embedding keeps the cost
            // strictly inside the clamp, so every element contributes to it.
            let first = pseudo_random_embedding(&mut state, dimension);
            let perturbation = pseudo_random_embedding(&mut state, dimension);
            let second: Vec<f32> = first
                .iter()
                .zip(&perturbation)
                .map(|(value, noise)| value + noise * 0.01)
                .collect();
            let mut store = EmbeddingStoreBuilder::new(nonzero(dimension));
            store.insert('a', first).unwrap();
            store.insert('b', second).unwrap();
            let store = store.build();
            let similarity: f64 = store
                .get(&'a')
                .unwrap()
                .iter()
                .zip(store.get(&'b').unwrap())
                .map(|(first, second)| f64::from(*first) * f64::from(*second))
                .sum();
            let costs = CosineEmbeddingCosts::new(store);

            let distance = costs.substitution(&'a', &'b').get();
            assert!((0.0..1.0).contains(&distance), "clamped to {distance}");
            assert_abs_diff_eq!(
                distance,
                (1.0 - similarity).clamp(0.0, 1.0) as f32,
                epsilon = dimension as f32 * f32::EPSILON
            );
        }
    }

    #[test]
    fn missing_embedding_cost_applies_if_either_embedding_is_absent() {
        let mut store = EmbeddingStoreBuilder::new(nonzero(2));
        store.insert('a', vec![1.0, 0.0]).unwrap();
        let costs = CosineEmbeddingCosts::new(store.build())
            .with_missing_substitution_cost(Cost::new_const(0.25));

        assert_eq!(costs.substitution(&'a', &'x'), Cost::new_const(0.25));
        assert_eq!(costs.substitution(&'x', &'y'), Cost::new_const(0.25));
    }

    #[test]
    fn configures_deletion_and_insertion_independently() {
        let costs =
            CosineEmbeddingCosts::new(EmbeddingStoreBuilder::<char>::new(nonzero(2)).build())
                .with_deletion_cost(Cost::new_const(0.25))
                .with_insertion_cost(Cost::new_const(0.75));

        assert_eq!(costs.deletion(&'a'), Cost::new_const(0.25));
        assert_eq!(costs.insertion(&'a'), Cost::new_const(0.75));
    }

    #[test]
    fn exposes_its_embedding_store() {
        let mut store = EmbeddingStoreBuilder::new(nonzero(2));
        store.insert('a', vec![1.0, 0.0]).unwrap();
        let costs = CosineEmbeddingCosts::new(store.build());

        assert_eq!(costs.embeddings().dimension(), nonzero(2));
        assert_eq!(costs.embeddings().len(), 1);
    }
}
