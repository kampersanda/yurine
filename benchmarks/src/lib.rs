use std::io::{self, Write};
use std::num::NonZeroUsize;

use yurine::costs::{EmbeddingStore, EmbeddingStoreBuilder};

pub const DEFAULT_QUERY_SOURCE_TEXT: &str = "t0000 t0001 t0002 t0003";

/// Largest vocabulary the generator accepts.
///
/// Embedding-based costs scan the whole vocabulary once per query position, so
/// the interesting workloads are much larger than a Levenshtein one needs.
pub const MAX_VOCABULARY: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorpusConfig {
    pub sequences: usize,
    pub tokens_per_sequence: usize,
    pub vocabulary: usize,
    pub hot_vocabulary: usize,
    pub seed: u64,
}

impl Default for CorpusConfig {
    fn default() -> Self {
        Self {
            sequences: 20_000,
            tokens_per_sequence: 32,
            vocabulary: 256,
            hot_vocabulary: 8,
            seed: 0x59d2_f15d_24b7_3a91,
        }
    }
}

impl CorpusConfig {
    pub fn validate(&self) -> io::Result<()> {
        if self.seed == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seed must be non-zero",
            ));
        }
        if self.vocabulary < 4
            || self.vocabulary > MAX_VOCABULARY
            || self.hot_vocabulary == 0
            || self.hot_vocabulary > self.vocabulary
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "vocabulary must be 4..={MAX_VOCABULARY} and hot-vocabulary must be within it"
                ),
            ));
        }
        Ok(())
    }
}

pub fn write_data_sequences(mut output: impl Write, config: CorpusConfig) -> io::Result<()> {
    config.validate()?;

    let mut state = config.seed;
    for sequence_index in 0..config.sequences {
        for token_index in 0..config.tokens_per_sequence {
            if token_index != 0 {
                output.write_all(b" ")?;
            }
            let token = if sequence_index % 128 == 0 && token_index < 4 {
                token_index
            } else {
                let random = next_random(&mut state) as usize;
                if random & 3 != 0 || config.hot_vocabulary == config.vocabulary {
                    (random >> 2) % config.hot_vocabulary
                } else {
                    config.hot_vocabulary
                        + (random >> 2) % (config.vocabulary - config.hot_vocabulary)
                }
            };
            write!(output, "t{token:04}")?;
        }
        output.write_all(b"\n")?;
    }
    Ok(())
}

/// Shape of the synthetic embedding matrix.
///
/// Tokens are spread over `clusters` clusters, and every embedding is the same
/// fixed blend of its cluster center and an independent random direction. Two
/// tokens in one cluster then have cosine similarity near `cohesion`, and two
/// tokens in different clusters are near-orthogonal. Drawing every embedding
/// independently would leave every pair near-orthogonal, so a substitution
/// neighborhood would hold nothing but the query token itself and the cost of
/// scanning neighbor lists would never appear.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbeddingConfig {
    pub dimension: NonZeroUsize,
    pub clusters: NonZeroUsize,
    /// Squared weight of the cluster center, in `0.0..1.0`.
    pub cohesion: f32,
    pub seed: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: NonZeroUsize::new(300).unwrap(),
            clusters: NonZeroUsize::new(64).unwrap(),
            cohesion: 0.85,
            seed: 0x3f81_c04a_9e13_5b27,
        }
    }
}

impl EmbeddingConfig {
    pub fn validate(&self) -> io::Result<()> {
        if self.seed == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seed must be non-zero",
            ));
        }
        if !(0.0..1.0).contains(&self.cohesion) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cohesion must be in 0.0..1.0",
            ));
        }
        Ok(())
    }
}

/// Builds a deterministic embedding store covering `tokens`.
///
/// Tokens are sorted and deduplicated first, so the same token set produces the
/// same matrix whatever order it arrives in, and the same configuration always
/// produces the same values. Cluster `index % clusters` owns the token at
/// sorted position `index`, which spreads frequent tokens across clusters
/// instead of gathering them into one.
pub fn build_embedding_store(
    tokens: &[String],
    config: EmbeddingConfig,
) -> io::Result<EmbeddingStore<String>> {
    config.validate()?;

    let dimension = config.dimension.get();
    let mut state = config.seed;
    let mut centers = Vec::with_capacity(config.clusters.get() * dimension);
    for _ in 0..config.clusters.get() {
        centers.extend(random_unit_vector(&mut state, dimension));
    }

    let center_weight = config.cohesion.sqrt();
    let noise_weight = (1.0 - config.cohesion).sqrt();
    let mut sorted: Vec<&String> = tokens.iter().collect();
    sorted.sort_unstable();
    sorted.dedup();

    let mut builder = EmbeddingStoreBuilder::new(config.dimension);
    for (index, token) in sorted.into_iter().enumerate() {
        let center_start = (index % config.clusters.get()) * dimension;
        let center = &centers[center_start..center_start + dimension];
        let noise = random_unit_vector(&mut state, dimension);
        let embedding: Vec<f32> = center
            .iter()
            .zip(&noise)
            .map(|(center, noise)| center * center_weight + noise * noise_weight)
            .collect();
        builder
            .insert(token.clone(), embedding)
            .map_err(io::Error::other)?;
    }
    Ok(builder.build())
}

/// Draws a direction with uniform elements and scales it to unit L2 norm.
fn random_unit_vector(state: &mut u64, dimension: usize) -> Vec<f32> {
    let mut values: Vec<f32> = (0..dimension)
        .map(|_| next_random(state) as u32 as f32 / u32::MAX as f32 - 0.5)
        .collect();
    let norm = values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    for value in &mut values {
        *value = (f64::from(*value) / norm) as f32;
    }
    values
}

fn next_random(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    value
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{CorpusConfig, EmbeddingConfig, build_embedding_store, write_data_sequences};

    fn tokens(count: usize) -> Vec<String> {
        (0..count).map(|index| format!("t{index:04}")).collect()
    }

    fn similarity(store: &yurine::costs::EmbeddingStore<String>, first: &str, second: &str) -> f32 {
        store
            .get(first)
            .unwrap()
            .iter()
            .zip(store.get(second).unwrap())
            .map(|(first, second)| first * second)
            .sum()
    }

    #[test]
    fn synthetic_data_sequences_are_reproducible() {
        let config = CorpusConfig {
            sequences: 4,
            tokens_per_sequence: 6,
            vocabulary: 16,
            hot_vocabulary: 4,
            seed: 7,
        };
        let mut first = Vec::new();
        let mut second = Vec::new();

        write_data_sequences(&mut first, config).unwrap();
        write_data_sequences(&mut second, config).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.iter().filter(|byte| **byte == b'\n').count(), 4);
        assert!(first.starts_with(b"t0000 t0001 t0002 t0003 "));
    }

    #[test]
    fn rejects_zero_seed() {
        let result = write_data_sequences(
            Vec::new(),
            CorpusConfig {
                seed: 0,
                ..CorpusConfig::default()
            },
        );

        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn accepts_a_vocabulary_larger_than_the_former_ten_thousand_limit() {
        let mut output = Vec::new();

        write_data_sequences(
            &mut output,
            CorpusConfig {
                sequences: 2,
                tokens_per_sequence: 4,
                vocabulary: 20_000,
                hot_vocabulary: 8,
                ..CorpusConfig::default()
            },
        )
        .unwrap();

        assert!(output.starts_with(b"t0000 t0001 t0002 t0003\n"));
    }

    #[test]
    fn synthetic_embeddings_are_reproducible() {
        let config = EmbeddingConfig {
            dimension: NonZeroUsize::new(64).unwrap(),
            clusters: NonZeroUsize::new(4).unwrap(),
            ..EmbeddingConfig::default()
        };
        let tokens = tokens(16);

        let first = build_embedding_store(&tokens, config).unwrap();
        let second =
            build_embedding_store(&tokens.iter().rev().cloned().collect::<Vec<_>>(), config)
                .unwrap();

        assert_eq!(first.len(), tokens.len());
        for token in &tokens {
            assert_eq!(first.get(token), second.get(token));
        }
    }

    /// Tokens sharing a cluster must stay close enough to populate a
    /// substitution neighborhood, and tokens from different clusters must stay
    /// far enough outside it.
    #[test]
    fn clustered_embeddings_separate_near_and_far_tokens() {
        let config = EmbeddingConfig {
            dimension: NonZeroUsize::new(256).unwrap(),
            clusters: NonZeroUsize::new(2).unwrap(),
            cohesion: 0.85,
            ..EmbeddingConfig::default()
        };

        let store = build_embedding_store(&tokens(4), config).unwrap();

        assert!(similarity(&store, "t0000", "t0002") > 0.75);
        assert!(similarity(&store, "t0001", "t0003") > 0.75);
        assert!(similarity(&store, "t0000", "t0001").abs() < 0.25);
    }

    #[test]
    fn rejects_cohesion_outside_the_unit_interval() {
        for cohesion in [-0.1, 1.0, f32::NAN] {
            let result = build_embedding_store(
                &tokens(2),
                EmbeddingConfig {
                    cohesion,
                    ..EmbeddingConfig::default()
                },
            );

            assert_eq!(
                result.unwrap_err().kind(),
                std::io::ErrorKind::InvalidInput,
                "cohesion {cohesion}"
            );
        }
    }

    #[test]
    fn rejects_vocabulary_smaller_than_fixed_query_sequence() {
        let result = write_data_sequences(
            Vec::new(),
            CorpusConfig {
                vocabulary: 3,
                hot_vocabulary: 3,
                ..CorpusConfig::default()
            },
        );

        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
    }
}
