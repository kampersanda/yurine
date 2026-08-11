use std::io::{self, Write};

pub const DEFAULT_QUERY: &str = "t0000 t0001 t0002 t0003";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorpusConfig {
    pub strings: usize,
    pub tokens_per_string: usize,
    pub vocabulary: usize,
    pub hot_vocabulary: usize,
    pub seed: u64,
}

impl Default for CorpusConfig {
    fn default() -> Self {
        Self {
            strings: 20_000,
            tokens_per_string: 32,
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
            || self.vocabulary > 10_000
            || self.hot_vocabulary == 0
            || self.hot_vocabulary > self.vocabulary
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "vocabulary must be 4..=10000 and hot-vocabulary must be within it",
            ));
        }
        Ok(())
    }
}

pub fn write_corpus(mut output: impl Write, config: CorpusConfig) -> io::Result<()> {
    config.validate()?;

    let mut state = config.seed;
    for string_index in 0..config.strings {
        for token_index in 0..config.tokens_per_string {
            if token_index != 0 {
                output.write_all(b" ")?;
            }
            let token = if string_index % 128 == 0 && token_index < 4 {
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
    use super::{CorpusConfig, write_corpus};

    #[test]
    fn synthetic_corpus_is_reproducible() {
        let config = CorpusConfig {
            strings: 4,
            tokens_per_string: 6,
            vocabulary: 16,
            hot_vocabulary: 4,
            seed: 7,
        };
        let mut first = Vec::new();
        let mut second = Vec::new();

        write_corpus(&mut first, config).unwrap();
        write_corpus(&mut second, config).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.iter().filter(|byte| **byte == b'\n').count(), 4);
        assert!(first.starts_with(b"t0000 t0001 t0002 t0003 "));
    }

    #[test]
    fn rejects_zero_seed() {
        let result = write_corpus(
            Vec::new(),
            CorpusConfig {
                seed: 0,
                ..CorpusConfig::default()
            },
        );

        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn rejects_vocabulary_smaller_than_fixed_query() {
        let result = write_corpus(
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
