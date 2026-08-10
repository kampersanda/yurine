//! Character tokenization.

use super::{Tokenized, Tokenizer};

/// Splits a string into Unicode scalar values.
#[derive(Debug, Clone, Copy, Default)]
pub struct CharacterTokenizer;

impl CharacterTokenizer {
    /// Creates a character tokenizer.
    pub const fn new() -> Self {
        Self
    }
}

impl Tokenizer for CharacterTokenizer {
    type Token = char;

    fn tokenize(&self, input: &str) -> Vec<Tokenized<Self::Token>> {
        input
            .char_indices()
            .map(|(start, value)| {
                let end = start + value.len_utf8();
                Tokenized::new(value, start..end)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::CharacterTokenizer;
    use crate::tokenization::{Tokenized, Tokenizer};

    #[test]
    fn splits_unicode_scalar_values() {
        let tokenizer = CharacterTokenizer::new();

        assert_eq!(
            tokenizer.tokenize("aあ🦀"),
            [
                Tokenized::new('a', 0..1),
                Tokenized::new('あ', 1..4),
                Tokenized::new('🦀', 4..8),
            ]
        );
    }

    #[test]
    fn handles_empty_input() {
        let tokenizer = CharacterTokenizer::new();

        assert!(tokenizer.tokenize("").is_empty());
    }
}
