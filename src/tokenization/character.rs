//! Character tokenization.

use super::Tokenizer;

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

    fn tokenize(&self, input: &str) -> Vec<Self::Token> {
        input.chars().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::CharacterTokenizer;
    use crate::tokenization::Tokenizer;

    #[test]
    fn splits_unicode_scalar_values() {
        let tokenizer = CharacterTokenizer::new();

        assert_eq!(tokenizer.tokenize("aあ🦀"), ['a', 'あ', '🦀']);
    }

    #[test]
    fn handles_empty_input() {
        let tokenizer = CharacterTokenizer::new();

        assert!(tokenizer.tokenize("").is_empty());
    }
}
