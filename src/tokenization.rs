//! Strategies for converting input strings into token sequences.

/// Converts an input string into a sequence of tokens.
///
/// Implementations define the token boundary. This allows callers to replace
/// character tokenization with N-gram or word tokenization without changing
/// the vocabulary or indexing code.
pub trait Tokenizer {
    /// The token produced by this tokenizer.
    type Token;

    /// Splits `input` into tokens.
    fn tokenize(&self, input: &str) -> Vec<Self::Token>;
}

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
    use super::{CharacterTokenizer, Tokenizer};

    #[test]
    fn character_tokenizer_splits_unicode_scalar_values() {
        let tokenizer = CharacterTokenizer::new();

        assert_eq!(tokenizer.tokenize("aあ🦀"), ['a', 'あ', '🦀']);
    }

    #[test]
    fn character_tokenizer_handles_empty_input() {
        let tokenizer = CharacterTokenizer::new();

        assert!(tokenizer.tokenize("").is_empty());
    }
}
