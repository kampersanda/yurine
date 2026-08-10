//! Strategies for converting input strings into token sequences.

pub mod character;

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
