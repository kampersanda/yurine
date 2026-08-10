//! Strategies for converting input strings into token sequences.

use std::ops::Range;

pub mod character;

/// A token together with its location in the original UTF-8 string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokenized<T> {
    /// The token value used for indexing and matching.
    pub value: T,
    /// The zero-based, end-exclusive byte range in the original string.
    pub byte_range: Range<usize>,
}

impl<T> Tokenized<T> {
    /// Creates a token with its location in the original UTF-8 string.
    pub fn new(value: T, byte_range: Range<usize>) -> Self {
        Self { value, byte_range }
    }
}

/// Converts an input string into a sequence of tokens.
///
/// Implementations define the token boundary. This allows callers to replace
/// character tokenization with N-gram or word tokenization without changing
/// the vocabulary or indexing code.
pub trait Tokenizer {
    /// The token produced by this tokenizer.
    type Token;

    /// Splits `input` into tokens and records their original byte ranges.
    ///
    /// Ranges must use UTF-8 byte offsets into `input` and follow the returned
    /// token order. Search results span from the first matched token's start
    /// offset through the last matched token's end offset.
    fn tokenize(&self, input: &str) -> Vec<Tokenized<Self::Token>>;
}
