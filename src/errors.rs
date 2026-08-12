//! Error types.

use crate::types::{Position, StringId, Symbol};

/// An error type for the library.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The corpus contains too many strings for a `u32` string identifier.
    #[error("string identifier exceeds u32")]
    StringIdOverflow,

    /// A string or range endpoint is too large for a `u32` position.
    #[error("position exceeds u32")]
    PositionOverflow,

    /// A UTF-8 byte offset in a single string is too large for `u32`.
    #[error("byte offset exceeds u32")]
    ByteOffsetOverflow,

    /// A token byte range is not a valid UTF-8 slice of its original string.
    #[error("token byte range {start}..{end} is invalid for a string of {string_len} bytes")]
    InvalidByteRange {
        /// The inclusive byte offset where the range starts.
        start: usize,
        /// The exclusive byte offset where the range ends.
        end: usize,
        /// The UTF-8 byte length of the original string.
        string_len: usize,
    },

    /// A vocabulary contains too many tokens for a `u32` symbol.
    #[error("symbol exceeds u32")]
    SymbolOverflow,

    /// A corpus contains a symbol that is not present in its vocabulary.
    #[error("corpus symbol {0} is not present in the vocabulary")]
    UnknownCorpusSymbol(Symbol),

    /// A fixed-width identifier or position cannot be represented by this platform.
    #[error("fixed-width value exceeds platform size")]
    PlatformSizeOverflow,

    /// A cost was negative, infinite, or not a number.
    #[error("cost must be finite and non-negative, not {0}")]
    InvalidCost(f32),

    /// An embedding does not have the dimension required by its store.
    #[error("embedding dimension must be {expected}, not {actual}")]
    InvalidEmbeddingDimension {
        /// The dimension configured for the store.
        expected: usize,
        /// The dimension of the supplied embedding.
        actual: usize,
    },

    /// An embedding contains a value that is infinite or not a number.
    #[error("embedding value at index {index} must be finite, not {value}")]
    InvalidEmbeddingValue {
        /// The zero-based index of the invalid value.
        index: usize,
        /// The invalid value.
        value: f32,
    },

    /// An embedding has zero L2 norm and cannot be normalized.
    #[error("embedding must have a non-zero L2 norm")]
    ZeroNormEmbedding,

    /// A search threshold was the largest representable cost.
    #[error("search threshold must be less than f32::MAX, not {0}")]
    InvalidThreshold(f32),

    /// A finite alphabet contains the same symbol more than once.
    #[error("finite alphabet contains a duplicate symbol")]
    DuplicateAlphabetSymbol,

    /// A selected query position is outside the query.
    #[error("query position {position} is out of bounds for query length {query_len}")]
    InvalidQueryPosition {
        /// The zero-based position supplied by the selector.
        position: Position,
        /// The number of symbols in the query.
        query_len: usize,
    },

    /// A candidate data position is outside its referenced string.
    #[error("data position {position} is out of bounds for data string length {data_len}")]
    InvalidDataPosition {
        /// The zero-based position supplied by the candidate.
        position: Position,
        /// The number of symbols in the referenced data string.
        data_len: usize,
    },

    /// A candidate refers to a string that is not present in the corpus.
    #[error("unknown string id: {0}")]
    UnknownString(StringId),

    /// No threshold subsequence can be constructed for the query, possibly
    /// because deletion costs are too small for the threshold.
    #[error(
        "a threshold subsequence cannot be constructed; deletion costs may be too small for the threshold"
    )]
    ThresholdSubsequenceUnavailable,
}

/// A specialized `Result` type for errors.
pub type Result<T> = std::result::Result<T, Error>;
