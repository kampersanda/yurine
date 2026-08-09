//! Error types.

use crate::types::{Position, StringId};

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

    /// A fixed-width identifier or position cannot be represented by this platform.
    #[error("fixed-width value exceeds platform size")]
    PlatformSizeOverflow,

    /// A cost was negative, infinite, or not a number.
    #[error("cost must be finite and non-negative, not {0}")]
    InvalidCost(f32),

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
