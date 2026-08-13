//! Error types.

#[cfg(feature = "persist")]
use std::path::Path;
use std::path::PathBuf;

/// An error type for the library.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An operating-system I/O operation failed.
    #[error("I/O error for {path:?} ({kind:?}): {message}")]
    Io {
        /// The file or directory involved in the failed operation.
        path: PathBuf,
        /// The portable category of the I/O failure.
        kind: std::io::ErrorKind,
        /// The operating-system error message.
        message: String,
    },

    /// A persisted file is malformed or internally inconsistent.
    #[error("invalid persisted file: {0}")]
    InvalidFile(&'static str),

    /// A persisted file uses an unsupported format version.
    #[error("unsupported format version: {0}")]
    UnsupportedFormatVersion(u32),

    /// A persisted file was written with a different byte order.
    #[error("persisted file is not little-endian")]
    EndiannessMismatch,

    /// This target cannot zero-copy little-endian persisted values.
    #[error("memory-mapped persistence requires a little-endian target")]
    UnsupportedHostEndianness,

    /// The supplied token codec does not match the persisted codec.
    #[error("token codec mismatch: expected {expected}, found {actual}")]
    CodecMismatch {
        /// The codec requested by the caller.
        expected: String,
        /// The codec recorded in the file.
        actual: String,
    },

    /// The supplied token codec version does not match the persisted version.
    #[error("token codec version mismatch: expected {expected}, found {actual}")]
    CodecVersionMismatch {
        /// The version requested by the caller.
        expected: u32,
        /// The version recorded in the file.
        actual: u32,
    },

    /// A token codec identifier exceeds the persisted format's limit.
    #[error("token codec identifier is {length} bytes, maximum is {max}")]
    CodecIdTooLong {
        /// The supplied identifier length in bytes.
        length: usize,
        /// The maximum identifier length accepted by the format.
        max: usize,
    },

    /// A token's persisted bytes are invalid for the selected codec.
    #[error("invalid token encoding: {0}")]
    InvalidTokenEncoding(String),

    /// The corpus contains too many sequences for a `u32` sequence identifier.
    #[error("sequence identifier exceeds u32")]
    SequenceIdOverflow,

    /// A string or range endpoint is too large for a `u32` position.
    #[error("position exceeds u32")]
    PositionOverflow,

    /// A vocabulary contains too many tokens for a `u32` symbol.
    #[error("symbol exceeds u32")]
    SymbolOverflow,

    /// An embedding store contains too many embeddings for a `u32` index.
    #[error("embedding index exceeds u32")]
    EmbeddingIndexOverflow,

    /// A string contains a symbol that is not present in its vocabulary.
    #[error("string symbol {0} is not present in the vocabulary")]
    UnknownStringSymbol(u32),

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
        position: usize,
        /// The number of symbols in the query.
        query_len: usize,
    },

    /// A candidate position is outside its referenced string.
    #[error("string position {position} is out of bounds for string length {string_len}")]
    InvalidStringPosition {
        /// The zero-based position supplied by the candidate.
        position: usize,
        /// The number of symbols in the referenced string.
        string_len: usize,
    },

    /// A candidate refers to a string that is not present in the corpus.
    #[error("unknown string id: {0}")]
    UnknownString(usize),

    /// No threshold subsequence can be constructed for the query, possibly
    /// because deletion costs are too small for the threshold.
    #[error(
        "a threshold subsequence cannot be constructed; deletion costs may be too small for the threshold"
    )]
    ThresholdSubsequenceUnavailable,
}

impl Error {
    #[cfg(feature = "persist")]
    pub(crate) fn io(path: &Path, error: std::io::Error) -> Self {
        Self::Io {
            path: path.to_owned(),
            kind: error.kind(),
            message: error.to_string(),
        }
    }
}

/// A specialized `Result` type for errors.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    #[cfg(feature = "persist")]
    use super::Error;

    #[test]
    #[cfg(feature = "persist")]
    fn io_error_preserves_path_kind_and_message() {
        let source = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "not allowed");

        assert_eq!(
            Error::io(std::path::Path::new("index.yurine"), source),
            Error::Io {
                path: "index.yurine".into(),
                kind: std::io::ErrorKind::PermissionDenied,
                message: "not allowed".to_owned(),
            }
        );
    }
}
