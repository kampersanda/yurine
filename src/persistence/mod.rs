//! Building blocks for persistent, memory-mapped indexes.
//!
//! Persisted files are immutable snapshots. While a file is mapped, callers
//! must not truncate or modify it in place. Publish replacements by atomically
//! renaming a completed new file instead.

mod codec;
// The format reader is consumed by the save/open APIs added in implementation
// unit 2. Keeping it internal avoids committing those APIs prematurely.
#[allow(dead_code)]
pub(crate) mod format;
#[allow(dead_code)]
pub(crate) mod storage;

pub use codec::{CharCodec, StringCodec, TokenCodec};
