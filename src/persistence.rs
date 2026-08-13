//! Building blocks for persistent, memory-mapped indexes.
//!
//! Persisted files are immutable snapshots. While a file is mapped, callers
//! must not truncate or modify it in place. Publish replacements by atomically
//! renaming a completed new file instead.
//!
//! The token codec is part of the file format. Use [`CharCodec`] for `char`,
//! [`StringCodec`] for [`String`], or implement [`TokenCodec`] for another
//! token type. A codec's identifier, version, and encoding must remain stable
//! for as long as its files need to be readable.
//!
//! ```
//! use tempfile::tempdir;
//! use yurine::persistence::StringCodec;
//! use yurine::search::{SearchEngine, SearchEngineBuilder};
//!
//! # fn main() -> yurine::errors::Result<()> {
//! let directory = tempdir().expect("create temporary directory");
//! let path = directory.path().join("index.yurine");
//!
//! let mut builder = SearchEngineBuilder::new();
//! builder.add_sequence(["Jinbocho", "book", "town", "curry"].map(str::to_owned))?;
//! builder.build()?.save_with(&path, &StringCodec)?;
//!
//! let engine = SearchEngine::open_with(&path, &StringCodec)?;
//! engine.verify()?;
//! # Ok(())
//! # }
//! ```

mod codec;
// The format reader is consumed by the save/open APIs added in implementation
// unit 2. Keeping it internal avoids committing those APIs prematurely.
#[allow(dead_code)]
pub(crate) mod format;
#[allow(dead_code)]
pub(crate) mod storage;

pub use codec::{CharCodec, StringCodec, TokenCodec};
