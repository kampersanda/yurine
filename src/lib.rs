//! Fast, exact search for sequence segments under weighted edit distance.
//!
//! Yurine indexes sequences of any cloneable, hashable token type. A query
//! matches every non-empty segment whose edit distance is at most the supplied
//! threshold. Returned ranges are zero-based token ranges; callers remain
//! responsible for tokenization and for mapping them back to source text.
//!
//! # Quick start
//!
//! ```
//! use yurine::costs::{Cost, levenshtein::LevenshteinCosts};
//! use yurine::search::{SearchEngineBuilder, range_search::RangeSearchParams};
//!
//! # fn main() -> yurine::errors::Result<()> {
//! let mut builder = SearchEngineBuilder::new();
//! let book_town = builder.add_sequence(["Jinbocho", "is", "a", "book", "town"])?;
//! builder.add_sequence(["Jinbocho", "is", "famous", "for", "curry"])?;
//!
//! let engine = builder.build()?;
//! let searcher = engine.range_searcher(LevenshteinCosts::new());
//! let matches = searcher.search(
//!     &["Jinbocho", "is", "a", "book", "town"],
//!     &RangeSearchParams::new(Cost::ZERO),
//! )?;
//!
//! assert_eq!(matches.len(), 1);
//! assert_eq!(matches[0].sequence_id, book_town);
//! assert_eq!(matches[0].token_range.start.get(), 0);
//! assert_eq!(matches[0].token_range.end.get(), 5);
//! # Ok(())
//! # }
//! ```
//!
//! Build an index once, then create any number of searchers with
//! [`levenshtein::LevenshteinCosts`](costs::levenshtein::LevenshteinCosts),
//! [`custom::CustomCosts`](costs::custom::CustomCosts), or
//! [`embedding::CosineEmbeddingCosts`](costs::embedding::CosineEmbeddingCosts).
//! Enable the `persist` feature to add a `persistence` module for saving
//! immutable, memory-mapped snapshots.
#![warn(missing_docs)]

pub mod costs;
pub mod errors;
#[cfg(feature = "persist")]
pub mod persistence;
mod postings;
pub mod search;
mod storage;
mod store;
pub mod types;
mod vocabulary;
