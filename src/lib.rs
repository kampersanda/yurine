//! Fast, exact search for sequence segments under weighted edit distance.
//!
//! Yurine indexes sequences of any cloneable, hashable token type. A query
//! matches every non-empty segment whose edit distance is at most the supplied
//! threshold. Returned ranges are zero-based token ranges; callers remain
//! responsible for tokenization and for mapping them back to source text.
//! Here, "exact search" means that qualifying approximate matches are not
//! omitted; it does not mean exact string matching.
//!
//! # Quick start
//!
//! ```
//! use yurine::costs::{Cost, custom::CustomCosts};
//! use yurine::search::{SearchEngineBuilder, range_search::RangeSearchParams};
//!
//! # fn main() -> yurine::errors::Result<()> {
//! let mut builder = SearchEngineBuilder::new();
//! let jinbocho = builder.add_sequence([
//!     "Jinbocho", "is", "a", "book", "town", "known", "for", "curry",
//! ])?;
//!
//! let engine = builder.build()?;
//! let mut costs = CustomCosts::default();
//! costs.set_substitution("district", "town", Cost::new_const(0.25));
//! let searcher = engine.range_searcher(costs);
//! let matches = searcher.search(
//!     &["book", "district", "known", "for", "curry"],
//!     &RangeSearchParams::new(Cost::new_const(0.25)),
//! )?;
//!
//! assert_eq!(matches.len(), 1);
//! assert_eq!(matches[0].sequence_id, jinbocho);
//! assert_eq!(matches[0].distance, Cost::new_const(0.25));
//! assert_eq!(matches[0].token_range.start.get(), 3);
//! assert_eq!(matches[0].token_range.end.get(), 8);
//! # Ok(())
//! # }
//! ```
//!
//! The query matches the `book town known for curry` segment rather than the
//! whole sequence. It does not occur verbatim: replacing `district` with
//! `town` costs `0.25`, while unspecified substitutions retain unit cost.
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
