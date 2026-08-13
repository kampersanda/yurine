//! Fast, exact search for sequence segments under weighted edit distance.
//!
//! Yurine indexes sequences of any cloneable, hashable token type. A query
//! matches every non-empty segment whose edit distance is at most the supplied
//! threshold. Returned ranges are zero-based token ranges; callers remain
//! responsible for tokenization and for mapping them back to source text.
//! Here, "exact search" means that qualifying approximate matches are not
//! omitted; it does not mean exact string matching.
//!
//! The search algorithm follows the filtering–verification framework of Koide
//! et al. \[1]; see [References](#references).
//!
//! # Quick start
//!
//! ```
//! use yurine::costs::CustomCosts;
//! use yurine::{Cost, RangeSearchParams, SearchEngineBuilder};
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
//!     &RangeSearchParams::new(0.25),
//! )?;
//!
//! assert_eq!(matches.len(), 1);
//! assert_eq!(matches[0].sequence_id, jinbocho);
//! assert_eq!(matches[0].distance, 0.25);
//! assert_eq!(matches[0].token_range, 3..8);
//! # Ok(())
//! # }
//! ```
//!
//! The query matches the `book town known for curry` segment rather than the
//! whole sequence. It does not occur verbatim: replacing `district` with
//! `town` costs `0.25`, while unspecified substitutions retain unit cost.
//!
//! # Embedding-based search
//!
//! [`CosineEmbeddingCosts`](costs::CosineEmbeddingCosts) derives
//! substitution costs from static token embeddings. Tokens with similar
//! vectors can therefore match without an explicit substitution rule.
//!
//! ```
//! use std::num::NonZeroUsize;
//! use yurine::costs::{CosineEmbeddingCosts, EmbeddingStoreBuilder};
//! use yurine::{RangeSearchParams, SearchEngineBuilder};
//!
//! # fn main() -> yurine::errors::Result<()> {
//! let mut builder = SearchEngineBuilder::new();
//! let jinbocho = builder.add_sequence([
//!     "Visitors", "enjoy", "books", "and", "curry", "in", "Jinbocho",
//! ])?;
//! let engine = builder.build()?;
//!
//! let mut embeddings = EmbeddingStoreBuilder::new(NonZeroUsize::new(2).unwrap());
//! embeddings.insert("literature", [1.0, 0.0])?;
//! embeddings.insert("books", [0.8, 0.6])?;
//! let costs = CosineEmbeddingCosts::new(embeddings.build());
//!
//! let matches = engine.range_searcher(costs).search(
//!     &["literature", "and", "curry"],
//!     &RangeSearchParams::new(0.2),
//! )?;
//!
//! assert_eq!(matches.len(), 1);
//! assert_eq!(matches[0].sequence_id, jinbocho);
//! assert_eq!(matches[0].token_range, 2..5);
//! assert!((matches[0].distance - 0.2).abs() < 1e-6);
//! # Ok(())
//! # }
//! ```
//!
//! Here, `literature` matches `books` with cosine distance `0.2`, returning the
//! `books and curry` segment from the longer sequence.
//!
//! # Saving and loading an index
//!
//! Enable the `persist` feature to save an immutable index and reopen it
//! without rebuilding. Opening restores the vocabulary in memory while the
//! corpus and postings remain memory-mapped.
//!
//! ```
//! # #[cfg(feature = "persist")]
//! # fn example() -> yurine::errors::Result<()> {
//! use tempfile::tempdir;
//! use yurine::costs::LevenshteinCosts;
//! use yurine::persistence::StringCodec;
//! use yurine::{RangeSearchParams, SearchEngine, SearchEngineBuilder};
//!
//! let directory = tempdir().expect("create temporary directory");
//! let path = directory.path().join("jinbocho.yurine");
//!
//! let mut builder = SearchEngineBuilder::new();
//! let jinbocho = builder.add_sequence(
//!     ["Jinbocho", "is", "a", "book", "town", "known", "for", "curry"]
//!         .map(str::to_owned),
//! )?;
//! builder.build()?.save_with(&path, &StringCodec)?;
//!
//! let engine = SearchEngine::open_with(&path, &StringCodec)?;
//! engine.verify()?;
//! let query = ["book", "town"].map(str::to_owned);
//! let matches = engine.range_searcher(LevenshteinCosts::new()).search(
//!     &query,
//!     &RangeSearchParams::new(0.0),
//! )?;
//!
//! assert_eq!(matches[0].sequence_id, jinbocho);
//! assert_eq!(matches[0].token_range, 3..5);
//! # Ok(())
//! # }
//! # #[cfg(feature = "persist")]
//! # example()?;
//! # Ok::<(), yurine::errors::Error>(())
//! ```
//!
//! `StringCodec` and `CharCodec` cover the built-in token types. Other token
//! types require a stable `TokenCodec` implementation.
//! Call [`SearchEngine::verify`](search::SearchEngine::verify) after opening an
//! untrusted snapshot, and never modify or truncate a file while it is mapped.
//!
//! Build an index once, then create any number of searchers with
//! [`LevenshteinCosts`](costs::LevenshteinCosts),
//! [`CustomCosts`](costs::CustomCosts), or
//! [`CosineEmbeddingCosts`](costs::CosineEmbeddingCosts).
//!
//! # References
//!
//! Yurine implements the weighted edit distance search algorithm proposed for
//! subtrajectory search in road networks: a query subsequence minimizing the
//! number of candidates is selected by a two-approximation algorithm,
//! candidate anchors are collected from an inverted index, and each candidate
//! is verified with a bidirectional trie. Yurine indexes sequences of
//! arbitrary tokens rather than road-network trajectories.
//!
//! 1. Satoshi Koide, Chuan Xiao, and Yoshiharu Ishikawa. Fast Subtrajectory
//!    Similarity Search in Road Networks under Weighted Edit Distance
//!    Constraints. *PVLDB*, 13(11): 2188–2201, 2020.
//!    <https://doi.org/10.14778/3407790.3407818>
#![warn(missing_docs)]

pub mod costs;
pub mod errors;
#[cfg(feature = "persist")]
pub mod persistence;
mod postings;
pub mod search;
mod storage;
mod store;
mod types;
mod vocabulary;

pub use costs::Cost;
pub use search::{Match, RangeSearchParams, SearchEngine, SearchEngineBuilder};

// Keep README examples in the same doctest suite as the public API docs without
// adding the README itself to the rendered crate documentation.
#[cfg(all(doctest, feature = "persist"))]
#[doc = include_str!("../README.md")]
mod readme {}
