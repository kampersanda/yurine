//! Postings index mapping symbols to their occurrences in the corpus.

use std::collections::HashMap;
use std::hash::Hash;

use crate::errors::Result;
use crate::types::Posting;

/// Postings index mapping symbols to their occurrences in the corpus.
pub struct PostingsIndex<Symbol> {
    postings: HashMap<Symbol, Vec<Posting>>,
}

impl<Symbol> PostingsIndex<Symbol>
where
    Symbol: Eq + Hash,
{
    /// Creates a new postings index.
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
        }
    }

    /// Adds a posting for the given symbol.
    pub fn add_posting(&mut self, symbol: Symbol, posting: Posting) {
        self.postings.entry(symbol).or_default().push(posting);
    }

    /// Returns indexed occurrences in `(StringId, Position)` order.
    ///
    /// The iterator does not emit duplicates.
    pub fn postings(&self, symbol: &Symbol) -> impl Iterator<Item = Posting> + '_ {
        self.postings.get(symbol).into_iter().flatten().copied()
    }

    /// Returns the total frequency of `symbol` in the corpus.
    pub fn frequency(&self, symbol: &Symbol) -> Result<usize> {
        Ok(self
            .postings
            .get(symbol)
            .map_or(0, |postings| postings.len()))
    }
}
