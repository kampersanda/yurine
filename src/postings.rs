//! Postings-index abstractions used during filtering.

use std::collections::HashMap;
use std::hash::Hash;

use crate::errors::Result;
use crate::types::Posting;

pub struct PostingsIndex<Symbol> {
    postings: HashMap<Symbol, Vec<Posting>>,
}

impl<Symbol> PostingsIndex<Symbol>
where
    Symbol: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
        }
    }

    /// Adds a posting for the given symbol.
    pub fn add_posting(&mut self, symbol: Symbol, posting: Posting) {
        self.postings.entry(symbol).or_default().push(posting);
    }

    /// Visits indexed occurrences in `(StringId, Position)` order.
    ///
    /// Implementations must not emit duplicates. A visitor keeps the
    /// in-memory implementation allocation-free while allowing a disk-backed
    /// implementation to decode a fallible cursor incrementally.
    pub fn visit_postings(
        &self,
        symbol: &Symbol,
        visitor: &mut dyn FnMut(Posting) -> Result<()>,
    ) -> Result<()> {
        if let Some(postings) = self.postings.get(symbol) {
            for posting in postings {
                visitor(*posting)?;
            }
        }
        Ok(())
    }

    /// Returns the total frequency of `symbol` in the corpus.
    pub fn frequency(&self, symbol: &Symbol) -> Result<usize> {
        Ok(self
            .postings
            .get(symbol)
            .map_or(0, |postings| postings.len()))
    }
}
