//! Postings-index abstractions used during filtering.

use crate::errors::{Error, Result};
use crate::types::Posting;

/// Read access to postings lists.
pub trait PostingsIndex<Symbol> {
    /// Visits indexed occurrences in `(StringId, Position)` order.
    ///
    /// Implementations must not emit duplicates. A visitor keeps the
    /// in-memory implementation allocation-free while allowing a disk-backed
    /// implementation to decode a fallible cursor incrementally.
    fn visit_postings(
        &self,
        symbol: &Symbol,
        visitor: &mut dyn FnMut(Posting) -> Result<()>,
    ) -> Result<()>;

    /// Returns the total frequency of `symbol` in the corpus.
    fn frequency(&self, symbol: &Symbol) -> Result<usize> {
        let mut frequency = 0usize;
        self.visit_postings(symbol, &mut |_| {
            frequency = frequency
                .checked_add(1)
                .ok_or(Error::PlatformSizeOverflow)?;
            Ok(())
        })?;
        Ok(frequency)
    }
}
