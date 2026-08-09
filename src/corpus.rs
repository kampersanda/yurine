//! Storage abstractions for indexed strings.

use crate::errors::Result;
use crate::types::StringId;

/// Read access to indexed token sequences.
pub trait CorpusStore<Symbol> {
    /// A sequence returned by this store.
    type Sequence<'a>: AsRef<[Symbol]>
    where
        Self: 'a,
        Symbol: 'a;

    /// Returns the sequence identified by `id`, or `None` when it is unknown.
    fn sequence(&self, id: StringId) -> Result<Option<Self::Sequence<'_>>>;

    /// Returns the number of indexed sequences.
    fn len(&self) -> usize;

    /// Returns whether this store contains no sequences.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the alphabet of symbols in the corpus.
    fn alphabet(&self) -> Vec<Symbol>;
}
