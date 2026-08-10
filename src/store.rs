//! Storage abstractions for indexed strings.

use std::collections::HashSet;

use crate::errors::Result;
use crate::types::{StringId, Symbol};

/// A builder for a [`CorpusStore`].
#[derive(Debug, Default)]
pub struct CorpusStoreBuilder {
    strings: Vec<Vec<Symbol>>,
    alphabet: HashSet<Symbol>,
}

impl CorpusStoreBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            alphabet: HashSet::new(),
        }
    }

    /// Adds a sequence to the corpus.
    pub fn add_sequence(&mut self, sequence: Vec<Symbol>) {
        self.alphabet.extend(sequence.iter().copied());
        self.strings.push(sequence);
    }

    /// Finalizes the builder and returns a [`CorpusStore`].
    pub fn build(self) -> CorpusStore {
        let mut alphabet: Vec<_> = self.alphabet.into_iter().collect();
        alphabet.sort_unstable();
        CorpusStore {
            strings: self.strings,
            alphabet,
        }
    }
}

pub struct CorpusStore {
    strings: Vec<Vec<Symbol>>,
    alphabet: Vec<Symbol>,
}

/// Read access to indexed token sequences.
impl CorpusStore {
    /// Returns the sequence identified by `id`, or `None` when it is unknown.
    pub fn sequence(&self, id: StringId) -> Result<Option<&[Symbol]>> {
        let index = id.as_usize();
        if index < self.strings.len() {
            Ok(Some(&self.strings[index]))
        } else {
            Ok(None)
        }
    }

    /// Returns the number of indexed sequences.
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns whether this store contains no sequences.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the alphabet of symbols in the corpus.
    pub fn alphabet(&self) -> &[Symbol] {
        &self.alphabet
    }
}

#[cfg(test)]
mod tests {
    use super::CorpusStoreBuilder;
    use crate::types::Symbol;

    #[test]
    fn alphabet_is_unique_across_sequences() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let third = Symbol::new(2);
        let mut builder = CorpusStoreBuilder::new();
        builder.add_sequence(vec![second, first, second]);
        builder.add_sequence(vec![first, third]);

        let store = builder.build();

        assert_eq!(store.alphabet(), [first, second, third]);
    }
}
