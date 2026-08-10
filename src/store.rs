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

    /// Adds a string to the corpus.
    pub fn add_string(&mut self, string: Vec<Symbol>) {
        self.alphabet.extend(string.iter().copied());
        self.strings.push(string);
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

/// Read access to indexed strings.
impl CorpusStore {
    /// Returns the string identified by `id`, or `None` when it is unknown.
    pub fn string(&self, id: StringId) -> Result<Option<&[Symbol]>> {
        let index = id.as_usize();
        if index < self.strings.len() {
            Ok(Some(&self.strings[index]))
        } else {
            Ok(None)
        }
    }

    /// Returns the number of indexed strings.
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns whether this store contains no strings.
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
    fn alphabet_is_unique_across_strings() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let third = Symbol::new(2);
        let mut builder = CorpusStoreBuilder::new();
        builder.add_string(vec![second, first, second]);
        builder.add_string(vec![first, third]);

        let store = builder.build();

        assert_eq!(store.alphabet(), [first, second, third]);
    }
}
