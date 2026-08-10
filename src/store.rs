//! Storage abstractions for indexed strings.

use std::collections::HashSet;
use std::hash::Hash;

use crate::errors::Result;
use crate::types::StringId;

/// A builder for a [`CorpusStore`].
pub struct CorpusStoreBuilder<Symbol> {
    strings: Vec<Vec<Symbol>>,
    alphabet: Vec<Symbol>,
}

impl<Symbol> CorpusStoreBuilder<Symbol>
where
    Symbol: Eq + Clone + Hash + Ord,
{
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            alphabet: Vec::new(),
        }
    }

    /// Adds a sequence to the corpus.
    pub fn add_sequence(&mut self, sequence: Vec<Symbol>) {
        let mut unique_symbols = HashSet::new();
        for symbol in &sequence {
            if unique_symbols.insert(symbol.clone()) {
                self.alphabet.push(symbol.clone());
            }
        }
        self.strings.push(sequence);
        self.alphabet.sort();
    }

    /// Finalizes the builder and returns a [`CorpusStore`].
    pub fn build(self) -> CorpusStore<Symbol> {
        CorpusStore {
            strings: self.strings,
            alphabet: self.alphabet,
        }
    }
}

pub struct CorpusStore<Symbol> {
    strings: Vec<Vec<Symbol>>,
    alphabet: Vec<Symbol>,
}

/// Read access to indexed token sequences.
impl<Symbol> CorpusStore<Symbol> {
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
