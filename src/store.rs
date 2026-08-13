//! Storage abstractions for indexed strings.

use std::collections::HashSet;

use crate::errors::Result;
use crate::types::{SequenceId, Symbol};

/// A builder for a [`CorpusStore`].
#[derive(Debug)]
pub(crate) struct CorpusStoreBuilder {
    symbols: Vec<Symbol>,
    string_offsets: Vec<u64>,
    alphabet: HashSet<Symbol>,
}

impl Default for CorpusStoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CorpusStoreBuilder {
    /// Creates a new builder.
    pub(crate) fn new() -> Self {
        Self {
            symbols: Vec::new(),
            string_offsets: vec![0],
            alphabet: HashSet::new(),
        }
    }

    /// Adds a data string.
    pub(crate) fn add_string(&mut self, string: Vec<Symbol>) {
        let string_end = self.symbols.len() as u64 + string.len() as u64;
        self.alphabet.extend(string.iter().copied());
        self.symbols.extend(string);
        self.string_offsets.push(string_end);
    }

    /// Finalizes the builder and returns a [`CorpusStore`].
    pub(crate) fn build(mut self) -> CorpusStore {
        self.symbols.shrink_to_fit();
        self.string_offsets.shrink_to_fit();
        let mut alphabet: Vec<_> = self.alphabet.into_iter().collect();
        alphabet.sort_unstable();
        CorpusStore {
            symbols: self.symbols,
            string_offsets: self.string_offsets,
            alphabet,
        }
    }
}

pub(crate) struct CorpusStore {
    symbols: Vec<Symbol>,
    string_offsets: Vec<u64>,
    alphabet: Vec<Symbol>,
}

/// Read access to indexed strings.
impl CorpusStore {
    /// Returns the string identified by `id`, or `None` when it is unknown.
    pub(crate) fn string(&self, id: SequenceId) -> Result<Option<&[Symbol]>> {
        let Some((start, end)) = self.string_bounds(id)? else {
            return Ok(None);
        };
        Ok(Some(&self.symbols[start..end]))
    }

    /// Returns the number of indexed strings.
    pub(crate) fn len(&self) -> usize {
        self.string_offsets.len() - 1
    }

    /// Returns whether this store contains no strings.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the alphabet of symbols in the corpus.
    pub(crate) fn alphabet(&self) -> &[Symbol] {
        &self.alphabet
    }

    fn string_bounds(&self, id: SequenceId) -> Result<Option<(usize, usize)>> {
        let index = id.as_usize();
        let Some(end_index) = index.checked_add(1) else {
            return Ok(None);
        };
        let (Some(&start), Some(&end)) = (
            self.string_offsets.get(index),
            self.string_offsets.get(end_index),
        ) else {
            return Ok(None);
        };
        Ok(Some((start as usize, end as usize)))
    }
}

#[cfg(test)]
mod tests {
    use super::CorpusStoreBuilder;
    use crate::types::{SequenceId, Symbol};

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

    #[test]
    fn returns_each_corpus_string() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let mut builder = CorpusStoreBuilder::new();
        builder.add_string(vec![first, second]);
        let store = builder.build();

        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert_eq!(
            store.string(SequenceId::new(0)).unwrap(),
            Some(&[first, second][..])
        );
    }

    #[test]
    fn stores_corpus_strings_in_flat_arrays() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let mut builder = CorpusStoreBuilder::new();
        builder.add_string(vec![first, second]);
        builder.add_string(Vec::new());
        builder.add_string(vec![second]);

        let store = builder.build();

        assert_eq!(store.symbols, [first, second, second]);
        assert_eq!(store.string_offsets, [0, 2, 2, 3]);
        assert_eq!(store.string(SequenceId::new(1)).unwrap(), Some(&[][..]));
    }

    #[test]
    fn unknown_string_returns_none() {
        let store = CorpusStoreBuilder::new().build();

        assert!(store.is_empty());
        assert_eq!(store.string(SequenceId::new(0)).unwrap(), None);
    }
}
