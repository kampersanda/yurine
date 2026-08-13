//! Storage abstractions for indexed strings.

use crate::errors::Result;
#[cfg(feature = "persist")]
use crate::persistence::storage::MappedSlice;
use crate::storage::Storage;
use crate::types::{SequenceId, Symbol};

/// A builder for a [`CorpusStore`].
#[derive(Debug)]
pub(crate) struct CorpusStoreBuilder {
    symbols: Vec<Symbol>,
    string_offsets: Vec<u64>,
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
        }
    }

    /// Adds a data string.
    pub(crate) fn add_string(&mut self, string: Vec<Symbol>) {
        let string_end = self.symbols.len() as u64 + string.len() as u64;
        self.symbols.extend(string);
        self.string_offsets.push(string_end);
    }

    /// Finalizes the builder and returns a [`CorpusStore`].
    pub(crate) fn build(mut self, symbol_count: usize) -> CorpusStore {
        self.symbols.shrink_to_fit();
        self.string_offsets.shrink_to_fit();
        CorpusStore {
            symbols: Storage::Owned(self.symbols.into_boxed_slice()),
            string_offsets: Storage::Owned(self.string_offsets.into_boxed_slice()),
            symbol_count,
        }
    }
}

pub(crate) struct CorpusStore {
    symbols: Storage<Symbol>,
    string_offsets: Storage<u64>,
    symbol_count: usize,
}

/// Read access to indexed strings.
impl CorpusStore {
    /// Returns the string identified by `id`, or `None` when it is unknown.
    pub(crate) fn string(&self, id: SequenceId) -> Result<Option<&[Symbol]>> {
        let Some((start, end)) = self.string_bounds(id)? else {
            return Ok(None);
        };
        let string = &self.symbols[start..end];
        if let Some(symbol) = string
            .iter()
            .find(|symbol| symbol.is_unknown() || symbol.as_usize() >= self.symbol_count)
        {
            return Err(crate::errors::Error::UnknownStringSymbol(symbol.get()));
        }
        Ok(Some(string))
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

    pub(crate) fn verify(&self) -> Result<()> {
        for raw_id in 0..self.len() {
            self.string(SequenceId::from_usize(raw_id)?)?;
        }
        Ok(())
    }

    pub(crate) fn symbol_len(&self) -> usize {
        self.symbols.len()
    }

    #[cfg(feature = "persist")]
    pub(crate) fn from_mapped(
        symbols: MappedSlice<Symbol>,
        string_offsets: MappedSlice<u64>,
        symbol_count: usize,
    ) -> Self {
        Self {
            symbols: Storage::Mapped(symbols),
            string_offsets: Storage::Mapped(string_offsets),
            symbol_count,
        }
    }

    #[cfg(feature = "persist")]
    pub(crate) fn symbols(&self) -> &[Symbol] {
        &self.symbols
    }

    #[cfg(feature = "persist")]
    pub(crate) fn string_offsets(&self) -> &[u64] {
        &self.string_offsets
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
    fn stores_symbols_from_each_string() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let third = Symbol::new(2);
        let mut builder = CorpusStoreBuilder::new();
        builder.add_string(vec![second, first, second]);
        builder.add_string(vec![first, third]);

        let store = builder.build(3);

        assert_eq!(
            store.string(SequenceId::new(0)).unwrap(),
            Some(&[second, first, second][..])
        );
    }

    #[test]
    fn returns_each_corpus_string() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let mut builder = CorpusStoreBuilder::new();
        builder.add_string(vec![first, second]);
        let store = builder.build(2);

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

        let store = builder.build(2);

        assert_eq!(&*store.symbols, [first, second, second]);
        assert_eq!(&*store.string_offsets, [0, 2, 2, 3]);
        assert_eq!(store.string(SequenceId::new(1)).unwrap(), Some(&[][..]));
    }

    #[test]
    fn unknown_string_returns_none() {
        let store = CorpusStoreBuilder::new().build(0);

        assert!(store.is_empty());
        assert_eq!(store.string(SequenceId::new(0)).unwrap(), None);
    }
}
