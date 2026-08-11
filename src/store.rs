//! Storage abstractions for indexed strings.

use std::collections::HashSet;
use std::ops::Range;

use crate::errors::{Error, Result};
use crate::types::{ByteRange, Position, StringId, Symbol};

/// A builder for a [`CorpusStore`].
#[derive(Debug)]
pub struct CorpusStoreBuilder {
    symbols: Vec<Symbol>,
    string_offsets: Vec<u64>,
    byte_ranges: Vec<ByteRange>,
    alphabet: HashSet<Symbol>,
}

impl Default for CorpusStoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CorpusStoreBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            string_offsets: vec![0],
            byte_ranges: Vec::new(),
            alphabet: HashSet::new(),
        }
    }

    /// Adds a string and the original UTF-8 byte range of each symbol.
    pub fn add_string(
        &mut self,
        string: Vec<Symbol>,
        byte_ranges: Vec<Range<usize>>,
    ) -> Result<()> {
        assert_eq!(string.len(), byte_ranges.len());
        let byte_ranges = byte_ranges
            .into_iter()
            .map(ByteRange::try_from)
            .collect::<Result<Vec<_>>>()?;
        let end = self
            .symbols
            .len()
            .checked_add(string.len())
            .and_then(|end| u64::try_from(end).ok())
            .ok_or(Error::PlatformSizeOverflow)?;
        self.alphabet.extend(string.iter().copied());
        self.symbols.extend(string);
        self.byte_ranges.extend(byte_ranges);
        self.string_offsets.push(end);
        Ok(())
    }

    /// Finalizes the builder and returns a [`CorpusStore`].
    pub fn build(self) -> CorpusStore {
        let mut alphabet: Vec<_> = self.alphabet.into_iter().collect();
        alphabet.sort_unstable();
        CorpusStore {
            symbols: self.symbols,
            string_offsets: self.string_offsets,
            byte_ranges: self.byte_ranges,
            alphabet,
        }
    }
}

pub struct CorpusStore {
    symbols: Vec<Symbol>,
    string_offsets: Vec<u64>,
    byte_ranges: Vec<ByteRange>,
    alphabet: Vec<Symbol>,
}

/// Read access to indexed strings.
impl CorpusStore {
    /// Returns the string identified by `id`, or `None` when it is unknown.
    pub fn string(&self, id: StringId) -> Result<Option<&[Symbol]>> {
        let Some((start, end)) = self.string_bounds(id)? else {
            return Ok(None);
        };
        Ok(Some(&self.symbols[start..end]))
    }

    /// Returns a token's original UTF-8 byte range by value.
    pub fn byte_range(&self, id: StringId, position: Position) -> Result<Option<Range<usize>>> {
        let Some((start, end)) = self.string_bounds(id)? else {
            return Ok(None);
        };
        let Some(index) = start.checked_add(position.as_usize()) else {
            return Ok(None);
        };
        if index >= end {
            return Ok(None);
        }
        self.byte_ranges[index].as_range().map(Some)
    }

    /// Returns the number of indexed strings.
    pub fn len(&self) -> usize {
        self.string_offsets.len() - 1
    }

    /// Returns whether this store contains no strings.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the alphabet of symbols in the corpus.
    pub fn alphabet(&self) -> &[Symbol] {
        &self.alphabet
    }

    fn string_bounds(&self, id: StringId) -> Result<Option<(usize, usize)>> {
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
        Ok(Some((
            usize::try_from(start).map_err(|_| Error::PlatformSizeOverflow)?,
            usize::try_from(end).map_err(|_| Error::PlatformSizeOverflow)?,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::CorpusStoreBuilder;
    use crate::errors::Error;
    use crate::types::{Position, StringId, Symbol};

    #[test]
    fn alphabet_is_unique_across_strings() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let third = Symbol::new(2);
        let mut builder = CorpusStoreBuilder::new();
        builder
            .add_string(vec![second, first, second], vec![0..1, 1..2, 2..3])
            .unwrap();
        builder
            .add_string(vec![first, third], vec![0..1, 1..2])
            .unwrap();

        let store = builder.build();

        assert_eq!(store.alphabet(), [first, second, third]);
    }

    #[test]
    fn returns_each_string_with_its_byte_ranges() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let mut builder = CorpusStoreBuilder::new();
        builder
            .add_string(vec![first, second], vec![0..1, 1..4])
            .unwrap();
        let store = builder.build();

        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert_eq!(
            store.string(StringId::new(0)).unwrap(),
            Some(&[first, second][..])
        );
        assert_eq!(
            store
                .byte_range(StringId::new(0), Position::new(0))
                .unwrap(),
            Some(0..1)
        );
        assert_eq!(
            store
                .byte_range(StringId::new(0), Position::new(1))
                .unwrap(),
            Some(1..4)
        );
    }

    #[test]
    fn stores_strings_and_byte_ranges_in_flat_arrays() {
        let first = Symbol::new(0);
        let second = Symbol::new(1);
        let mut builder = CorpusStoreBuilder::new();
        builder
            .add_string(vec![first, second], vec![0..1, 1..2])
            .unwrap();
        builder.add_string(Vec::new(), Vec::new()).unwrap();
        builder
            .add_string(vec![second], std::iter::once(0..1).collect())
            .unwrap();

        let store = builder.build();

        assert_eq!(store.symbols, [first, second, second]);
        assert_eq!(store.string_offsets, [0, 2, 2, 3]);
        assert_eq!(store.byte_ranges.len(), 3);
        assert_eq!(store.string(StringId::new(1)).unwrap(), Some(&[][..]));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_byte_offsets_larger_than_u32() {
        let too_large = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
        let mut builder = CorpusStoreBuilder::new();

        let error = builder
            .add_string(
                vec![Symbol::new(0)],
                std::iter::once(0..too_large).collect(),
            )
            .unwrap_err();

        assert_eq!(error, Error::ByteOffsetOverflow);
        assert_eq!(builder.string_offsets, [0]);
        assert!(builder.symbols.is_empty());
        assert!(builder.byte_ranges.is_empty());
    }

    #[test]
    fn unknown_string_returns_none() {
        let store = CorpusStoreBuilder::new().build();

        assert!(store.is_empty());
        assert_eq!(store.string(StringId::new(0)).unwrap(), None);
        assert_eq!(
            store
                .byte_range(StringId::new(0), Position::new(0))
                .unwrap(),
            None
        );
    }
}
