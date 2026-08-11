//! Basic types for Yurine.

use std::fmt::Display;
use std::ops::Range;

use crate::errors::{Error, Result};

/// Identifies a string.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StringId(u32);

impl StringId {
    /// Creates an identifier from its zero-based integer representation.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the zero-based integer representation.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Creates an identifier from a usize value.
    pub fn from_usize(value: usize) -> Result<Self> {
        u32::try_from(value)
            .map(Self)
            .map_err(|_| Error::StringIdOverflow)
    }

    /// Returns the identifier as a usize value.
    pub fn as_usize(self) -> usize {
        usize::try_from(self.0).unwrap()
    }
}

impl Display for StringId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// A zero-based symbol position.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Position(u32);

impl Position {
    /// Creates a position from its zero-based integer representation.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the zero-based integer representation.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Creates a position from a usize value.
    pub fn from_usize(value: usize) -> Result<Self> {
        u32::try_from(value)
            .map(Self)
            .map_err(|_| Error::PositionOverflow)
    }

    /// Returns the position as a usize value.
    pub fn as_usize(self) -> usize {
        usize::try_from(self.0).unwrap()
    }
}

impl Display for Position {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// A posting in the corpus, consisting of a string identifier and a symbol position.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Posting {
    pub string_id: StringId,
    pub position: Position,
}

/// Identifies a token in a vocabulary.
///
/// A symbol is meaningful only together with the vocabulary that assigned it.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(u32);

impl Symbol {
    /// The symbol reserved for tokens absent from a vocabulary.
    pub const UNKNOWN: Self = Self(u32::MAX);

    /// Creates a symbol from its zero-based integer representation.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the zero-based integer representation.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Creates a symbol from a usize value.
    pub fn from_usize(value: usize) -> Result<Self> {
        let value = u32::try_from(value).map_err(|_| Error::SymbolOverflow)?;
        if value == Self::UNKNOWN.0 {
            Err(Error::SymbolOverflow)
        } else {
            Ok(Self(value))
        }
    }

    /// Returns whether this is the reserved unknown-token symbol.
    pub const fn is_unknown(self) -> bool {
        self.0 == Self::UNKNOWN.0
    }

    /// Returns the symbol as a usize value.
    pub fn as_usize(self) -> usize {
        usize::try_from(self.0).unwrap()
    }
}

impl Display for Symbol {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// A UTF-8 byte offset relative to a single string.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteOffset(u32);

impl ByteOffset {
    /// Creates an offset from its fixed-width integer representation.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the fixed-width integer representation.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Creates an offset from a usize value.
    pub fn from_usize(value: usize) -> Result<Self> {
        u32::try_from(value)
            .map(Self)
            .map_err(|_| Error::ByteOffsetOverflow)
    }

    /// Returns the offset as a usize value.
    pub fn as_usize(self) -> usize {
        usize::try_from(self.0).unwrap()
    }
}

impl Display for ByteOffset {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// A token's UTF-8 byte range relative to its original string.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    start: ByteOffset,
    end: ByteOffset,
}

impl ByteRange {
    /// Creates a byte range from fixed-width endpoints.
    pub const fn new(start: ByteOffset, end: ByteOffset) -> Self {
        Self { start, end }
    }

    /// Returns the inclusive start byte offset.
    pub const fn start(self) -> ByteOffset {
        self.start
    }

    /// Returns the exclusive end byte offset.
    pub const fn end(self) -> ByteOffset {
        self.end
    }

    /// Converts the fixed-width endpoints to platform-sized offsets.
    pub fn as_range(self) -> Range<usize> {
        self.start.as_usize()..self.end.as_usize()
    }
}

impl TryFrom<Range<usize>> for ByteRange {
    type Error = Error;

    fn try_from(range: Range<usize>) -> Result<Self> {
        Ok(Self {
            start: ByteOffset::from_usize(range.start)?,
            end: ByteOffset::from_usize(range.end)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, size_of};

    use super::{ByteOffset, ByteRange, Position, Posting, StringId, Symbol};
    use crate::errors::Error;

    #[test]
    fn fixed_width_values_round_trip_through_public_representations() {
        let string_id = StringId::from_usize(12).unwrap();
        let position = Position::from_usize(34).unwrap();
        let symbol = Symbol::from_usize(56).unwrap();
        let byte_offset = ByteOffset::from_usize(78).unwrap();
        let byte_range = ByteRange::try_from(78..90).unwrap();

        assert_eq!(string_id.get(), 12);
        assert_eq!(string_id.as_usize(), 12);
        assert_eq!(string_id.to_string(), "12");
        assert_eq!(position.get(), 34);
        assert_eq!(position.as_usize(), 34);
        assert_eq!(position.to_string(), "34");
        assert_eq!(symbol.get(), 56);
        assert_eq!(symbol.as_usize(), 56);
        assert_eq!(symbol.to_string(), "56");
        assert_eq!(byte_offset.get(), 78);
        assert_eq!(byte_offset.as_usize(), 78);
        assert_eq!(byte_offset.to_string(), "78");
        assert_eq!(byte_range.start(), byte_offset);
        assert_eq!(byte_range.end(), ByteOffset::new(90));
        assert_eq!(byte_range.as_range(), 78..90);
    }

    #[test]
    fn unknown_symbol_is_reserved_from_vocabulary_values() {
        assert_eq!(
            Symbol::from_usize(u32::MAX as usize),
            Err(Error::SymbolOverflow)
        );
        assert!(Symbol::UNKNOWN.is_unknown());
        assert!(!Symbol::new(u32::MAX - 1).is_unknown());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn byte_offset_rejects_values_larger_than_u32() {
        let too_large = usize::try_from(u64::from(u32::MAX) + 1).unwrap();

        assert_eq!(
            ByteOffset::from_usize(too_large),
            Err(Error::ByteOffsetOverflow)
        );
    }

    #[test]
    fn storage_types_have_fixed_width_layouts() {
        assert_eq!(size_of::<StringId>(), 4);
        assert_eq!(size_of::<Position>(), 4);
        assert_eq!(size_of::<Symbol>(), 4);
        assert_eq!(size_of::<ByteOffset>(), 4);
        assert_eq!(align_of::<ByteOffset>(), 4);
        assert_eq!(size_of::<Posting>(), 8);
        assert_eq!(align_of::<Posting>(), 4);
        assert_eq!(size_of::<ByteRange>(), 8);
        assert_eq!(align_of::<ByteRange>(), 4);
    }
}
