//! Basic types for Yurine.

pub mod costs;

use std::fmt::Display;

use crate::errors::{Error, Result};

/// Identifies a string.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Posting {
    pub string_id: StringId,
    pub position: Position,
}
