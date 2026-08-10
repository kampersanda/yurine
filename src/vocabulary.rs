//! Bidirectional mappings between tokens and compact symbols.

use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;

use crate::errors::{Error, Result};

/// A compact identifier assigned to a token in a [`Vocabulary`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(u32);

impl Symbol {
    /// Creates a symbol from its zero-based integer representation.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the zero-based integer representation.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Returns the symbol as a `usize` value.
    pub fn as_usize(self) -> usize {
        usize::try_from(self.0).unwrap()
    }
}

impl Display for Symbol {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Builds a [`Vocabulary`] by assigning stable, consecutive symbols to tokens.
#[derive(Debug, Clone)]
pub struct VocabularyBuilder<Token> {
    tokens: Vec<Token>,
    symbols: HashMap<Token, Symbol>,
}

impl<Token> VocabularyBuilder<Token>
where
    Token: Clone + Eq + Hash,
{
    /// Creates an empty vocabulary builder.
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            symbols: HashMap::new(),
        }
    }

    /// Returns the existing symbol for `token`, or assigns the next symbol.
    pub fn insert(&mut self, token: Token) -> Result<Symbol> {
        if let Some(&symbol) = self.symbols.get(&token) {
            return Ok(symbol);
        }
        let value = u32::try_from(self.tokens.len()).map_err(|_| Error::SymbolOverflow)?;
        let symbol = Symbol::new(value);
        self.tokens.push(token.clone());
        self.symbols.insert(token, symbol);
        Ok(symbol)
    }

    /// Inserts tokens as needed and returns their symbols in input order.
    pub fn encode<I>(&mut self, tokens: I) -> Result<Vec<Symbol>>
    where
        I: IntoIterator<Item = Token>,
    {
        tokens.into_iter().map(|token| self.insert(token)).collect()
    }

    /// Finalizes this builder and returns a read-only vocabulary.
    pub fn build(self) -> Vocabulary<Token> {
        Vocabulary {
            tokens: self.tokens,
            symbols: self.symbols,
        }
    }
}

impl<Token> Default for VocabularyBuilder<Token>
where
    Token: Clone + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Read-only access to mappings between tokens and symbols.
#[derive(Debug, Clone)]
pub struct Vocabulary<Token> {
    tokens: Vec<Token>,
    symbols: HashMap<Token, Symbol>,
}

impl<Token> Vocabulary<Token>
where
    Token: Eq + Hash,
{
    /// Returns the symbol assigned to `token`.
    pub fn symbol(&self, token: &Token) -> Option<Symbol> {
        self.symbols.get(token).copied()
    }

    /// Returns the token assigned to `symbol`.
    pub fn token(&self, symbol: Symbol) -> Option<&Token> {
        self.tokens.get(symbol.as_usize())
    }

    /// Returns the number of distinct tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns whether the vocabulary contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{Symbol, VocabularyBuilder};

    #[test]
    fn insert_assigns_consecutive_symbols_and_reuses_existing_ones() {
        let mut builder = VocabularyBuilder::new();

        assert_eq!(builder.insert('b').unwrap(), Symbol::new(0));
        assert_eq!(builder.insert('a').unwrap(), Symbol::new(1));
        assert_eq!(builder.insert('b').unwrap(), Symbol::new(0));

        let vocabulary = builder.build();
        assert_eq!(vocabulary.len(), 2);
    }

    #[test]
    fn mappings_are_bidirectional() {
        let mut builder = VocabularyBuilder::new();
        let symbol = builder.insert("東京").unwrap();
        let vocabulary = builder.build();

        assert_eq!(vocabulary.symbol(&"東京"), Some(symbol));
        assert_eq!(vocabulary.token(symbol), Some(&"東京"));
        assert_eq!(vocabulary.token(Symbol::new(1)), None);
    }

    #[test]
    fn encode_preserves_order_and_repeated_tokens() {
        let mut builder = VocabularyBuilder::new();

        let symbols = builder.encode(['a', 'b', 'a']).unwrap();

        assert_eq!(symbols, [Symbol::new(0), Symbol::new(1), Symbol::new(0)]);
    }
}
