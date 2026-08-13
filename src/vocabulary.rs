//! Bidirectional mappings between tokens and compact symbols.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::hash::Hash;

use crate::errors::Result;
use crate::types::Symbol;

/// Collects token frequencies for building a [`Vocabulary`].
#[derive(Debug, Clone)]
pub struct VocabularyBuilder<T> {
    tokens: Vec<T>,
    frequencies: HashMap<T, usize>,
}

impl<T> VocabularyBuilder<T>
where
    T: Clone + Eq + Hash,
{
    /// Creates an empty vocabulary builder.
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            frequencies: HashMap::new(),
        }
    }

    /// Records one occurrence of `token`.
    pub fn insert(&mut self, token: T) {
        if let Some(frequency) = self.frequencies.get_mut(&token) {
            *frequency += 1;
        } else {
            self.tokens.push(token.clone());
            self.frequencies.insert(token, 1);
        }
    }

    /// Records every token from `tokens`.
    pub fn insert_all<I>(&mut self, tokens: I)
    where
        I: IntoIterator<Item = T>,
    {
        for token in tokens {
            self.insert(token);
        }
    }

    /// Assigns consecutive symbols by descending token frequency.
    ///
    /// Tokens with the same frequency retain their first-seen order.
    pub fn build(self) -> Result<Vocabulary<T>> {
        let mut tokens = self.tokens;
        tokens.sort_by_key(|token| Reverse(self.frequencies[token]));

        let mut symbols = HashMap::with_capacity(tokens.len());
        for (index, token) in tokens.iter().cloned().enumerate() {
            symbols.insert(token, Symbol::from_usize(index)?);
        }

        Ok(Vocabulary { tokens, symbols })
    }
}

impl<T> Default for VocabularyBuilder<T>
where
    T: Clone + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Read-only access to mappings between tokens and symbols.
#[derive(Debug, Clone)]
pub struct Vocabulary<T> {
    tokens: Vec<T>,
    symbols: HashMap<T, Symbol>,
}

impl<T> Vocabulary<T>
where
    T: Eq + Hash,
{
    /// Returns the symbol assigned to `token`, or [`Symbol::UNKNOWN`].
    pub fn symbol(&self, token: &T) -> Symbol {
        self.symbols.get(token).copied().unwrap_or(Symbol::UNKNOWN)
    }

    /// Returns the token assigned to `symbol`.
    pub fn token(&self, symbol: Symbol) -> Option<&T> {
        if symbol.is_unknown() {
            None
        } else {
            self.tokens.get(symbol.as_usize())
        }
    }

    /// Converts tokens to symbols in input order.
    ///
    /// Tokens absent from this vocabulary are mapped to [`Symbol::UNKNOWN`].
    pub fn encode<I>(&self, tokens: I) -> Vec<Symbol>
    where
        I: IntoIterator<Item = T>,
    {
        tokens
            .into_iter()
            .map(|token| self.symbol(&token))
            .collect()
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
    use super::VocabularyBuilder;
    use crate::types::Symbol;

    #[test]
    fn build_assigns_symbols_by_descending_frequency() {
        let mut builder = VocabularyBuilder::new();

        builder.insert('a');
        builder.insert('b');
        builder.insert('b');

        let vocabulary = builder.build().unwrap();
        assert_eq!(vocabulary.symbol(&'b'), Symbol::new(0));
        assert_eq!(vocabulary.symbol(&'a'), Symbol::new(1));
    }

    #[test]
    fn equal_frequencies_preserve_first_seen_order() {
        let mut builder = VocabularyBuilder::new();
        builder.insert_all(['b', 'a']);

        let vocabulary = builder.build().unwrap();

        assert_eq!(vocabulary.symbol(&'b'), Symbol::new(0));
        assert_eq!(vocabulary.symbol(&'a'), Symbol::new(1));
    }

    #[test]
    fn mappings_are_bidirectional() {
        let mut builder = VocabularyBuilder::new();
        builder.insert("東京");
        let vocabulary = builder.build().unwrap();
        let symbol = vocabulary.symbol(&"東京");

        assert_eq!(vocabulary.symbol(&"東京"), symbol);
        assert_eq!(vocabulary.token(symbol), Some(&"東京"));
        assert_eq!(vocabulary.token(Symbol::new(1)), None);
    }

    #[test]
    fn encode_preserves_order_and_repeated_tokens() {
        let mut builder = VocabularyBuilder::new();
        builder.insert_all(['a', 'b', 'a']);
        let vocabulary = builder.build().unwrap();

        let string = vocabulary.encode(['a', 'b', 'a']);

        assert_eq!(string, [Symbol::new(0), Symbol::new(1), Symbol::new(0)]);
    }

    #[test]
    fn unknown_tokens_use_the_reserved_symbol() {
        let mut builder = VocabularyBuilder::new();
        builder.insert('a');
        let vocabulary = builder.build().unwrap();

        assert_eq!(vocabulary.symbol(&'b'), Symbol::UNKNOWN);
        assert_eq!(
            vocabulary.encode(['a', 'b']),
            [Symbol::new(0), Symbol::UNKNOWN]
        );
        assert_eq!(vocabulary.token(Symbol::UNKNOWN), None);
    }

    #[test]
    fn empty_builder_produces_an_empty_vocabulary() {
        let vocabulary = VocabularyBuilder::<char>::new().build().unwrap();

        assert!(vocabulary.is_empty());
        assert_eq!(vocabulary.len(), 0);
        assert_eq!(vocabulary.symbol(&'a'), Symbol::UNKNOWN);
    }
}
