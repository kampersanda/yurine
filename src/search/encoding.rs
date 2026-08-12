//! Query-local conversion between public tokens and internal symbols.

use std::collections::HashMap;
use std::hash::Hash;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::types::Symbol;
use crate::vocabulary::Vocabulary;

/// A query sequence represented in the vocabulary's symbol space.
///
/// Known tokens use their vocabulary symbols. Query-only tokens use symbols
/// starting at `vocabulary.len()` and exist only for this search call.
pub(super) struct EncodedQuery<T> {
    symbols: Vec<Symbol>,
    // `unknown_tokens[i]` is represented by the temporary symbol whose raw
    // value is `vocabulary.len() + i`.
    unknown_tokens: Vec<T>,
}

impl<T> EncodedQuery<T>
where
    T: Clone + Eq + Hash,
{
    pub(super) fn new(tokens: Vec<T>, vocabulary: &Vocabulary<T>) -> Result<Self> {
        let mut symbols = Vec::with_capacity(tokens.len());
        let mut unknown_tokens = Vec::new();
        let mut unknown_symbols = HashMap::new();

        for token in tokens {
            let symbol = vocabulary.symbol(&token);
            if symbol.is_unknown() {
                // Reuse one temporary symbol for repeated occurrences so a
                // query token keeps a stable identity throughout filtering
                // and verification.
                if let Some(symbol) = unknown_symbols.get(&token) {
                    symbols.push(*symbol);
                } else {
                    let index = vocabulary
                        .len()
                        .checked_add(unknown_tokens.len())
                        .ok_or(Error::SymbolOverflow)?;
                    let symbol = Symbol::from_usize(index)?;
                    unknown_symbols.insert(token.clone(), symbol);
                    unknown_tokens.push(token);
                    symbols.push(symbol);
                }
            } else {
                symbols.push(symbol);
            }
        }

        Ok(Self {
            symbols,
            unknown_tokens,
        })
    }

    pub(super) fn symbols(&self) -> &[Symbol] {
        &self.symbols
    }

    pub(super) fn costs<'a, C>(
        &'a self,
        vocabulary: &'a Vocabulary<T>,
        costs: &'a C,
    ) -> EncodedCosts<'a, T, C> {
        EncodedCosts {
            vocabulary,
            unknown_tokens: &self.unknown_tokens,
            costs,
        }
    }
}

/// Adapts public token costs to the symbol costs used by search internals.
pub(super) struct EncodedCosts<'a, T, C> {
    vocabulary: &'a Vocabulary<T>,
    unknown_tokens: &'a [T],
    costs: &'a C,
}

impl<T, C> EncodedCosts<'_, T, C>
where
    T: Eq + Hash,
{
    fn token(&self, symbol: &Symbol) -> &T {
        // SearchEngine validates that every string symbol belongs to the
        // vocabulary. A symbol outside it is therefore query-local, and its
        // offset identifies the corresponding entry in `unknown_tokens`.
        self.vocabulary
            .token(*symbol)
            .unwrap_or_else(|| &self.unknown_tokens[symbol.as_usize() - self.vocabulary.len()])
    }
}

impl<T, C> EditCosts<Symbol> for EncodedCosts<'_, T, C>
where
    T: Eq + Hash,
    C: EditCosts<T>,
{
    fn substitution(&self, from: &Symbol, to: &Symbol) -> Cost {
        self.costs.substitution(self.token(from), self.token(to))
    }

    fn deletion(&self, symbol: &Symbol) -> Cost {
        self.costs.deletion(self.token(symbol))
    }

    fn insertion(&self, symbol: &Symbol) -> Cost {
        self.costs.insertion(self.token(symbol))
    }
}

#[cfg(test)]
mod tests {
    use super::EncodedQuery;
    use crate::costs::{Cost, EditCosts};
    use crate::types::Symbol;
    use crate::vocabulary::VocabularyBuilder;

    struct CharacterCosts;

    impl EditCosts<char> for CharacterCosts {
        fn substitution(&self, from: &char, to: &char) -> Cost {
            if from == to {
                Cost::ZERO
            } else {
                Cost::new_const(0.25)
            }
        }

        fn deletion(&self, _token: &char) -> Cost {
            Cost::new_const(0.5)
        }

        fn insertion(&self, _token: &char) -> Cost {
            Cost::new_const(0.75)
        }
    }

    #[test]
    fn query_only_tokens_receive_stable_distinct_symbols() {
        let mut builder = VocabularyBuilder::new();
        builder.insert('a');
        let vocabulary = builder.build().unwrap();

        let query = EncodedQuery::new(vec!['x', 'y', 'x'], &vocabulary).unwrap();

        assert_eq!(
            query.symbols(),
            [Symbol::new(1), Symbol::new(2), Symbol::new(1)]
        );
    }

    #[test]
    fn encoded_costs_delegate_known_and_query_only_symbols_to_token_costs() {
        let mut builder = VocabularyBuilder::new();
        builder.insert('a');
        let vocabulary = builder.build().unwrap();
        let query = EncodedQuery::new(vec!['a', 'x'], &vocabulary).unwrap();
        let encoded_costs = query.costs(&vocabulary, &CharacterCosts);
        let known = Symbol::new(0);
        let query_only = Symbol::new(1);

        assert_eq!(encoded_costs.substitution(&known, &known), Cost::ZERO);
        assert_eq!(
            encoded_costs.substitution(&query_only, &known),
            Cost::new_const(0.25)
        );
        assert_eq!(encoded_costs.deletion(&query_only), Cost::new_const(0.5));
        assert_eq!(encoded_costs.insertion(&known), Cost::new_const(0.75));
    }
}
