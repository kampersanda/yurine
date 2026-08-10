//! Query-local conversion between public tokens and internal symbols.

use std::collections::HashMap;
use std::hash::Hash;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::types::Symbol;
use crate::vocabulary::Vocabulary;

/// A query represented in the corpus symbol space.
///
/// Known tokens use their vocabulary symbols. Query-only tokens use symbols
/// starting at `vocabulary.len()` and exist only for this search call.
pub(super) struct EncodedQuery<Token> {
    symbols: Vec<Symbol>,
    // `unknown_tokens[i]` is represented by the temporary symbol whose raw
    // value is `vocabulary.len() + i`.
    unknown_tokens: Vec<Token>,
}

impl<Token> EncodedQuery<Token>
where
    Token: Clone + Eq + Hash,
{
    pub(super) fn new(tokens: Vec<Token>, vocabulary: &Vocabulary<Token>) -> Result<Self> {
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

    pub(super) fn costs<'a, Costs>(
        &'a self,
        vocabulary: &'a Vocabulary<Token>,
        costs: &'a Costs,
    ) -> EncodedCosts<'a, Token, Costs> {
        EncodedCosts {
            vocabulary,
            unknown_tokens: &self.unknown_tokens,
            costs,
        }
    }
}

/// Adapts public token costs to the symbol costs used by search internals.
pub(super) struct EncodedCosts<'a, Token, Costs> {
    vocabulary: &'a Vocabulary<Token>,
    unknown_tokens: &'a [Token],
    costs: &'a Costs,
}

impl<Token, Costs> EncodedCosts<'_, Token, Costs>
where
    Token: Eq + Hash,
{
    fn token(&self, symbol: &Symbol) -> &Token {
        // SearchEngine validates that every corpus symbol belongs to the
        // vocabulary. A symbol outside it is therefore query-local, and its
        // offset identifies the corresponding entry in `unknown_tokens`.
        self.vocabulary
            .token(*symbol)
            .unwrap_or_else(|| &self.unknown_tokens[symbol.as_usize() - self.vocabulary.len()])
    }
}

impl<Token, Costs> EditCosts<Symbol> for EncodedCosts<'_, Token, Costs>
where
    Token: Eq + Hash,
    Costs: EditCosts<Token>,
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
    use crate::types::Symbol;
    use crate::vocabulary::VocabularyBuilder;

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
}
