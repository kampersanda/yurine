//! Postings index mapping symbols to their occurrences in the corpus.

use std::collections::HashMap;

use crate::types::{Posting, Symbol};

/// Postings index mapping symbols to their occurrences in the corpus.
pub struct PostingsIndex {
    postings: HashMap<Symbol, Vec<Posting>>,
}

impl PostingsIndex {
    /// Returns indexed occurrences in `(StringId, Position)` order.
    ///
    /// The iterator does not emit duplicates.
    pub fn postings(&self, symbol: Symbol) -> impl Iterator<Item = Posting> + '_ {
        self.postings.get(&symbol).into_iter().flatten().copied()
    }

    /// Returns the total frequency of `symbol` in the corpus.
    pub fn frequency(&self, symbol: Symbol) -> usize {
        self.postings
            .get(&symbol)
            .map_or(0, |postings| postings.len())
    }
}

/// Builds a postings index from symbols and their occurrences.
#[derive(Debug, Default)]
pub struct PostingsIndexBuilder {
    postings: HashMap<Symbol, Vec<Posting>>,
}

impl PostingsIndexBuilder {
    /// Creates a new postings index builder.
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
        }
    }

    /// Adds a posting for the given symbol.
    pub fn add_posting(&mut self, symbol: Symbol, posting: Posting) {
        self.postings.entry(symbol).or_default().push(posting);
    }

    /// Builds an index whose postings are ordered and contain no duplicates.
    pub fn build(mut self) -> PostingsIndex {
        for postings in self.postings.values_mut() {
            postings.sort_unstable_by_key(|posting| (posting.string_id, posting.position));
            postings.dedup();
        }
        PostingsIndex {
            postings: self.postings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PostingsIndexBuilder;
    use crate::types::{Position, Posting, StringId, Symbol};

    #[test]
    fn build_orders_and_deduplicates_postings() {
        let first = Posting {
            string_id: StringId::new(0),
            position: Position::new(2),
        };
        let second = Posting {
            string_id: StringId::new(1),
            position: Position::new(0),
        };
        let third = Posting {
            string_id: StringId::new(1),
            position: Position::new(1),
        };

        let mut builder = PostingsIndexBuilder::new();
        let symbol = Symbol::new(0);
        builder.add_posting(symbol, third);
        builder.add_posting(symbol, first);
        builder.add_posting(symbol, second);
        builder.add_posting(symbol, first);

        let index = builder.build();

        assert_eq!(
            index.postings(symbol).collect::<Vec<_>>(),
            [first, second, third]
        );
        assert_eq!(index.frequency(symbol), 3);
    }

    #[test]
    fn absent_symbol_has_no_postings() {
        let index = PostingsIndexBuilder::new().build();
        let absent = Symbol::new(7);

        assert_eq!(index.postings(absent).collect::<Vec<_>>(), []);
        assert_eq!(index.frequency(absent), 0);
    }
}
