//! Postings index mapping symbols to their occurrences in the corpus.

use crate::errors::{Error, Result};
use crate::persistence::storage::Storage;
use crate::types::{Posting, Symbol};

/// Postings index mapping symbols to their occurrences in the corpus.
pub(crate) struct PostingsIndex {
    postings: Storage<Posting>,
    posting_offsets: Storage<u64>,
}

impl PostingsIndex {
    /// Returns indexed occurrences in `(SequenceId, Position)` order.
    ///
    /// The iterator does not emit duplicates.
    pub(crate) fn postings(&self, symbol: Symbol) -> impl Iterator<Item = Posting> + '_ {
        self.posting_slice(symbol).iter().copied()
    }

    /// Returns the total frequency of `symbol` in the corpus.
    pub(crate) fn frequency(&self, symbol: Symbol) -> usize {
        self.posting_slice(symbol).len()
    }

    fn posting_slice(&self, symbol: Symbol) -> &[Posting] {
        if symbol.is_unknown() {
            return &[];
        }
        let index = symbol.as_usize();
        let (Some(&start), Some(&end)) = (
            self.posting_offsets.get(index),
            self.posting_offsets.get(index + 1),
        ) else {
            return &[];
        };
        &self.postings[start as usize..end as usize]
    }
}

/// Builds a postings index from symbols and their occurrences.
#[derive(Debug)]
pub(crate) struct PostingsIndexBuilder {
    symbol_count: usize,
    postings: Vec<(Symbol, Posting)>,
}

impl PostingsIndexBuilder {
    /// Creates a builder for a vocabulary containing `symbol_count` symbols.
    pub(crate) fn new(symbol_count: usize) -> Self {
        Self {
            symbol_count,
            postings: Vec::new(),
        }
    }

    /// Adds a posting for the given symbol.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownStringSymbol`] if `symbol` is not present in
    /// the configured vocabulary.
    pub(crate) fn add_posting(&mut self, symbol: Symbol, posting: Posting) -> Result<()> {
        if symbol.is_unknown() || symbol.as_usize() >= self.symbol_count {
            return Err(Error::UnknownStringSymbol(symbol.get()));
        }
        self.postings.push((symbol, posting));
        Ok(())
    }

    /// Builds an index whose postings are ordered and contain no duplicates.
    pub(crate) fn build(mut self) -> PostingsIndex {
        self.postings.sort_unstable_by_key(|(symbol, posting)| {
            (*symbol, posting.string_id, posting.position)
        });
        self.postings.dedup();

        let mut posting_offsets = vec![0u64; self.symbol_count + 1];
        for (symbol, _) in &self.postings {
            posting_offsets[symbol.as_usize() + 1] += 1;
        }
        for index in 0..self.symbol_count {
            posting_offsets[index + 1] += posting_offsets[index];
        }
        let mut postings: Vec<_> = self
            .postings
            .into_iter()
            .map(|(_, posting)| posting)
            .collect();
        // Collection can reuse the larger `(Symbol, Posting)` allocation.
        postings.shrink_to_fit();

        PostingsIndex {
            postings: Storage::Owned(postings.into_boxed_slice()),
            posting_offsets: Storage::Owned(posting_offsets.into_boxed_slice()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PostingsIndexBuilder;
    use crate::errors::Error;
    use crate::types::{Position, Posting, SequenceId, Symbol};

    #[test]
    fn build_orders_and_deduplicates_postings() {
        let first = Posting {
            string_id: SequenceId::new(0),
            position: Position::new(2),
        };
        let second = Posting {
            string_id: SequenceId::new(1),
            position: Position::new(0),
        };
        let third = Posting {
            string_id: SequenceId::new(1),
            position: Position::new(1),
        };

        let mut builder = PostingsIndexBuilder::new(1);
        let symbol = Symbol::new(0);
        builder.add_posting(symbol, third).unwrap();
        builder.add_posting(symbol, first).unwrap();
        builder.add_posting(symbol, second).unwrap();
        builder.add_posting(symbol, first).unwrap();

        let index = builder.build();

        assert_eq!(
            index.postings(symbol).collect::<Vec<_>>(),
            [first, second, third]
        );
        assert_eq!(index.frequency(symbol), 3);
    }

    #[test]
    fn flat_offsets_represent_absent_and_trailing_symbols() {
        let posting = Posting {
            string_id: SequenceId::new(0),
            position: Position::new(0),
        };
        let mut builder = PostingsIndexBuilder::new(3);
        builder.add_posting(Symbol::new(1), posting).unwrap();

        let index = builder.build();

        assert_eq!(index.postings.as_slice(), [posting]);
        assert_eq!(index.posting_offsets.as_slice(), [0, 0, 1, 1]);
        assert_eq!(index.frequency(Symbol::new(0)), 0);
        assert_eq!(index.frequency(Symbol::new(1)), 1);
        assert_eq!(index.frequency(Symbol::new(2)), 0);
        assert_eq!(index.frequency(Symbol::new(3)), 0);
        assert_eq!(index.frequency(Symbol::UNKNOWN), 0);
    }

    #[test]
    fn groups_out_of_order_postings_by_symbol() {
        let for_first = Posting {
            string_id: SequenceId::new(0),
            position: Position::new(1),
        };
        let for_second = Posting {
            string_id: SequenceId::new(1),
            position: Position::new(0),
        };
        let later_for_third = Posting {
            string_id: SequenceId::new(2),
            position: Position::new(2),
        };
        let earlier_for_third = Posting {
            string_id: SequenceId::new(0),
            position: Position::new(3),
        };
        let mut builder = PostingsIndexBuilder::new(3);

        builder
            .add_posting(Symbol::new(2), later_for_third)
            .unwrap();
        builder.add_posting(Symbol::new(0), for_first).unwrap();
        builder
            .add_posting(Symbol::new(2), earlier_for_third)
            .unwrap();
        builder.add_posting(Symbol::new(1), for_second).unwrap();

        let index = builder.build();

        assert_eq!(index.posting_offsets.as_slice(), [0, 1, 2, 4]);
        assert_eq!(
            index.postings(Symbol::new(0)).collect::<Vec<_>>(),
            [for_first]
        );
        assert_eq!(
            index.postings(Symbol::new(1)).collect::<Vec<_>>(),
            [for_second]
        );
        assert_eq!(
            index.postings(Symbol::new(2)).collect::<Vec<_>>(),
            [earlier_for_third, later_for_third]
        );
    }

    #[test]
    fn rejects_symbols_outside_the_vocabulary() {
        let posting = Posting {
            string_id: SequenceId::new(0),
            position: Position::new(0),
        };
        let mut builder = PostingsIndexBuilder::new(1);

        assert_eq!(
            builder.add_posting(Symbol::new(1), posting),
            Err(Error::UnknownStringSymbol(1))
        );
        assert_eq!(
            builder.add_posting(Symbol::UNKNOWN, posting),
            Err(Error::UnknownStringSymbol(u32::MAX))
        );
    }

    #[test]
    fn empty_vocabulary_has_no_postings() {
        let index = PostingsIndexBuilder::new(0).build();

        assert_eq!(index.posting_offsets.as_slice(), [0]);
        assert_eq!(index.postings(Symbol::new(0)).collect::<Vec<_>>(), []);
        assert_eq!(index.frequency(Symbol::new(0)), 0);
    }
}
