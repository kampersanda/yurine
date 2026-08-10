mod builder;
mod encoding;
mod filtering;
pub mod range_search;
mod verification;

use std::hash::Hash;
use std::ops::Range;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::postings::PostingsIndex;
use crate::store::CorpusStore;
use crate::tokenization::Tokenizer;
use crate::types::{Position, StringId};
use crate::vocabulary::Vocabulary;

use filtering::neighborhood::SubstitutionNeighborhood;

pub use builder::SearchEngineBuilder;

/// A candidate match of a query in a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Candidate {
    string_id: StringId,
    data_position: Position,
    query_position: Position,
}

/// A verified substring satisfying the inclusive distance threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// The data string containing the match.
    pub string_id: StringId,
    /// The matched zero-based, end-exclusive token range.
    pub range: Range<Position>,
    /// The weighted edit distance from the query to the substring.
    pub distance: Cost,
}

/// Coordinates threshold-subsequence filtering and exact verification.
///
/// Create an engine with [`SearchEngineBuilder`].
pub struct SearchEngine<T, Costs>
where
    T: Tokenizer,
{
    tokenizer: T,
    vocabulary: Vocabulary<T::Token>,
    costs: Costs,
    index: PostingsIndex,
    store: CorpusStore,
    neighborhood: SubstitutionNeighborhood,
}

impl<T, Costs> SearchEngine<T, Costs>
where
    T: Tokenizer,
    T::Token: Clone + Eq + Hash,
    Costs: EditCosts<T::Token>,
{
    pub(crate) fn from_parts(
        tokenizer: T,
        vocabulary: Vocabulary<T::Token>,
        costs: Costs,
        index: PostingsIndex,
        store: CorpusStore,
    ) -> Result<Self> {
        for symbol in store.alphabet() {
            if vocabulary.token(*symbol).is_none() {
                return Err(Error::UnknownCorpusSymbol(*symbol));
            }
        }
        let neighborhood = SubstitutionNeighborhood::new(store.alphabet().iter().copied())?;
        Ok(Self {
            tokenizer,
            vocabulary,
            costs,
            index,
            store,
            neighborhood,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SearchEngine;
    use crate::costs::levenshtein::LevenshteinCosts;
    use crate::errors::Error;
    use crate::postings::PostingsIndexBuilder;
    use crate::store::CorpusStoreBuilder;
    use crate::tokenization::character::CharacterTokenizer;
    use crate::types::Symbol;
    use crate::vocabulary::VocabularyBuilder;

    #[test]
    fn rejects_corpus_symbol_absent_from_vocabulary() {
        let mut vocabulary_builder = VocabularyBuilder::new();
        vocabulary_builder.insert('a');
        let vocabulary = vocabulary_builder.build().unwrap();
        let unknown_symbol = Symbol::new(1);

        let mut store_builder = CorpusStoreBuilder::new();
        store_builder.add_string(vec![unknown_symbol]);

        let result = SearchEngine::from_parts(
            CharacterTokenizer::new(),
            vocabulary,
            LevenshteinCosts::new(),
            PostingsIndexBuilder::new().build(),
            store_builder.build(),
        );

        assert!(matches!(
            result,
            Err(Error::UnknownCorpusSymbol(symbol)) if symbol == unknown_symbol
        ));
    }
}
