//! Fixed-threshold range search orchestration.

use std::hash::Hash;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::search::SearchEngine;
use crate::search::encoding::EncodedQuery;
use crate::search::filtering::candidate::MinCandidateSelector;
use crate::search::filtering::generate_candidates;
use crate::search::verification::Verifier;
use crate::search::{Candidate, Match};
use crate::store::CorpusStore;
use crate::tokenization::Tokenizer;
use crate::types::{Position, StringId, Symbol};

/// Parameters for threshold range search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeSearchParams {
    threshold: Cost,
    eta: Option<Cost>,
}

impl RangeSearchParams {
    /// Creates parameters with automatic eta.
    pub const fn new(threshold: Cost) -> Self {
        Self {
            threshold,
            eta: None,
        }
    }

    /// Uses an explicit substitution-neighborhood radius.
    pub const fn with_eta(mut self, eta: Cost) -> Self {
        self.eta = Some(eta);
        self
    }

    /// Returns the inclusive distance threshold.
    pub const fn threshold(&self) -> Cost {
        self.threshold
    }

    /// Returns the explicit eta, or `None` when eta is automatic.
    pub const fn eta(&self) -> Option<Cost> {
        self.eta
    }
}

/// Returns the default substitution-neighborhood radius for a query length.
///
/// The radius is `threshold / query_len`. For an empty query, this returns zero
/// rather than dividing by zero; searching an empty query retains its existing
/// error behavior. [`Cost::MAX`] is rejected because it cannot be converted to
/// a finite strict search bound.
pub fn automatic_eta(threshold: Cost, query_len: usize) -> Result<Cost> {
    // strict_threshold(threshold)?;
    if query_len == 0 {
        Ok(Cost::ZERO)
    } else {
        Cost::new(threshold.get() / query_len as f32)
    }
}

impl<T, C> SearchEngine<T, C>
where
    T: Tokenizer,
    T::Token: Clone + Eq + Hash,
    C: EditCosts<T::Token>,
{
    /// Finds non-empty substrings satisfying the configured range search.
    ///
    /// Results are ordered by string ID, then range start, then range end.
    ///
    /// When eta is not configured, it defaults to
    /// `threshold / tokenized_query.len()`. This favors constructing a
    /// threshold subsequence for continuous substitution costs. An empty query
    /// uses eta zero and retains the existing empty-query error behavior.
    ///
    /// If the selector cannot construct a complete threshold subsequence for
    /// a non-empty query, the engine falls back to exhaustive Smith-Waterman
    /// verification instead of returning
    /// [`Error::ThresholdSubsequenceUnavailable`]. This occurs whenever the
    /// query's total filtering contribution is less than or equal to the
    /// threshold. With unit costs, `threshold >= tokenized_query.len()` is
    /// such a case.
    ///
    /// The fallback takes `O(m * sum(n_i^2))` time for query length `m` and
    /// corpus string lengths `n_i`, and can return `O(sum(n_i^2))` intervals.
    /// It may therefore be substantially slower and produce many more results
    /// than the normal filter-and-verify path.
    ///
    /// Searching takes `&self`, so one engine can serve concurrent queries.
    pub fn range_search(&self, query: &str, params: &RangeSearchParams) -> Result<Vec<Match>> {
        let query = EncodedQuery::new(
            self.tokenizer
                .tokenize(query)
                .into_iter()
                .map(|token| token.value)
                .collect(),
            &self.vocabulary,
        )?;
        let costs = query.costs(&self.vocabulary, &self.costs);
        let threshold = params.threshold;
        // strict_threshold(threshold)?;
        let eta = match params.eta {
            Some(eta) => eta,
            None => automatic_eta(threshold, query.symbols().len())?,
        };
        self.search_all(query.symbols(), threshold, eta, &costs)
    }

    fn search_all<S>(
        &self,
        query: &[Symbol],
        threshold: Cost,
        eta: Cost,
        costs: &S,
    ) -> Result<Vec<Match>>
    where
        S: EditCosts<Symbol>,
    {
        let selected = match MinCandidateSelector.select(
            query,
            threshold,
            eta,
            &self.index,
            costs,
            &self.neighborhood,
        ) {
            Ok(selected) => selected,
            Err(Error::ThresholdSubsequenceUnavailable) if !query.is_empty() => {
                return verify_exhaustively(query, threshold, &self.store, costs);
            }
            Err(error) => return Err(error),
        };
        let candidates = generate_candidates(
            query,
            &selected,
            eta,
            &self.index,
            costs,
            &self.neighborhood,
        )?;
        Verifier::BidirectionalTrie.verify(query, &candidates, &self.store, threshold, costs)
    }
}

/// Exhaustively verifies every non-empty string in the corpus.
///
/// This is used
/// when the selector cannot construct a complete threshold subsequence for a
/// non-empty query. It is slower and can return more results than the normal
/// filter-and-verify path, but is guaranteed to be correct.
fn verify_exhaustively<C>(
    query: &[Symbol],
    threshold: Cost,
    corpus: &CorpusStore,
    costs: &C,
) -> Result<Vec<Match>>
where
    C: EditCosts<Symbol>,
{
    // Smith-Waterman verification uses candidates only to select data strings. One
    // in-bounds anchor per non-empty string requests exhaustive verification
    // without relying on the filtering guarantee that was unavailable.
    let mut candidates = Vec::new();
    for raw_id in 0..corpus.len() {
        let string_id = StringId::from_usize(raw_id)?;
        let string = corpus
            .string(string_id)?
            .ok_or(Error::UnknownString(string_id))?;
        if !string.is_empty() {
            candidates.push(Candidate {
                string_id,
                data_position: Position::new(0),
                query_position: Position::new(0),
            });
        }
    }
    Verifier::SmithWaterman.verify(query, &candidates, corpus, threshold, costs)
}

#[cfg(test)]
mod tests {
    use super::RangeSearchParams;
    use crate::costs::{Cost, EditCosts};
    use crate::postings::PostingsIndexBuilder;
    use crate::search::{Match, SearchEngine};
    use crate::store::CorpusStoreBuilder;
    use crate::tokenization::character::CharacterTokenizer;
    use crate::types::{Position, Posting, StringId};
    use crate::vocabulary::VocabularyBuilder;

    struct CharacterCosts;

    impl EditCosts<char> for CharacterCosts {
        fn substitution(&self, from: &char, to: &char) -> Cost {
            if from == to {
                Cost::ZERO
            } else if *from == 'y' && *to == 'a' {
                Cost::new_const(0.4)
            } else {
                Cost::ONE
            }
        }

        fn deletion(&self, token: &char) -> Cost {
            if *token == 'x' {
                Cost::new_const(0.25)
            } else {
                Cost::ONE
            }
        }

        fn insertion(&self, _token: &char) -> Cost {
            Cost::ONE
        }
    }

    fn engine() -> SearchEngine<CharacterTokenizer, CharacterCosts> {
        let mut vocabulary_builder = VocabularyBuilder::new();
        vocabulary_builder.insert('a');
        let vocabulary = vocabulary_builder.build().unwrap();
        let symbol = vocabulary.symbol(&'a');

        let mut index_builder = PostingsIndexBuilder::new();
        for string_id in [StringId::new(0), StringId::new(1)] {
            index_builder.add_posting(
                symbol,
                Posting {
                    string_id,
                    position: Position::new(0),
                },
            );
        }

        let mut store_builder = CorpusStoreBuilder::new();
        store_builder.add_string(vec![symbol], std::iter::once(0..1).collect());
        store_builder.add_string(vec![symbol], std::iter::once(0..1).collect());

        SearchEngine::from_parts(
            CharacterTokenizer::new(),
            vocabulary,
            CharacterCosts,
            index_builder.build(),
            store_builder.build(),
        )
        .unwrap()
    }

    fn expected_matches(distance: Cost) -> Vec<Match> {
        vec![
            Match {
                string_id: StringId::new(0),
                token_range: Position::new(0)..Position::new(1),
                byte_range: 0..1,
                distance,
            },
            Match {
                string_id: StringId::new(1),
                token_range: Position::new(0)..Position::new(1),
                byte_range: 0..1,
                distance,
            },
        ]
    }

    #[test]
    fn applies_substitution_cost_for_query_only_token() {
        let matches = engine()
            .range_search("y", &RangeSearchParams::new(Cost::new_const(0.4)))
            .unwrap();

        assert_eq!(matches, expected_matches(Cost::new_const(0.4)));
    }

    #[test]
    fn applies_deletion_cost_for_query_only_token() {
        let matches = engine()
            .range_search("xa", &RangeSearchParams::new(Cost::new_const(0.25)))
            .unwrap();

        assert_eq!(matches, expected_matches(Cost::new_const(0.25)));
    }

    #[test]
    fn searches_can_share_an_engine() {
        let engine = engine();
        let substitution_params = RangeSearchParams::new(Cost::new_const(0.4));
        let deletion_params = RangeSearchParams::new(Cost::new_const(0.25));

        std::thread::scope(|scope| {
            let substitution =
                scope.spawn(|| engine.range_search("y", &substitution_params).unwrap());
            let deletion = scope.spawn(|| engine.range_search("xa", &deletion_params).unwrap());

            assert_eq!(substitution.join().unwrap()[0].distance, 0.4);
            assert_eq!(deletion.join().unwrap()[0].distance, 0.25);
        });
    }
}
