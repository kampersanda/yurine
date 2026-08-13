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
use crate::types::{Position, SequenceId, Symbol};

/// Parameters for threshold range search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeSearchParams {
    threshold: Cost,
    eta: Option<Cost>,
}

/// Measurements from the filtering phase of one range search.
///
/// Counts are exposed so benchmark tooling can compare candidate generation
/// across implementations without making elapsed time a test assertion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RangeSearchMetrics {
    /// Whether filtering was unavailable and exhaustive verification was used.
    pub used_exhaustive_verification: bool,
    /// Number of query positions chosen for candidate generation.
    ///
    /// This is zero when exhaustive verification was used.
    pub selected_query_positions: usize,
    /// Number of candidate anchors generated from postings.
    ///
    /// This is zero when exhaustive verification was used.
    pub generated_candidates: usize,
}

/// Executes range searches against one engine with a fixed edit-cost policy.
///
/// Create a searcher with [`SearchEngine::range_searcher`]. The searcher owns
/// its edit costs and borrows the engine, so creating one does not copy or
/// reference-count the index.
pub struct RangeSearcher<'a, T, C> {
    engine: &'a SearchEngine<T>,
    costs: C,
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
/// The radius is `threshold / query_sequence_len`. For an empty query sequence,
/// this returns zero rather than dividing by zero; searching an empty query
/// sequence retains its existing error behavior. [`Cost::MAX`] is rejected
/// because it cannot be converted to a finite strict search bound.
pub fn automatic_eta(threshold: Cost, query_sequence_len: usize) -> Result<Cost> {
    // strict_threshold(threshold)?;
    if query_sequence_len == 0 {
        Ok(Cost::ZERO)
    } else {
        Cost::new(threshold.get() / query_sequence_len as f32)
    }
}

impl<T> SearchEngine<T> {
    /// Creates a range searcher with the supplied edit-cost policy.
    pub fn range_searcher<C>(&self, costs: C) -> RangeSearcher<'_, T, C> {
        RangeSearcher {
            engine: self,
            costs,
        }
    }
}

impl<T, C> RangeSearcher<'_, T, C>
where
    T: Clone + Eq + Hash,
    C: EditCosts<T>,
{
    /// Finds non-empty data segments satisfying the configured range search.
    ///
    /// Results are ordered by data sequence ID, then token-range start, then
    /// token-range end.
    ///
    /// When eta is not configured, it defaults to
    /// `threshold / query_sequence.len()`. This favors constructing a
    /// threshold subsequence for continuous substitution costs. An empty query
    /// sequence uses eta zero and retains the existing empty-query error
    /// behavior.
    ///
    /// If the selector cannot construct a complete threshold subsequence for
    /// a non-empty query sequence, the engine falls back to exhaustive
    /// Smith-Waterman verification instead of returning
    /// [`Error::ThresholdSubsequenceUnavailable`]. This occurs whenever the
    /// query sequence's total filtering contribution is less than or equal to
    /// the threshold. With unit costs,
    /// `threshold >= query_sequence.len()` is
    /// such a case.
    ///
    /// The fallback takes `O(m * sum(n_i^2))` time for query-sequence length
    /// `m` and
    /// data string lengths `n_i`, and can return `O(sum(n_i^2))` data
    /// segments.
    /// It may therefore be substantially slower and produce many more results
    /// than the normal filter-and-verify path.
    ///
    /// Searching takes `&self`, so one searcher can serve concurrent queries.
    pub fn search(&self, query_sequence: &[T], params: &RangeSearchParams) -> Result<Vec<Match>> {
        self.search_with_metrics(query_sequence, params)
            .map(|(matches, _)| matches)
    }

    /// Finds matches and returns filtering measurements for reproducible
    /// performance comparisons.
    pub fn search_with_metrics(
        &self,
        query_sequence: &[T],
        params: &RangeSearchParams,
    ) -> Result<(Vec<Match>, RangeSearchMetrics)> {
        let encoded_query = EncodedQuery::new(query_sequence.to_vec(), &self.engine.vocabulary)?;
        let costs = encoded_query.costs(&self.engine.vocabulary, &self.costs);
        let threshold = params.threshold;
        // strict_threshold(threshold)?;
        let eta = match params.eta {
            Some(eta) => eta,
            None => automatic_eta(threshold, encoded_query.string().len())?,
        };
        self.search_query_string(encoded_query.string(), threshold, eta, &costs)
    }

    fn search_query_string<S>(
        &self,
        query_string: &[Symbol],
        threshold: Cost,
        eta: Cost,
        costs: &S,
    ) -> Result<(Vec<Match>, RangeSearchMetrics)>
    where
        S: EditCosts<Symbol>,
    {
        let selected = match MinCandidateSelector.select(
            query_string,
            threshold,
            eta,
            &self.engine.index,
            costs,
            &self.engine.neighborhood,
        ) {
            Ok(selected) => selected,
            Err(Error::ThresholdSubsequenceUnavailable) if !query_string.is_empty() => {
                let matches =
                    verify_exhaustively(query_string, threshold, &self.engine.store, costs)?;
                return Ok((
                    matches,
                    RangeSearchMetrics {
                        used_exhaustive_verification: true,
                        ..RangeSearchMetrics::default()
                    },
                ));
            }
            Err(error) => return Err(error),
        };
        let candidates = generate_candidates(
            query_string,
            &selected,
            eta,
            &self.engine.index,
            costs,
            &self.engine.neighborhood,
        )?;
        let matches = Verifier::BidirectionalTrie.verify(
            query_string,
            &candidates,
            &self.engine.store,
            threshold,
            costs,
        )?;
        Ok((
            matches,
            RangeSearchMetrics {
                used_exhaustive_verification: false,
                selected_query_positions: selected.len(),
                generated_candidates: candidates.len(),
            },
        ))
    }
}

/// Exhaustively verifies every non-empty string in the corpus.
///
/// This is used
/// when the selector cannot construct a complete threshold subsequence for a
/// non-empty query. It is slower and can return more results than the normal
/// filter-and-verify path, but is guaranteed to be correct.
fn verify_exhaustively<C>(
    query_string: &[Symbol],
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
        let string_id = SequenceId::from_usize(raw_id)?;
        let string = corpus
            .string(string_id)?
            .ok_or(Error::UnknownString(string_id))?;
        if !string.is_empty() {
            candidates.push(Candidate {
                string_id,
                string_position: Position::new(0),
                query_position: Position::new(0),
            });
        }
    }
    Verifier::SmithWaterman.verify(query_string, &candidates, corpus, threshold, costs)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{MinCandidateSelector, RangeSearchParams, automatic_eta, verify_exhaustively};
    use crate::costs::embedding::{CosineEmbeddingCosts, EmbeddingStore};
    use crate::costs::levenshtein::LevenshteinCosts;
    use crate::costs::{Cost, EditCosts};
    use crate::errors::Error;
    use crate::postings::PostingsIndexBuilder;
    use crate::search::SearchEngineBuilder;
    use crate::search::encoding::EncodedQuery;
    use crate::search::{Match, SearchEngine};
    use crate::store::CorpusStoreBuilder;
    use crate::types::{Position, Posting, SequenceId};
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

    fn engine() -> SearchEngine<char> {
        let mut vocabulary_builder = VocabularyBuilder::new();
        vocabulary_builder.insert('a');
        let vocabulary = vocabulary_builder.build().unwrap();
        let symbol = vocabulary.symbol(&'a');

        let mut index_builder = PostingsIndexBuilder::new(vocabulary.len());
        for string_id in [SequenceId::new(0), SequenceId::new(1)] {
            index_builder
                .add_posting(
                    symbol,
                    Posting {
                        string_id,
                        position: Position::new(0),
                    },
                )
                .unwrap();
        }

        let mut store_builder = CorpusStoreBuilder::new();
        store_builder.add_string(vec![symbol]);
        store_builder.add_string(vec![symbol]);

        SearchEngine::from_parts(vocabulary, index_builder.build(), store_builder.build()).unwrap()
    }

    fn expected_matches(distance: Cost) -> Vec<Match> {
        vec![
            Match {
                sequence_id: SequenceId::new(0),
                token_range: Position::new(0)..Position::new(1),
                distance,
            },
            Match {
                sequence_id: SequenceId::new(1),
                token_range: Position::new(0)..Position::new(1),
                distance,
            },
        ]
    }

    #[test]
    fn applies_substitution_cost_for_query_only_token() {
        let engine = engine();
        let matches = engine
            .range_searcher(CharacterCosts)
            .search(&['y'], &RangeSearchParams::new(Cost::new_const(0.4)))
            .unwrap();

        assert_eq!(matches, expected_matches(Cost::new_const(0.4)));
    }

    #[test]
    fn applies_deletion_cost_for_query_only_token() {
        let engine = engine();
        let matches = engine
            .range_searcher(CharacterCosts)
            .search(&['x', 'a'], &RangeSearchParams::new(Cost::new_const(0.25)))
            .unwrap();

        assert_eq!(matches, expected_matches(Cost::new_const(0.25)));
    }

    #[test]
    fn searcher_can_serve_concurrent_queries() {
        let engine = engine();
        let searcher = engine.range_searcher(CharacterCosts);
        let substitution_params = RangeSearchParams::new(Cost::new_const(0.4));
        let deletion_params = RangeSearchParams::new(Cost::new_const(0.25));

        std::thread::scope(|scope| {
            let substitution =
                scope.spawn(|| searcher.search(&['y'], &substitution_params).unwrap());
            let deletion = scope.spawn(|| searcher.search(&['x', 'a'], &deletion_params).unwrap());

            assert_eq!(substitution.join().unwrap()[0].distance, 0.4);
            assert_eq!(deletion.join().unwrap()[0].distance, 0.25);
        });
    }

    #[test]
    fn one_engine_accepts_different_cost_policies() {
        let engine = engine();
        let params = RangeSearchParams::new(Cost::new_const(0.4));
        let character = engine.range_searcher(CharacterCosts);
        let levenshtein = engine.range_searcher(LevenshteinCosts::new());

        assert_eq!(
            character.search(&['y'], &params).unwrap(),
            expected_matches(Cost::new_const(0.4))
        );
        assert!(levenshtein.search(&['y'], &params).unwrap().is_empty());
    }

    #[test]
    fn parameters_expose_threshold_and_optional_eta() {
        let threshold = Cost::new_const(0.75);
        let eta = Cost::new_const(0.25);

        let automatic = RangeSearchParams::new(threshold);
        let explicit = automatic.with_eta(eta);

        assert_eq!(automatic.threshold(), threshold);
        assert_eq!(automatic.eta(), None);
        assert_eq!(explicit.threshold(), threshold);
        assert_eq!(explicit.eta(), Some(eta));
    }

    #[test]
    fn automatic_eta_divides_threshold_by_query_sequence_length() {
        assert_eq!(automatic_eta(Cost::new_const(0.75), 3).unwrap(), 0.25);
        assert_eq!(automatic_eta(Cost::ONE, 0).unwrap(), Cost::ZERO);
    }

    #[test]
    fn falls_back_to_exhaustive_search_when_no_threshold_subsequence_exists() {
        let engine = engine();
        let matches = engine
            .range_searcher(CharacterCosts)
            .search(&['x'], &RangeSearchParams::new(Cost::ONE))
            .unwrap();

        assert_eq!(matches, expected_matches(Cost::ONE));
    }

    #[test]
    fn empty_query_sequence_reports_unavailable_threshold_subsequence() {
        let engine = engine();
        let result = engine
            .range_searcher(CharacterCosts)
            .search(&[], &RangeSearchParams::new(Cost::ZERO));

        assert_eq!(result, Err(Error::ThresholdSubsequenceUnavailable));
    }

    #[test]
    fn embedding_filter_and_verify_matches_exhaustive_verification() {
        let mut embeddings = EmbeddingStore::new(NonZeroUsize::new(2).unwrap());
        embeddings.insert('x', vec![0.8, 0.6]).unwrap();
        embeddings.insert('y', vec![0.6, 0.8]).unwrap();
        embeddings.insert('a', vec![1.0, 0.0]).unwrap();
        embeddings.insert('b', vec![0.0, 1.0]).unwrap();
        embeddings.insert('c', vec![-1.0, 0.0]).unwrap();

        let costs = CosineEmbeddingCosts::new(embeddings);
        let mut builder = SearchEngineBuilder::new();
        builder.add_sequence(['a', 'b']).unwrap();
        builder.add_sequence(['b', 'a']).unwrap();
        builder.add_sequence(['a', 'c']).unwrap();
        let engine = builder.build().unwrap();
        let searcher = engine.range_searcher(costs);
        let threshold = Cost::new_const(0.5);
        let eta = Cost::new_const(0.25);

        let encoded_query = EncodedQuery::new(vec!['x', 'y'], &engine.vocabulary).unwrap();
        let encoded_costs = encoded_query.costs(&engine.vocabulary, &searcher.costs);
        assert!(
            MinCandidateSelector
                .select(
                    encoded_query.string(),
                    threshold,
                    eta,
                    &engine.index,
                    &encoded_costs,
                    &engine.neighborhood,
                )
                .is_ok()
        );

        let filtered = searcher
            .search(
                &['x', 'y'],
                &RangeSearchParams::new(threshold).with_eta(eta),
            )
            .unwrap();
        let exhaustive = verify_exhaustively(
            encoded_query.string(),
            threshold,
            &engine.store,
            &encoded_costs,
        )
        .unwrap();

        assert_eq!(filtered, exhaustive);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].sequence_id, SequenceId::new(0));
        assert_eq!(filtered[0].token_range, Position::new(0)..Position::new(2));
    }
}
