//! Fixed-threshold range search orchestration.

use std::hash::Hash;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::search::SearchEngine;
use crate::search::bound::StrictBound;
use crate::search::encoding::EncodedQuery;
use crate::search::filtering::{
    MinCandidateSelector, SelectedPosition, generate_candidates, smallest_selectable_eta,
};
use crate::search::verification::Verifier;
use crate::search::{Candidate, Match};
use crate::store::CorpusStore;
use crate::types::{Position, SequenceId, Symbol};

/// Parameters for threshold range search.
///
/// The threshold is inclusive: a result is returned when its distance is less
/// than or equal to [`Self::threshold`]. Most callers only need [`Self::new`];
/// [`Self::with_eta`] is a filtering-performance tuning control and does not
/// change which results are correct. Both values must be finite and
/// non-negative, and the threshold must also be less than [`f32::MAX`] so that
/// a search bound exists above it; they are validated when a search starts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeSearchParams {
    threshold: f32,
    eta: Option<f32>,
}

/// Measurements from the filtering phase of one range search.
///
/// Counts are exposed so benchmark tooling can compare candidate generation
/// across implementations without making elapsed time a test assertion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RangeSearchMetrics {
    /// Whether filtering was unavailable and exhaustive verification was used.
    pub used_exhaustive_verification: bool,
    /// The eta filtering was re-tuned to, or `None` when the configured eta
    /// selected query positions on its own.
    ///
    /// An eta too small to select is raised to the smallest one that can,
    /// because the alternative is exhaustive verification. This is `None` when
    /// exhaustive verification was used, since re-tuning is what failed there.
    pub adjusted_eta: Option<Cost>,
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
    pub const fn new(threshold: f32) -> Self {
        Self {
            threshold,
            eta: None,
        }
    }

    /// Uses an explicit substitution-neighborhood radius for filtering.
    ///
    /// Leave this unset unless profiling shows that the automatic value is a
    /// bottleneck. The radius affects candidate generation, not the distance
    /// threshold used to verify results.
    ///
    /// A radius too small to construct a threshold subsequence is raised to
    /// the smallest one that can, since the alternative is exhaustive
    /// verification;
    /// [`RangeSearchMetrics::adjusted_eta`] reports the radius
    /// a search settled on.
    pub const fn with_eta(mut self, eta: f32) -> Self {
        self.eta = Some(eta);
        self
    }

    /// Returns the inclusive distance threshold.
    pub const fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Returns the explicit eta, or `None` when eta is automatic.
    pub const fn eta(&self) -> Option<f32> {
        self.eta
    }
}

/// Returns the default substitution-neighborhood radius for a query length.
///
/// The radius is `threshold / query_sequence_len`, computed from the inclusive
/// threshold. For an empty query sequence, this returns zero rather than
/// dividing by zero; searching an empty query sequence retains its existing
/// error behavior.
pub(crate) fn automatic_eta(threshold: Cost, query_sequence_len: usize) -> Result<Cost> {
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
    /// If the configured eta cannot construct a complete threshold subsequence
    /// for a non-empty query sequence, the search raises eta to the smallest
    /// radius that can and filters with that. A query position contributes at
    /// most the cost of deleting its token, so no radius helps exactly when
    /// deleting the whole query sequence stays within the threshold. Only then
    /// does the engine fall back to exhaustive Smith-Waterman verification
    /// instead of returning [`Error::ThresholdSubsequenceUnavailable`]. With
    /// unit costs, `threshold >= query_sequence.len()` is
    /// such a case.
    ///
    /// The fallback therefore also covers every search that could match a data
    /// segment without pairing any tokens with it. Such an alignment costs at
    /// least the cost of deleting the whole query sequence, which is exactly
    /// the sum no eta can lift the contributions above. Filtering may
    /// therefore answer with anchored data segments whatever the relationship
    /// between the operation costs; see [`EditCosts`].
    ///
    /// The fallback takes `O(m * sum(n_i^2))` time for query-sequence length
    /// `m` and
    /// data string lengths `n_i`, and can return `O(sum(n_i^2))` data
    /// segments.
    /// It may therefore be substantially slower and produce many more results
    /// than the normal filter-and-verify path.
    ///
    /// Searching takes `&self`, so one searcher can serve concurrent queries.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty query, invalid parameters, or costs that
    /// cannot be represented safely by the search algorithm.
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
        let threshold = Cost::new(params.threshold)?;
        // The public threshold is inclusive and everything below this line
        // searches for distances strictly below a bound, so the conversion
        // happens here, once, where the two meet.
        let bound = StrictBound::from_inclusive(threshold)?;
        let eta = params.eta.map(Cost::new).transpose()?;
        let encoded_query = EncodedQuery::new(query_sequence.to_vec(), &self.engine.vocabulary)?;
        let costs = encoded_query.costs(&self.engine.vocabulary, &self.costs);
        let eta = match eta {
            Some(eta) => eta,
            // Eta bounds substitution costs rather than distances, so it
            // divides the inclusive threshold and not the strict bound.
            None => automatic_eta(threshold, encoded_query.string().len())?,
        };
        self.search_query_string(encoded_query.string(), bound, eta, &costs)
    }

    fn search_query_string<S>(
        &self,
        query_string: &[Symbol],
        bound: StrictBound,
        eta: Cost,
        costs: &S,
    ) -> Result<(Vec<Match>, RangeSearchMetrics)>
    where
        S: EditCosts<Symbol>,
    {
        let Some((selected, adjusted_eta)) =
            self.select_positions(query_string, bound, eta, costs)?
        else {
            let matches = verify_exhaustively(query_string, bound, &self.engine.store, costs)?;
            return Ok((
                matches,
                RangeSearchMetrics {
                    used_exhaustive_verification: true,
                    ..RangeSearchMetrics::default()
                },
            ));
        };
        let selected_query_positions = selected.len();
        let candidates = generate_candidates(selected, &self.engine.index);
        let matches = Verifier::BidirectionalTrie.verify(
            query_string,
            &candidates,
            &self.engine.store,
            bound,
            costs,
        )?;
        Ok((
            matches,
            RangeSearchMetrics {
                used_exhaustive_verification: false,
                adjusted_eta,
                selected_query_positions,
                generated_candidates: candidates.len(),
            },
        ))
    }

    /// Selects query positions, widening eta when the configured radius cannot
    /// construct a threshold subsequence.
    ///
    /// Returns the selected positions together with the radius they were
    /// re-tuned to, or `None` when no radius selects and the search has to
    /// verify exhaustively.
    fn select_positions<S>(
        &self,
        query_string: &[Symbol],
        bound: StrictBound,
        eta: Cost,
        costs: &S,
    ) -> Result<Option<(Vec<SelectedPosition>, Option<Cost>)>>
    where
        S: EditCosts<Symbol>,
    {
        match self.select_at(query_string, bound, eta, costs) {
            Ok(selected) => return Ok(Some((selected, None))),
            // An empty query sequence keeps reporting the unavailable
            // subsequence rather than verifying every string against nothing.
            Err(Error::ThresholdSubsequenceUnavailable) if !query_string.is_empty() => {}
            Err(error) => return Err(error),
        }

        let Some(retuned) =
            smallest_selectable_eta(query_string, bound, eta, costs, &self.engine.neighborhood)
        else {
            return Ok(None);
        };
        match self.select_at(query_string, bound, retuned, costs) {
            Ok(selected) => Ok(Some((selected, Some(retuned)))),
            // Re-tuning adds the contributions in query order while selection
            // adds the ones it picks, so at the bound the two can part by a
            // final bit. Exhaustive verification answers either way.
            Err(Error::ThresholdSubsequenceUnavailable) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Selects query positions at one eta.
    fn select_at<S>(
        &self,
        query_string: &[Symbol],
        bound: StrictBound,
        eta: Cost,
        costs: &S,
    ) -> Result<Vec<SelectedPosition>>
    where
        S: EditCosts<Symbol>,
    {
        MinCandidateSelector.select(
            query_string,
            bound,
            eta,
            &self.engine.index,
            costs,
            &self.engine.neighborhood,
        )
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
    bound: StrictBound,
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
            .ok_or(Error::UnknownString(string_id.as_usize()))?;
        if !string.is_empty() {
            candidates.push(Candidate {
                string_id,
                string_position: Position::new(0),
                query_position: Position::new(0),
            });
        }
    }
    Verifier::SmithWaterman.verify(query_string, &candidates, corpus, bound, costs)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        MinCandidateSelector, RangeSearchParams, StrictBound, automatic_eta, verify_exhaustively,
    };
    use crate::costs::embedding::{CosineEmbeddingCosts, EmbeddingStoreBuilder};
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

        SearchEngine::from_parts(vocabulary, index_builder.build(), store_builder.build(2)).unwrap()
    }

    fn expected_matches(distance: f32) -> Vec<Match> {
        vec![
            Match {
                sequence_id: 0,
                token_range: 0..1,
                distance,
            },
            Match {
                sequence_id: 1,
                token_range: 0..1,
                distance,
            },
        ]
    }

    #[test]
    fn applies_substitution_cost_for_query_only_token() {
        let engine = engine();
        let matches = engine
            .range_searcher(CharacterCosts)
            .search(&['y'], &RangeSearchParams::new(0.4))
            .unwrap();

        assert_eq!(matches, expected_matches(0.4));
    }

    #[test]
    fn applies_deletion_cost_for_query_only_token() {
        let engine = engine();
        let matches = engine
            .range_searcher(CharacterCosts)
            .search(&['x', 'a'], &RangeSearchParams::new(0.25))
            .unwrap();

        assert_eq!(matches, expected_matches(0.25));
    }

    #[test]
    fn searcher_can_serve_concurrent_queries() {
        let engine = engine();
        let searcher = engine.range_searcher(CharacterCosts);
        let substitution_params = RangeSearchParams::new(0.4);
        let deletion_params = RangeSearchParams::new(0.25);

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
        let params = RangeSearchParams::new(0.4);
        let character = engine.range_searcher(CharacterCosts);
        let levenshtein = engine.range_searcher(LevenshteinCosts::new());

        assert_eq!(
            character.search(&['y'], &params).unwrap(),
            expected_matches(0.4)
        );
        assert!(levenshtein.search(&['y'], &params).unwrap().is_empty());
    }

    #[test]
    fn parameters_expose_threshold_and_optional_eta() {
        let threshold = 0.75;
        let eta = 0.25;

        let automatic = RangeSearchParams::new(threshold);
        let explicit = automatic.with_eta(eta);

        assert_eq!(automatic.threshold(), threshold);
        assert_eq!(automatic.eta(), None);
        assert_eq!(explicit.threshold(), threshold);
        assert_eq!(explicit.eta(), Some(eta));
    }

    #[test]
    fn rejects_invalid_public_search_parameters() {
        let engine = engine();
        let searcher = engine.range_searcher(CharacterCosts);

        assert!(matches!(
            searcher.search(&['a'], &RangeSearchParams::new(f32::NAN)),
            Err(Error::InvalidCost(value)) if value.is_nan()
        ));
        assert_eq!(
            searcher.search(&['a'], &RangeSearchParams::new(0.0).with_eta(-1.0)),
            Err(Error::InvalidCost(-1.0))
        );
    }

    #[test]
    fn rejects_a_threshold_without_a_strict_upper_bound() {
        let engine = engine();

        assert_eq!(
            engine
                .range_searcher(CharacterCosts)
                .search(&['a'], &RangeSearchParams::new(f32::MAX)),
            Err(Error::InvalidThreshold(f32::MAX))
        );
    }

    #[test]
    fn automatic_eta_divides_threshold_by_query_sequence_length() {
        assert_eq!(automatic_eta(Cost::new_const(0.75), 3).unwrap(), 0.25);
        assert_eq!(automatic_eta(Cost::ONE, 0).unwrap(), Cost::ZERO);
    }

    #[test]
    fn falls_back_to_exhaustive_search_when_no_threshold_subsequence_exists() {
        // Deleting 'x' costs 0.25, so no eta lifts the contribution above the
        // threshold of 1.0 and re-tuning has nothing to offer.
        let engine = engine();
        let (matches, metrics) = engine
            .range_searcher(CharacterCosts)
            .search_with_metrics(&['x'], &RangeSearchParams::new(1.0))
            .unwrap();

        assert_eq!(matches, expected_matches(1.0));
        assert!(metrics.used_exhaustive_verification);
        assert_eq!(metrics.adjusted_eta, None);
    }

    #[test]
    fn raises_an_eta_too_small_to_select_instead_of_verifying_exhaustively() {
        // At eta zero the cheapest way out of 'y' is substituting it for 'a'
        // at 0.4, which the threshold admits. Raising eta to 0.4 brings 'a'
        // into the neighborhood and leaves deletion, at 1.0, as the way out.
        let engine = engine();
        let (matches, metrics) = engine
            .range_searcher(CharacterCosts)
            .search_with_metrics(&['y'], &RangeSearchParams::new(0.4).with_eta(0.0))
            .unwrap();

        assert_eq!(matches, expected_matches(0.4));
        assert!(!metrics.used_exhaustive_verification);
        assert_eq!(metrics.adjusted_eta, Some(Cost::new_const(0.4)));
    }

    #[test]
    fn a_raised_eta_returns_what_exhaustive_verification_would() {
        let mut builder = SearchEngineBuilder::new();
        builder.add_sequence("xayb".chars()).unwrap();
        builder.add_sequence("aax".chars()).unwrap();
        let engine = builder.build().unwrap();
        let searcher = engine.range_searcher(CharacterCosts);
        // At eta zero the two positions contribute 0.4 and 1.0, which the
        // threshold admits; raising eta to 0.4 makes both contribute 1.0.
        let threshold = Cost::new_const(1.4);

        let (filtered, metrics) = searcher
            .search_with_metrics(&['y', 'a'], &RangeSearchParams::new(1.4).with_eta(0.0))
            .unwrap();

        let encoded_query = EncodedQuery::new(vec!['y', 'a'], &engine.vocabulary).unwrap();
        let exhaustive = verify_exhaustively(
            encoded_query.string(),
            StrictBound::from_inclusive(threshold).unwrap(),
            &engine.store,
            &encoded_query.costs(&engine.vocabulary, &searcher.costs),
        )
        .unwrap();

        assert!(metrics.adjusted_eta.is_some());
        assert!(!metrics.used_exhaustive_verification);
        assert!(!filtered.is_empty());
        assert_eq!(filtered, exhaustive);
    }

    #[test]
    fn an_eta_that_selects_on_its_own_is_left_alone() {
        let engine = engine();
        let (_, metrics) = engine
            .range_searcher(CharacterCosts)
            .search_with_metrics(&['y'], &RangeSearchParams::new(0.4))
            .unwrap();

        assert_eq!(metrics.adjusted_eta, None);
    }

    /// Costs that break `substitution <= deletion + insertion`, so the cheapest
    /// alignment of a segment can pair no tokens with the query sequence.
    struct CheapEditCosts;

    impl EditCosts<char> for CheapEditCosts {
        fn substitution(&self, from: &char, to: &char) -> Cost {
            if from == to { Cost::ZERO } else { Cost::ONE }
        }

        fn deletion(&self, _token: &char) -> Cost {
            Cost::new_const(0.1)
        }

        fn insertion(&self, _token: &char) -> Cost {
            Cost::new_const(0.1)
        }
    }

    #[test]
    fn segments_matched_without_pairing_a_token_are_verified_exhaustively() {
        // Deleting the query token and inserting a data token costs 0.2, while
        // substituting one for the other costs 1.0. Anchored verification would
        // miss every segment here, so the search must not reach it.
        let mut builder = SearchEngineBuilder::new();
        builder.add_sequence("xbz".chars()).unwrap();
        let engine = builder.build().unwrap();

        let (matches, metrics) = engine
            .range_searcher(CheapEditCosts)
            .search_with_metrics(&['a'], &RangeSearchParams::new(0.5))
            .unwrap();

        assert!(metrics.used_exhaustive_verification);
        assert_eq!(
            matches
                .iter()
                .map(|matched| (matched.token_range.clone(), matched.distance))
                .collect::<Vec<_>>(),
            [
                (0..1, 0.2),
                (0..2, 0.3),
                (0..3, 0.4),
                (1..2, 0.2),
                (1..3, 0.3),
                (2..3, 0.2),
            ]
        );
    }

    #[test]
    fn empty_query_sequence_reports_unavailable_threshold_subsequence() {
        let engine = engine();
        let result = engine
            .range_searcher(CharacterCosts)
            .search(&[], &RangeSearchParams::new(0.0));

        assert_eq!(result, Err(Error::ThresholdSubsequenceUnavailable));
    }

    #[test]
    fn embedding_filter_and_verify_matches_exhaustive_verification() {
        let mut embeddings = EmbeddingStoreBuilder::new(NonZeroUsize::new(2).unwrap());
        embeddings.insert('x', vec![0.8, 0.6]).unwrap();
        embeddings.insert('y', vec![0.6, 0.8]).unwrap();
        embeddings.insert('a', vec![1.0, 0.0]).unwrap();
        embeddings.insert('b', vec![0.0, 1.0]).unwrap();
        embeddings.insert('c', vec![-1.0, 0.0]).unwrap();

        let costs = CosineEmbeddingCosts::new(embeddings.build());
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
                    StrictBound::from_inclusive(threshold).unwrap(),
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
                &RangeSearchParams::new(threshold.into()).with_eta(eta.into()),
            )
            .unwrap();
        let exhaustive = verify_exhaustively(
            encoded_query.string(),
            StrictBound::from_inclusive(threshold).unwrap(),
            &engine.store,
            &encoded_costs,
        )
        .unwrap();

        assert_eq!(filtered, exhaustive);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].sequence_id, 0);
        assert_eq!(filtered[0].token_range, 0..2);
    }
}
