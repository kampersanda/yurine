//! Fixed-threshold range search orchestration.

use std::hash::Hash;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::search::Match;
use crate::search::SearchEngine;
use crate::search::bound::StrictBound;
use crate::search::encoding::EncodedQuery;
use crate::search::filtering::{
    MinCandidateSelector, SelectedPosition, any_radius_can_select, count_candidates,
    generate_candidates, wider_radii,
};
use crate::search::verification::{Verifier, keep_best_per_overlap};
use crate::types::Symbol;

/// Parameters for threshold range search.
///
/// The threshold is inclusive: a result is returned when its distance is less
/// than or equal to [`Self::threshold`]. It must be finite, non-negative, and
/// less than [`f32::MAX`] so that a search bound exists above it; this is
/// validated when a search starts.
///
/// The substitution-neighborhood radius filtering works at is derived from the
/// threshold and the query length, not configured here. See
/// [`RangeSearcher::search`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeSearchParams {
    threshold: f32,
    max_candidates: Option<usize>,
}

/// Measurements from the filtering phase of one range search.
///
/// Counts are exposed so benchmark tooling can compare candidate generation
/// across implementations without making elapsed time a test assertion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RangeSearchMetrics {
    /// The radius filtering was widened to, or `None` when the automatic
    /// radius selected query positions on its own.
    ///
    /// A radius too small to select is raised to the smallest one that can.
    pub adjusted_eta: Option<Cost>,
    /// Number of query positions chosen for candidate generation.
    pub selected_query_positions: usize,
    /// Number of candidate anchors generated from postings.
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
    /// Creates parameters for an inclusive distance threshold.
    pub const fn new(threshold: f32) -> Self {
        Self {
            threshold,
            max_candidates: None,
        }
    }

    /// Limits how many candidates filtering may generate.
    ///
    /// Filtering work grows with the threshold whatever a search returns, so a
    /// loose threshold can make a search slow while it answers with little.
    /// The candidate count is known exactly before any candidate is generated,
    /// so a search above `max_candidates` is declined with
    /// [`Error::SearchTooExpensive`] rather than answered slowly.
    ///
    /// Searches are unlimited by default.
    /// [`RangeSearchMetrics::generated_candidates`] reports what a search
    /// generated, which is the measurement to choose a limit from.
    pub const fn with_max_candidates(mut self, max_candidates: usize) -> Self {
        self.max_candidates = Some(max_candidates);
        self
    }

    /// Returns the inclusive distance threshold.
    pub const fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Returns the maximum number of candidates, or `None` when a search may
    /// generate as many as filtering asks for.
    pub const fn max_candidates(&self) -> Option<usize> {
        self.max_candidates
    }
}

/// Returns the substitution-neighborhood radius for a query length.
///
/// The radius is `threshold / query_sequence_len`, computed from the inclusive
/// threshold. For an empty query sequence, this returns zero rather than
/// dividing by zero; such a search is declined either way.
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
    /// Overlapping segments describe one match, so a search returns one segment
    /// for each group of them: the closest to the query, and among equally
    /// close ones the shortest and then the leftmost. Segments returned for a
    /// data sequence therefore never overlap. Where rounding leaves two
    /// distances indistinguishable, which of the two segments is returned
    /// follows the distances before rounding. A segment that a threshold admits
    /// is not returned when a segment overlapping it is closer; raising the
    /// threshold reaches further segments rather than more of the same match.
    ///
    /// Results are ordered by data sequence ID, then token-range start, then
    /// token-range end.
    ///
    /// Filtering derives its substitution-neighborhood radius from the search
    /// itself, as `threshold / query_sequence.len()`. This favors constructing
    /// a threshold subsequence for continuous substitution costs.
    ///
    /// If that radius cannot construct a complete threshold subsequence, the
    /// search raises it to the smallest one that can and filters with that.
    /// Widening costs `O((m * a + m^2) * log(m * a))` time for query-sequence
    /// length `m` and alphabet size `a`, which does not grow with the corpus,
    /// and a search the derived radius already filters with does not pay it at
    /// all. [`RangeSearchMetrics::adjusted_eta`] reports what a search settled
    /// on.
    ///
    /// Searching takes `&self`, so one searcher can serve concurrent queries.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ThresholdSubsequenceUnavailable`] when the threshold
    /// reaches the cost of deleting the whole query sequence. A query position
    /// contributes at most that token's deletion cost to filtering, so no eta
    /// selects for such a search; equally, a data segment could match it
    /// without pairing any tokens with it, which anchored verification does
    /// not measure. Both are why the search declines it rather than answering
    /// it slowly. With unit costs, `threshold >= query_sequence.len()` is such
    /// a case, and so is an empty query sequence. Filtering answers every
    /// accepted search with anchored data segments whatever the relationship
    /// between the operation costs; see [`EditCosts`].
    ///
    /// Returns [`Error::SearchTooExpensive`] when
    /// [`RangeSearchParams::with_max_candidates`] set a maximum and filtering
    /// would generate more candidates than that. Nothing is generated or
    /// verified in that case.
    ///
    /// Also returns an error for invalid parameters, or costs that cannot be
    /// represented safely by the search algorithm.
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
        let encoded_query = EncodedQuery::new(query_sequence.to_vec(), &self.engine.vocabulary)?;
        let costs = encoded_query.costs(&self.engine.vocabulary, &self.costs);
        // Eta bounds substitution costs rather than distances, so it divides
        // the inclusive threshold and not the strict bound.
        let eta = automatic_eta(threshold, encoded_query.string().len())?;
        self.search_query_string(
            encoded_query.string(),
            bound,
            eta,
            params.max_candidates,
            &costs,
        )
    }

    fn search_query_string<S>(
        &self,
        query_string: &[Symbol],
        bound: StrictBound,
        eta: Cost,
        max_candidates: Option<usize>,
        costs: &S,
    ) -> Result<(Vec<Match>, RangeSearchMetrics)>
    where
        S: EditCosts<Symbol>,
    {
        // A query position contributes at most the cost of deleting its
        // token, so a bound admitting the whole query's deletions admits an
        // alignment that pairs nothing. Neither filtering nor anchored
        // verification answers such a search, and this is the one check that
        // keeps both out of that territory.
        if !any_radius_can_select(query_string, bound, costs) {
            return Err(Error::ThresholdSubsequenceUnavailable);
        }
        let (selected, adjusted_eta) = self.select_positions(query_string, bound, eta, costs)?;
        let selected_query_positions = selected.len();
        // Counting reads posting-list lengths for the radius selection settled
        // on, so a declined search has neither generated a candidate nor
        // verified one.
        if let Some(limit) = max_candidates {
            let candidates = count_candidates(&selected, &self.engine.index);
            if candidates > limit {
                return Err(Error::SearchTooExpensive { candidates, limit });
            }
        }
        let candidates = generate_candidates(selected, &self.engine.index);
        let matches = Verifier::BidirectionalTrie.verify(
            query_string,
            &candidates,
            &self.engine.store,
            bound,
            costs,
        )?;
        // Anchors that reach the same part of a data sequence report segments
        // that overlap. Reducing them here is what makes one result stand for
        // one match.
        let matches = keep_best_per_overlap(matches);
        Ok((
            matches,
            RangeSearchMetrics {
                adjusted_eta,
                selected_query_positions,
                generated_candidates: candidates.len(),
            },
        ))
    }

    /// Selects query positions, widening eta when the derived radius cannot
    /// construct a threshold subsequence.
    ///
    /// Returns the selected positions together with the radius they were
    /// re-tuned to. Callers check [`any_radius_can_select`] first, so a radius
    /// that selects exists; the search reports
    /// [`Error::ThresholdSubsequenceUnavailable`] if none is found anyway.
    ///
    /// Widening collects the radii for `O(m * a * log(m * a))` and then binary
    /// searches them, running one selection per step, so for query-string
    /// length `m` and alphabet size `a` the whole of it stays within
    /// `O((m * a + m^2) * log(m * a))`. That is bounded by the query and the
    /// alphabet alone, and none of it is paid when the derived radius selects
    /// on its own.
    fn select_positions<S>(
        &self,
        query_string: &[Symbol],
        bound: StrictBound,
        eta: Cost,
        costs: &S,
    ) -> Result<(Vec<SelectedPosition>, Option<Cost>)>
    where
        S: EditCosts<Symbol>,
    {
        match self.select_at(query_string, bound, eta, costs) {
            Ok(selected) => return Ok((selected, None)),
            Err(Error::ThresholdSubsequenceUnavailable) => {}
            Err(error) => return Err(error),
        }

        // Selection is what a radius has to satisfy, so the search runs it at
        // each radius it weighs rather than predicting the outcome. A radius
        // reported here therefore selected in fact, and the widest radius the
        // alphabet offers is always tried before giving up.
        let radii = wider_radii(query_string, eta, costs, &self.engine.neighborhood);
        let mut narrowest = None;
        let (mut low, mut high) = (0, radii.len());
        while low < high {
            let middle = low + (high - low) / 2;
            match self.select_at(query_string, bound, radii[middle], costs) {
                Ok(selected) => {
                    narrowest = Some((selected, Some(radii[middle])));
                    high = middle;
                }
                Err(Error::ThresholdSubsequenceUnavailable) => low = middle + 1,
                Err(error) => return Err(error),
            }
        }
        narrowest.ok_or(Error::ThresholdSubsequenceUnavailable)
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{MinCandidateSelector, RangeSearchParams, StrictBound, automatic_eta};
    use crate::costs::embedding::{CosineEmbeddingCosts, EmbeddingStoreBuilder};
    use crate::costs::levenshtein::LevenshteinCosts;
    use crate::costs::{Cost, EditCosts};
    use crate::errors::{Error, Result};
    use crate::postings::PostingsIndexBuilder;
    use crate::search::SearchEngineBuilder;
    use crate::search::encoding::EncodedQuery;
    use crate::search::verification::Verifier;
    use crate::search::verification::tests::assert_answers;
    use crate::search::{Candidate, Match, SearchEngine};
    use crate::store::{CorpusStore, CorpusStoreBuilder};
    use crate::types::{Position, Posting, SequenceId, Symbol};
    use crate::vocabulary::VocabularyBuilder;

    /// Exhaustively verifies every non-empty string in the corpus.
    ///
    /// Range search answers with anchored verification alone, so this is the
    /// independent oracle its results are checked against.
    fn verify_exhaustively<C>(
        query_string: &[Symbol],
        bound: StrictBound,
        corpus: &CorpusStore,
        costs: &C,
    ) -> Result<Vec<Match>>
    where
        C: EditCosts<Symbol>,
    {
        // Smith-Waterman verification uses candidates only to select data
        // strings. One in-bounds anchor per non-empty string requests
        // exhaustive verification without relying on the filtering guarantee.
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
    fn parameters_expose_the_threshold() {
        let threshold = 0.75;

        assert_eq!(RangeSearchParams::new(threshold).threshold(), threshold);
    }

    #[test]
    fn parameters_are_unlimited_until_a_candidate_maximum_is_set() {
        let params = RangeSearchParams::new(0.75);

        assert_eq!(params.max_candidates(), None);
        assert_eq!(params.with_max_candidates(3).max_candidates(), Some(3));
    }

    #[test]
    fn accepts_a_search_at_the_candidate_maximum() {
        let engine = engine();
        let params = RangeSearchParams::new(0.4).with_max_candidates(2);

        assert_eq!(
            engine
                .range_searcher(CharacterCosts)
                .search(&['y'], &params)
                .unwrap(),
            expected_matches(0.4)
        );
    }

    #[test]
    fn declines_a_search_above_the_candidate_maximum() {
        let engine = engine();
        let searcher = engine.range_searcher(CharacterCosts);
        // The count a limit is compared against is the one an unlimited search
        // reports, so the two are checked against each other here.
        let (_, metrics) = searcher
            .search_with_metrics(&['y'], &RangeSearchParams::new(0.4))
            .unwrap();
        let limit = metrics.generated_candidates - 1;

        assert_eq!(
            searcher.search(
                &['y'],
                &RangeSearchParams::new(0.4).with_max_candidates(limit)
            ),
            Err(Error::SearchTooExpensive {
                candidates: metrics.generated_candidates,
                limit,
            })
        );
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
            searcher.search(&['a'], &RangeSearchParams::new(-1.0)),
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
    fn declines_a_threshold_reaching_the_query_deletion_cost() {
        // Deleting 'x' costs 0.25, which the threshold of 1.0 admits, so no
        // eta lifts the contribution above it and the search is declined.
        let engine = engine();

        assert_eq!(
            engine
                .range_searcher(CharacterCosts)
                .search(&['x'], &RangeSearchParams::new(1.0)),
            Err(Error::ThresholdSubsequenceUnavailable)
        );
    }

    /// Deletion costs that differ between tokens, as `CustomCosts` allows.
    ///
    /// One token is nearly free to delete, so it can never contribute much,
    /// and the derived radius leaves the other capped by a cheap substitution
    /// rather than by its own deletion cost.
    struct SkewedCosts;

    impl EditCosts<char> for SkewedCosts {
        fn substitution(&self, from: &char, to: &char) -> Cost {
            if from == to {
                Cost::ZERO
            } else {
                Cost::new_const(0.6)
            }
        }

        fn deletion(&self, token: &char) -> Cost {
            if *token == 'a' {
                Cost::new_const(0.05)
            } else {
                Cost::new_const(2.0)
            }
        }

        fn insertion(&self, _token: &char) -> Cost {
            Cost::ONE
        }
    }

    /// An engine over the tokens `SkewedCosts` distinguishes.
    fn skewed_engine() -> SearchEngine<char> {
        let mut builder = SearchEngineBuilder::new();
        builder.add_sequence("ab".chars()).unwrap();
        builder.build().unwrap()
    }

    #[test]
    fn raises_a_radius_too_small_to_select() {
        // The derived radius is 1.0 / 2 = 0.5, where 'a' contributes its 0.05
        // deletion and 'b' is capped at the 0.6 substitution, for 0.65 against
        // a threshold of 1.0. At 0.6 that substitution joins the neighborhood
        // and 'b' contributes its 2.0 deletion instead.
        let engine = skewed_engine();

        let (matches, metrics) = engine
            .range_searcher(SkewedCosts)
            .search_with_metrics(&['a', 'b'], &RangeSearchParams::new(1.0))
            .unwrap();

        assert_eq!(metrics.adjusted_eta, Some(Cost::new_const(0.6)));
        assert_eq!(metrics.selected_query_positions, 1);
        assert_eq!(
            matches
                .iter()
                .map(|matched| (matched.token_range.clone(), matched.distance))
                .collect::<Vec<_>>(),
            // The whole string matches the query exactly. The two
            // single-token segments overlapping it are further away, at
            // 0.65 and 0.05, so the group reduces to the exact match.
            [(0..2, 0.0)]
        );
    }

    #[test]
    fn a_raised_radius_returns_what_exhaustive_verification_would() {
        let engine = skewed_engine();
        let searcher = engine.range_searcher(SkewedCosts);
        let threshold = Cost::ONE;

        let (filtered, metrics) = searcher
            .search_with_metrics(&['a', 'b'], &RangeSearchParams::new(1.0))
            .unwrap();

        let encoded_query = EncodedQuery::new(vec!['a', 'b'], &engine.vocabulary).unwrap();
        let exhaustive = verify_exhaustively(
            encoded_query.string(),
            StrictBound::from_inclusive(threshold).unwrap(),
            &engine.store,
            &encoded_query.costs(&engine.vocabulary, &searcher.costs),
        )
        .unwrap();

        assert!(metrics.adjusted_eta.is_some());
        assert!(!filtered.is_empty());
        assert_answers(&filtered, &exhaustive);
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
    fn declines_a_search_a_segment_could_match_without_pairing_a_token() {
        // Deleting the query token and inserting a data token costs 0.2, while
        // substituting one for the other costs 1.0. Anchored verification
        // would miss every segment here, so the search must not reach it, and
        // the deletion cost of 0.1 is what the threshold of 0.5 reaches.
        let mut builder = SearchEngineBuilder::new();
        builder.add_sequence("xbz".chars()).unwrap();
        let engine = builder.build().unwrap();

        assert_eq!(
            engine
                .range_searcher(CheapEditCosts)
                .search(&['a'], &RangeSearchParams::new(0.5)),
            Err(Error::ThresholdSubsequenceUnavailable)
        );
    }

    /// The oracle the declined search above would have needed.
    ///
    /// Anchored verification cannot answer it, which is why range search
    /// declines it rather than returning these segments.
    #[test]
    fn exhaustive_verification_finds_the_segments_anchoring_would_miss() {
        let mut builder = SearchEngineBuilder::new();
        builder.add_sequence("xbz".chars()).unwrap();
        let engine = builder.build().unwrap();
        let encoded_query = EncodedQuery::new(vec!['a'], &engine.vocabulary).unwrap();

        let matches = verify_exhaustively(
            encoded_query.string(),
            StrictBound::from_inclusive(Cost::new_const(0.5)).unwrap(),
            &engine.store,
            &encoded_query.costs(&engine.vocabulary, &CheapEditCosts),
        )
        .unwrap();

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

        // Two query tokens against this threshold derive exactly `eta`.
        let filtered = searcher
            .search(&['x', 'y'], &RangeSearchParams::new(threshold.into()))
            .unwrap();
        let exhaustive = verify_exhaustively(
            encoded_query.string(),
            StrictBound::from_inclusive(threshold).unwrap(),
            &engine.store,
            &encoded_costs,
        )
        .unwrap();

        assert_answers(&filtered, &exhaustive);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].sequence_id, 0);
        assert_eq!(filtered[0].token_range, 0..2);
    }
}
