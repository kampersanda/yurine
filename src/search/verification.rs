//! Verification of candidates against a distance threshold.

mod bidirectional_trie;
mod smith_waterman;

use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::search::{Candidate, Match};
use crate::store::CorpusStore;
use crate::types::{SequenceId, Symbol};

/// Verification algorithm used to check filtering candidates.
pub(in crate::search) enum Verifier {
    BidirectionalTrie,
    SmithWaterman,
}

impl Verifier {
    /// Returns exactly the non-empty substrings whose distance is at most
    /// `threshold`.
    ///
    /// Each substring must occur exactly once. Results must be ordered by data
    /// string ID, then symbol-range start, then symbol-range end.
    pub(in crate::search) fn verify<C>(
        &self,
        query_string: &[Symbol],
        candidates: &[Candidate],
        corpus: &CorpusStore,
        threshold: Cost,
        costs: &C,
    ) -> Result<Vec<Match>>
    where
        C: EditCosts<Symbol>,
    {
        match self {
            Self::BidirectionalTrie => {
                bidirectional_trie::verify(query_string, candidates, corpus, threshold, costs)
            }
            Self::SmithWaterman => {
                smith_waterman::verify(query_string, candidates, corpus, threshold, costs)
            }
        }
    }
}

fn create_match(
    string_id: SequenceId,
    symbol_start: usize,
    symbol_end: usize,
    distance: Cost,
) -> Match {
    Match {
        sequence_id: string_id.as_usize(),
        token_range: symbol_start..symbol_end,
        distance: distance.into(),
    }
}

/// Validates that a candidate's string ID and positions are within bounds.
fn validated_candidate_string<'a>(
    query_string: &[Symbol],
    candidate: &Candidate,
    corpus: &'a CorpusStore,
) -> Result<&'a [Symbol]> {
    let string = corpus
        .string(candidate.string_id)?
        .ok_or(Error::UnknownString(candidate.string_id.as_usize()))?;
    let string_position = candidate.string_position.as_usize();
    if string_position >= string.len() {
        return Err(Error::InvalidStringPosition {
            position: string_position,
            string_len: string.len(),
        });
    }
    let query_position = candidate.query_position.as_usize();
    if query_position >= query_string.len() {
        return Err(Error::InvalidQueryPosition {
            position: query_position,
            query_len: query_string.len(),
        });
    }
    Ok(string)
}

/// Adds two non-negative DP distances, mapping an exact sum above
/// [`f32::MAX`] to infinity.
///
/// IEEE-754 addition can round `f32::MAX +` a sufficiently small positive
/// value back to `f32::MAX`. The error-free transformation below recovers that
/// positive residual so an unrepresentable distance cannot qualify at a
/// finite strict search threshold.
fn add_distance(left: f32, right: f32) -> f32 {
    let sum = left + right;
    if !sum.is_finite() {
        return f32::INFINITY;
    }
    if sum != f32::MAX {
        return sum;
    }
    let right_rounded = sum - left;
    let residual = (left - (sum - right_rounded)) + (right - right_rounded);
    if residual > 0.0 { f32::INFINITY } else { sum }
}

/// Initializes the weighted-edit-distance column for an empty data prefix.
///
/// Internal DP cells use `f32` so accumulation above [`Cost::MAX`] becomes
/// infinity instead of being confused with an exact, representable maximum.
fn root_column<C>(query_string: &[Symbol], costs: &C) -> Vec<f32>
where
    C: EditCosts<Symbol>,
{
    // `column[r]` is wed(query_string[..r], empty). Reaching the empty data
    // prefix requires deleting every symbol in the query-string prefix.
    let mut column = Vec::with_capacity(query_string.len() + 1);
    column.push(0.0);
    for query_symbol in query_string {
        column.push(add_distance(
            column.last().copied().unwrap_or(0.0),
            costs.deletion(query_symbol).get(),
        ));
    }
    column
}

/// Advances a weighted-edit-distance column by one data symbol.
fn step_dp<C>(
    query_string: &[Symbol],
    string_symbol: Symbol,
    previous: &[f32],
    costs: &C,
) -> Vec<f32>
where
    C: EditCosts<Symbol>,
{
    debug_assert_eq!(previous.len(), query_string.len() + 1);

    // If `previous[r]` describes a processed data prefix P, `current[r]`
    // describes P followed by `string_symbol`. Row zero therefore inserts the
    // new data symbol into an empty query string.
    let mut current = Vec::with_capacity(query_string.len() + 1);
    current.push(add_distance(
        previous[0],
        costs.insertion(&string_symbol).get(),
    ));
    for (query_index, query_symbol) in query_string.iter().enumerate() {
        // The three predecessors consume both symbols, only the string symbol,
        // or only the query symbol, respectively. This fixes the direction as
        // wed(query_string, string prefix).
        let substitution = add_distance(
            previous[query_index],
            costs.substitution(query_symbol, &string_symbol).get(),
        );
        let insertion = add_distance(
            previous[query_index + 1],
            costs.insertion(&string_symbol).get(),
        );
        let deletion = add_distance(current[query_index], costs.deletion(query_symbol).get());
        current.push(substitution.min(insertion).min(deletion));
    }
    current
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{Verifier, add_distance};
    use crate::costs::{Cost, EditCosts};
    use crate::errors::Error;
    use crate::search::{Candidate, Match};
    use crate::store::{CorpusStore, CorpusStoreBuilder};
    use crate::types::{Position, SequenceId, Symbol};

    /// Ordinary Levenshtein costs.
    struct UnitCosts;

    impl EditCosts<Symbol> for UnitCosts {
        fn substitution(&self, from: &Symbol, to: &Symbol) -> Cost {
            if from == to { Cost::ZERO } else { Cost::ONE }
        }

        fn deletion(&self, _symbol: &Symbol) -> Cost {
            Cost::ONE
        }

        fn insertion(&self, _symbol: &Symbol) -> Cost {
            Cost::ONE
        }
    }

    /// Cost policies that parameterize the verification tests.
    ///
    /// Every policy keeps `substitution(from, to) <= deletion(from) +
    /// insertion(to)`. Anchored verification only considers alignments pairing
    /// at least one query symbol with one data symbol, and that inequality is
    /// what makes such an alignment optimal, so both verifiers must agree.
    ///
    /// Apart from [`CostPolicy::Unrepresentable`], all costs are multiples of
    /// `0.25` so that the sums produced by short alignments are exact in `f32`
    /// and threshold comparisons cannot depend on the order in which an
    /// implementation accumulates them.
    #[derive(Debug, Clone, Copy)]
    enum CostPolicy {
        /// Symmetric unit costs.
        Unit,
        /// Deletion is cheaper than insertion, so confusing the query direction
        /// with the data direction changes the result.
        Asymmetric,
        /// Costs depend on the symbols involved, and the substitution cost
        /// weighs the query symbol and the data symbol differently, so DP
        /// columns cached for one symbol must never be reused for another.
        SymbolDependent,
        /// Inserting [`UNREPRESENTABLE`] costs [`Cost::MAX`], so a DP cell that
        /// grows past the largest representable distance sits next to cells
        /// holding ordinary finite distances.
        ///
        /// The reference adds with plain `f32` arithmetic while the verifiers
        /// use [`add_distance`], so the two disagree above [`Cost::MAX`]. Every
        /// such value is far above the thresholds used here, so both sides
        /// still agree on which substrings match and on their distances.
        Unrepresentable,
    }

    /// The data symbol that [`CostPolicy::Unrepresentable`] cannot insert.
    const UNREPRESENTABLE: Symbol = Symbol::new(2);

    impl EditCosts<Symbol> for CostPolicy {
        fn substitution(&self, from: &Symbol, to: &Symbol) -> Cost {
            if from == to {
                return Cost::ZERO;
            }
            match self {
                Self::Unit | Self::Unrepresentable => Cost::ONE,
                Self::Asymmetric => Cost::new_const(1.25),
                // The query symbol and the data symbol carry different
                // coefficients, so this cost changes when the two are swapped.
                Self::SymbolDependent => {
                    Cost::new(0.25 * from.get() as f32 + 0.5 * to.get() as f32 + 0.25).unwrap()
                }
            }
        }

        fn deletion(&self, symbol: &Symbol) -> Cost {
            match self {
                Self::Unit | Self::Unrepresentable => Cost::ONE,
                Self::Asymmetric => Cost::new_const(0.5),
                Self::SymbolDependent => Cost::new(0.25 * (symbol.get() as f32 + 1.0)).unwrap(),
            }
        }

        fn insertion(&self, symbol: &Symbol) -> Cost {
            match self {
                Self::Unit => Cost::ONE,
                Self::Asymmetric => Cost::new_const(1.5),
                Self::SymbolDependent => Cost::new(0.5 * (symbol.get() as f32 + 1.0)).unwrap(),
                Self::Unrepresentable => {
                    if *symbol == UNREPRESENTABLE {
                        Cost::MAX
                    } else {
                        Cost::ONE
                    }
                }
            }
        }
    }

    /// Encodes `'a'` as symbol zero, `'b'` as symbol one, and so on.
    fn symbols(text: &str) -> Vec<Symbol> {
        text.chars()
            .map(|character| Symbol::new(character as u32 - 'a' as u32))
            .collect()
    }

    fn corpus(texts: &[&str]) -> CorpusStore {
        let mut builder = CorpusStoreBuilder::new();
        for text in texts {
            builder.add_string(symbols(text));
        }
        builder.build(26)
    }

    /// Returns every in-bounds anchor of every corpus string.
    ///
    /// Filtering guarantees that answers are anchored, so verification tests
    /// supply the complete anchor set instead of depending on the selector.
    fn all_candidates(texts: &[&str], query_len: usize) -> Vec<Candidate> {
        let mut candidates = Vec::new();
        for (raw_id, text) in texts.iter().enumerate() {
            for string_position in 0..text.chars().count() {
                for query_position in 0..query_len {
                    candidates.push(Candidate {
                        string_id: SequenceId::from_usize(raw_id).unwrap(),
                        string_position: Position::from_usize(string_position).unwrap(),
                        query_position: Position::from_usize(query_position).unwrap(),
                    });
                }
            }
        }
        candidates
    }

    fn candidate(string_id: u32, string_position: u32, query_position: u32) -> Candidate {
        Candidate {
            string_id: SequenceId::new(string_id),
            string_position: Position::new(string_position),
            query_position: Position::new(query_position),
        }
    }

    /// Computes wed(`query_string`, `string`) with a full DP matrix.
    ///
    /// This deliberately duplicates neither verifier: it keeps the whole matrix
    /// instead of streaming columns, so a shared indexing or direction mistake
    /// is unlikely to cancel out.
    fn weighted_edit_distance<C>(query_string: &[Symbol], string: &[Symbol], costs: &C) -> f32
    where
        C: EditCosts<Symbol>,
    {
        let mut matrix = vec![vec![0.0; string.len() + 1]; query_string.len() + 1];
        for (query_index, query_symbol) in query_string.iter().enumerate() {
            matrix[query_index + 1][0] =
                matrix[query_index][0] + costs.deletion(query_symbol).get();
        }
        for (string_index, string_symbol) in string.iter().enumerate() {
            matrix[0][string_index + 1] =
                matrix[0][string_index] + costs.insertion(string_symbol).get();
        }
        for (query_index, query_symbol) in query_string.iter().enumerate() {
            for (string_index, string_symbol) in string.iter().enumerate() {
                let substitution = matrix[query_index][string_index]
                    + costs.substitution(query_symbol, string_symbol).get();
                let deletion =
                    matrix[query_index][string_index + 1] + costs.deletion(query_symbol).get();
                let insertion =
                    matrix[query_index + 1][string_index] + costs.insertion(string_symbol).get();
                matrix[query_index + 1][string_index + 1] =
                    substitution.min(deletion).min(insertion);
            }
        }
        matrix[query_string.len()][string.len()]
    }

    /// Enumerates every non-empty substring satisfying the inclusive threshold,
    /// in the order required of a verifier.
    fn reference_matches<C>(
        query_string: &[Symbol],
        texts: &[&str],
        threshold: f32,
        costs: &C,
    ) -> Vec<Match>
    where
        C: EditCosts<Symbol>,
    {
        let mut matches = Vec::new();
        for (sequence_id, text) in texts.iter().enumerate() {
            let string = symbols(text);
            for symbol_start in 0..string.len() {
                for symbol_end in symbol_start + 1..=string.len() {
                    let distance = weighted_edit_distance(
                        query_string,
                        &string[symbol_start..symbol_end],
                        costs,
                    );
                    if distance <= threshold {
                        matches.push(Match {
                            sequence_id,
                            token_range: symbol_start..symbol_end,
                            distance,
                        });
                    }
                }
            }
        }
        matches
    }

    /// The reference comparison below assumes the inequality documented on
    /// [`CostPolicy`], so the policies are checked against it directly.
    #[rstest]
    fn every_cost_policy_keeps_substitution_within_deletion_and_insertion(
        #[values(
            CostPolicy::Unit,
            CostPolicy::Asymmetric,
            CostPolicy::SymbolDependent,
            CostPolicy::Unrepresentable
        )]
        costs: CostPolicy,
    ) {
        for from in 0..26 {
            for to in 0..26 {
                let (from, to) = (Symbol::new(from), Symbol::new(to));
                assert!(
                    costs.substitution(&from, &to).get()
                        <= costs.deletion(&from).get() + costs.insertion(&to).get(),
                    "{costs:?} substitutes {from} with {to} more cheaply than it edits them"
                );
            }
        }
    }

    #[test]
    fn symbol_dependent_costs_separate_the_query_side_from_the_data_side() {
        let costs = CostPolicy::SymbolDependent;
        let from = Symbol::new(0);
        let to = Symbol::new(1);

        // Swapping the two symbols must change the substitution cost, otherwise
        // the policy cannot detect a verifier that confuses the two directions.
        assert_ne!(
            costs.substitution(&from, &to),
            costs.substitution(&to, &from)
        );
        assert_ne!(costs.deletion(&from), costs.insertion(&from));
    }

    #[rstest]
    #[case::single_symbol("a", &["a"])]
    #[case::exact_and_shifted_occurrences("ab", &["xaby"])]
    #[case::repeated_symbols("aa", &["aaaa"])]
    #[case::interleaved_repetitions("abc", &["abcabc"])]
    #[case::several_strings("bc", &["abc", "cba", "b", "cbcb"])]
    #[case::empty_string_between_strings("ab", &["", "ab", ""])]
    #[case::query_longer_than_every_string("abcd", &["ab", "cd"])]
    #[case::disjoint_alphabets("ab", &["cd"])]
    #[case::transposition("ab", &["ba"])]
    fn verifier_returns_exactly_the_substrings_below_the_threshold(
        #[case] query_text: &str,
        #[case] texts: &[&str],
        #[values(Verifier::BidirectionalTrie, Verifier::SmithWaterman)] verifier: Verifier,
        #[values(
            CostPolicy::Unit,
            CostPolicy::Asymmetric,
            CostPolicy::SymbolDependent,
            CostPolicy::Unrepresentable
        )]
        costs: CostPolicy,
        #[values(0.0, 0.5, 1.0, 2.0, 3.5)] threshold: f32,
    ) {
        let query_string = symbols(query_text);
        let corpus = corpus(texts);
        let candidates = all_candidates(texts, query_string.len());

        let matches = verifier
            .verify(
                &query_string,
                &candidates,
                &corpus,
                Cost::new(threshold).unwrap(),
                &costs,
            )
            .unwrap();

        assert_eq!(
            matches,
            reference_matches(&query_string, texts, threshold, &costs)
        );
    }

    #[rstest]
    fn verification_orders_matches_by_string_then_range(
        #[values(Verifier::BidirectionalTrie, Verifier::SmithWaterman)] verifier: Verifier,
    ) {
        // The exact match of the query sits between two costlier ranges of the
        // same string, so ordering by distance rather than by range is visible.
        let texts = ["ba", "ab"];
        let corpus = corpus(&texts);
        let candidates = all_candidates(&texts, 2);

        let matches = verifier
            .verify(&symbols("ab"), &candidates, &corpus, Cost::ONE, &UnitCosts)
            .unwrap();

        assert_eq!(
            matches,
            [
                Match {
                    sequence_id: 0,
                    token_range: 0..1,
                    distance: 1.0,
                },
                Match {
                    sequence_id: 0,
                    token_range: 1..2,
                    distance: 1.0,
                },
                Match {
                    sequence_id: 1,
                    token_range: 0..1,
                    distance: 1.0,
                },
                Match {
                    sequence_id: 1,
                    token_range: 0..2,
                    distance: 0.0,
                },
                Match {
                    sequence_id: 1,
                    token_range: 1..2,
                    distance: 1.0,
                },
            ]
        );
    }

    #[rstest]
    fn verification_without_candidates_returns_no_matches(
        #[values(Verifier::BidirectionalTrie, Verifier::SmithWaterman)] verifier: Verifier,
    ) {
        let corpus = corpus(&["ab"]);

        let matches = verifier
            .verify(&symbols("ab"), &[], &corpus, Cost::ONE, &UnitCosts)
            .unwrap();

        assert!(matches.is_empty());
    }

    #[rstest]
    fn verification_reports_each_substring_once_at_its_smallest_distance(
        #[values(Verifier::BidirectionalTrie, Verifier::SmithWaterman)] verifier: Verifier,
    ) {
        // Range 0..2 is reachable from several anchors: aligning the query
        // symbols in order costs nothing, while anchoring the second query
        // symbol on the first data symbol costs one insertion and one deletion.
        // The costlier anchor is listed first so that reporting the
        // first-discovered distance instead of the smallest one is visible.
        let corpus = corpus(&["aa"]);
        let candidates = [
            candidate(0, 0, 1),
            candidate(0, 1, 0),
            candidate(0, 0, 0),
            candidate(0, 1, 1),
        ];

        let matches = verifier
            .verify(&symbols("aa"), &candidates, &corpus, Cost::ONE, &UnitCosts)
            .unwrap();

        assert_eq!(
            matches,
            [
                Match {
                    sequence_id: 0,
                    token_range: 0..1,
                    distance: 1.0,
                },
                Match {
                    sequence_id: 0,
                    token_range: 0..2,
                    distance: 0.0,
                },
                Match {
                    sequence_id: 0,
                    token_range: 1..2,
                    distance: 1.0,
                },
            ]
        );
    }

    #[rstest]
    fn verification_ignores_duplicated_candidates(
        #[values(Verifier::BidirectionalTrie, Verifier::SmithWaterman)] verifier: Verifier,
    ) {
        let corpus = corpus(&["ab"]);
        let anchor = candidate(0, 0, 0);

        let matches = verifier
            .verify(
                &symbols("ab"),
                &[anchor, anchor, anchor],
                &corpus,
                Cost::ZERO,
                &UnitCosts,
            )
            .unwrap();

        assert_eq!(
            matches,
            [Match {
                sequence_id: 0,
                token_range: 0..2,
                distance: 0.0,
            }]
        );
    }

    #[rstest]
    #[case(0.75, false)]
    #[case(1.0, true)]
    fn verification_applies_the_threshold_inclusively(
        #[case] threshold: f32,
        #[case] expected: bool,
        #[values(Verifier::BidirectionalTrie, Verifier::SmithWaterman)] verifier: Verifier,
    ) {
        // Substituting the second symbol turns "ac" into the query "ab" at a
        // distance of exactly one.
        let corpus = corpus(&["ac"]);
        let candidates = all_candidates(&["ac"], 2);

        let matches = verifier
            .verify(
                &symbols("ab"),
                &candidates,
                &corpus,
                Cost::new(threshold).unwrap(),
                &UnitCosts,
            )
            .unwrap();

        assert_eq!(
            matches
                .iter()
                .any(|matched| matched.token_range == (0..2) && matched.distance == 1.0),
            expected
        );
    }

    #[rstest]
    fn verification_keeps_finite_matches_next_to_unrepresentable_distances(
        #[values(Verifier::BidirectionalTrie, Verifier::SmithWaterman)] verifier: Verifier,
    ) {
        // Inserting "c" costs `Cost::MAX`, so the DP cell that only inserts it
        // is unrepresentable once the following "b" is inserted as well, while
        // the neighboring cell substituting "c" for the query symbol stays
        // finite. Both cells are reached: the unrepresentable one must not
        // suppress the ranges that remain within the threshold, and the ranges
        // that pass through it must not become matches.
        let corpus = corpus(&["acb"]);
        let candidates = all_candidates(&["acb"], 2);

        let matches = verifier
            .verify(
                &symbols("ab"),
                &candidates,
                &corpus,
                Cost::ONE,
                &CostPolicy::Unrepresentable,
            )
            .unwrap();

        assert_eq!(
            matches,
            [
                Match {
                    sequence_id: 0,
                    token_range: 0..1,
                    distance: 1.0,
                },
                Match {
                    sequence_id: 0,
                    token_range: 0..2,
                    distance: 1.0,
                },
                Match {
                    sequence_id: 0,
                    token_range: 1..3,
                    distance: 1.0,
                },
                Match {
                    sequence_id: 0,
                    token_range: 2..3,
                    distance: 1.0,
                },
            ]
        );
    }

    #[rstest]
    fn verification_rejects_a_threshold_without_a_strict_upper_bound(
        #[values(Verifier::BidirectionalTrie, Verifier::SmithWaterman)] verifier: Verifier,
    ) {
        let corpus = corpus(&["a"]);

        let result = verifier.verify(
            &symbols("a"),
            &[candidate(0, 0, 0)],
            &corpus,
            Cost::MAX,
            &UnitCosts,
        );

        assert_eq!(result, Err(Error::InvalidCost(f32::MAX)));
    }

    #[rstest]
    fn verification_rejects_unknown_string_before_returning_matches(
        #[values(Verifier::BidirectionalTrie, Verifier::SmithWaterman)] verifier: Verifier,
    ) {
        let corpus = corpus(&["a"]);

        let result = verifier.verify(
            &symbols("a"),
            &[candidate(0, 0, 0), candidate(1, 0, 0)],
            &corpus,
            Cost::ZERO,
            &UnitCosts,
        );

        assert_eq!(result, Err(Error::UnknownString(1)));
    }

    #[rstest]
    #[case::past_the_last_symbol("a", 1)]
    #[case::far_past_the_last_symbol("a", 7)]
    #[case::empty_string("", 0)]
    fn verification_rejects_out_of_bounds_string_position(
        #[case] text: &str,
        #[case] string_position: u32,
        #[values(Verifier::BidirectionalTrie, Verifier::SmithWaterman)] verifier: Verifier,
    ) {
        let corpus = corpus(&[text]);

        let result = verifier.verify(
            &symbols("a"),
            &[candidate(0, string_position, 0)],
            &corpus,
            Cost::ZERO,
            &UnitCosts,
        );

        assert_eq!(
            result,
            Err(Error::InvalidStringPosition {
                position: string_position as usize,
                string_len: text.len(),
            })
        );
    }

    #[rstest]
    #[case::past_the_last_symbol("a", 1)]
    #[case::far_past_the_last_symbol("a", 7)]
    #[case::empty_query_string("", 0)]
    fn verification_rejects_out_of_bounds_query_position(
        #[case] query_text: &str,
        #[case] query_position: u32,
        #[values(Verifier::BidirectionalTrie, Verifier::SmithWaterman)] verifier: Verifier,
    ) {
        let corpus = corpus(&["a"]);

        let result = verifier.verify(
            &symbols(query_text),
            &[candidate(0, 0, query_position)],
            &corpus,
            Cost::ZERO,
            &UnitCosts,
        );

        assert_eq!(
            result,
            Err(Error::InvalidQueryPosition {
                position: query_position as usize,
                query_len: query_text.len(),
            })
        );
    }

    #[rstest]
    #[case(0.0, 0.0, 0.0)]
    #[case(0.25, 0.5, 0.75)]
    #[case(f32::MAX, 0.0, f32::MAX)]
    #[case(f32::MAX / 2.0, f32::MAX / 2.0, f32::MAX)]
    #[case(f32::MAX, f32::MIN_POSITIVE, f32::INFINITY)]
    #[case(f32::MAX, 1.0, f32::INFINITY)]
    #[case(f32::MAX, f32::MAX, f32::INFINITY)]
    #[case(f32::INFINITY, 1.0, f32::INFINITY)]
    fn distance_addition_does_not_saturate_at_maximum(
        #[case] left: f32,
        #[case] right: f32,
        #[case] expected: f32,
    ) {
        assert_eq!(add_distance(left, right), expected);
    }
}
