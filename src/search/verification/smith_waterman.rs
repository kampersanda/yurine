//! Exhaustive Smith-Waterman-based verification.

use std::collections::BTreeSet;

use super::{add_distance, create_match, validated_candidate_string};
use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::search::bound::StrictBound;
use crate::search::{Candidate, Match};
use crate::store::CorpusStore;
use crate::types::{SequenceId, Symbol};

/// Exact Smith-Waterman-based verification that preserves every start position.
///
/// Unlike the usual `O(mn)` semi-global variant, this verifier's contract is to
/// enumerate every substring the bound admits. Its intended baseline
/// implementation therefore uses `O(n^2 m)` time and `O(m)` DP working space,
/// as described in `docs/development/smith-waterman-verification.md`. It
/// additionally uses `O(u)` space to deduplicate the `u` candidate-referenced
/// data strings.
///
/// Candidate anchors select data strings, but do not localize the baseline DP.
/// Each selected string is exhaustively verified once. Candidate string IDs,
/// string positions, and query positions are validated before verification.
///
/// Measuring every alignment rather than only the anchored ones is also what
/// makes this the fallback for cost policies that substitute more expensively
/// than they delete and insert; see the [`bidirectional_trie`] module
/// documentation.
///
/// [`bidirectional_trie`]: super::bidirectional_trie
pub(super) fn verify<C>(
    query_string: &[Symbol],
    candidates: &[Candidate],
    corpus: &CorpusStore,
    bound: StrictBound,
    costs: &C,
) -> Result<Vec<Match>>
where
    C: EditCosts<Symbol>,
{
    let string_ids = validated_candidate_strings(query_string, candidates, corpus)?;
    let mut matches = Vec::new();

    for string_id in string_ids {
        let string = corpus
            .string(string_id)?
            .ok_or(Error::UnknownString(string_id.as_usize()))?;
        enumerate_substring_matches(query_string, string, string_id, bound, costs, &mut matches)?;
    }

    Ok(matches)
}

/// Validates that every candidate's string ID is known and positions are within bounds.
/// Returns the unique string IDs referenced by the candidates.
fn validated_candidate_strings(
    query_string: &[Symbol],
    candidates: &[Candidate],
    corpus: &CorpusStore,
) -> Result<BTreeSet<SequenceId>> {
    let mut string_ids = BTreeSet::new();
    for candidate in candidates {
        validated_candidate_string(query_string, candidate, corpus)?;
        string_ids.insert(candidate.string_id);
    }
    Ok(string_ids)
}

/// Enumerates every non-empty substring of `string` whose distance from
/// `query_string` is admitted by `bound`. Each match is pushed to `matches`.
fn enumerate_substring_matches<C>(
    query_string: &[Symbol],
    string: &[Symbol],
    string_id: SequenceId,
    bound: StrictBound,
    costs: &C,
    matches: &mut Vec<Match>,
) -> Result<()>
where
    C: EditCosts<Symbol>,
{
    // The two DP columns are reused across all O(n^2) cells of one data
    // string; only their contents are rewritten below.
    let mut previous = Vec::with_capacity(query_string.len() + 1);
    let mut current = Vec::with_capacity(query_string.len() + 1);

    // Fix the substring start so that each non-empty substring is considered
    // exactly once. Starts and ends increase monotonically, which also gives
    // deterministic range ordering within a data string.
    for symbol_start in 0..string.len() {
        // `previous[i]` is wed(query_string[..i],
        // string[symbol_start..symbol_end)), where the data-string prefix is
        // initially empty. Matching a query-string prefix against an empty
        // data string deletes every query symbol.
        previous.clear();
        previous.push(0.0);
        for query_symbol in query_string {
            let deletion = add_distance(
                previous.last().copied().unwrap_or(0.0),
                costs.deletion(query_symbol).get(),
            );
            previous.push(deletion);
        }

        for (symbol_end, string_symbol) in string.iter().enumerate().skip(symbol_start) {
            // `current[0]` matches an empty query string against
            // string[symbol_start..=symbol_end], so the newly consumed string
            // symbol is an insertion.
            current.clear();
            current.push(add_distance(
                previous[0],
                costs.insertion(string_symbol).get(),
            ));

            for (query_index, query_symbol) in query_string.iter().enumerate() {
                // Extend the DP column for
                // string[symbol_start..=symbol_end]. The direction is always
                // query string -> data substring: substitution consumes both
                // symbols, deletion consumes only the query symbol, and
                // insertion consumes only the string symbol.
                let substitution = add_distance(
                    previous[query_index],
                    costs.substitution(query_symbol, string_symbol).get(),
                );
                let deletion =
                    add_distance(current[query_index], costs.deletion(query_symbol).get());
                let insertion = add_distance(
                    previous[query_index + 1],
                    costs.insertion(string_symbol).get(),
                );
                current.push(substitution.min(deletion).min(insertion));
            }

            // The final cell is
            // wed(query_string, string[symbol_start..=symbol_end]). Convert the
            // inclusive `symbol_end` used by this loop to the public
            // end-exclusive token range.
            if bound.admits(current[query_string.len()]) {
                matches.push(create_match(
                    string_id,
                    symbol_start,
                    symbol_end + 1,
                    Cost::new(current[query_string.len()])?,
                ));
            }
            std::mem::swap(&mut previous, &mut current);
        }
    }
    Ok(())
}
