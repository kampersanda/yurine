//! Exhaustive Smith-Waterman-based verification.

use std::collections::BTreeSet;

use super::{add_distance, create_match, validated_candidate_string};
use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::search::{Candidate, Match};
use crate::store::CorpusStore;
use crate::types::{StringId, Symbol};

/// Exact Smith-Waterman-based verification that preserves every start position.
///
/// Unlike the usual `O(mn)` semi-global variant, this verifier's contract is to
/// enumerate every interval below the threshold. Its intended baseline
/// implementation therefore uses `O(n^2 m)` time and `O(m)` DP working space,
/// as described in `docs/development/smith-waterman-verification.md`. It
/// additionally uses `O(u)` space to deduplicate the `u` candidate-referenced
/// corpus strings.
///
/// Candidate anchors select corpus strings, but do not localize the baseline DP.
/// Each selected string is exhaustively verified once. Candidate string IDs,
/// corpus positions, and query positions are validated before verification.
pub(super) fn verify<C>(
    query: &[Symbol],
    candidates: &[Candidate],
    corpus: &CorpusStore,
    threshold: Cost,
    costs: &C,
) -> Result<Vec<Match>>
where
    C: EditCosts<Symbol>,
{
    let threshold = threshold.next_up()?;
    let string_ids = validated_candidate_strings(query, candidates, corpus)?;
    let mut matches = Vec::new();

    for string_id in string_ids {
        let string = corpus
            .string(string_id)?
            .ok_or(Error::UnknownString(string_id))?;
        enumerate_matches(query, string, string_id, threshold, costs, &mut matches)?;
    }

    Ok(matches)
}

/// Validates that every candidate's string ID is known and positions are within bounds.
/// Returns the unique string IDs referenced by the candidates.
fn validated_candidate_strings(
    query: &[Symbol],
    candidates: &[Candidate],
    corpus: &CorpusStore,
) -> Result<BTreeSet<StringId>> {
    let mut string_ids = BTreeSet::new();
    for candidate in candidates {
        validated_candidate_string(query, candidate, corpus)?;
        string_ids.insert(candidate.string_id);
    }
    Ok(string_ids)
}

/// Enumerates every non-empty substring of `string` whose distance from `query` is
/// strictly less than `threshold`. Each match is pushed to `matches`.
fn enumerate_matches<C>(
    query: &[Symbol],
    string: &[Symbol],
    string_id: StringId,
    threshold: Cost,
    costs: &C,
    matches: &mut Vec<Match>,
) -> Result<()>
where
    C: EditCosts<Symbol>,
{
    // The two DP columns are reused across all O(n^2) cells of one corpus
    // string; only their contents are rewritten below.
    let mut previous = Vec::with_capacity(query.len() + 1);
    let mut current = Vec::with_capacity(query.len() + 1);

    // Fix the substring start so that each non-empty interval is considered
    // exactly once. Starts and ends increase monotonically, which also gives
    // deterministic range ordering within a corpus string.
    for start in 0..string.len() {
        // `previous[i]` is wed(query[..i], string[start..end)), where the corpus
        // prefix is initially empty. Matching a query prefix against an empty
        // corpus string deletes every query symbol.
        previous.clear();
        previous.push(0.0);
        for query_symbol in query {
            let deletion = add_distance(
                previous.last().copied().unwrap_or(0.0),
                costs.deletion(query_symbol).get(),
            );
            previous.push(deletion);
        }

        for (end, corpus_symbol) in string.iter().enumerate().skip(start) {
            // `current[0]` matches an empty query against string[start..=end],
            // so the newly consumed corpus symbol is an insertion.
            current.clear();
            current.push(add_distance(
                previous[0],
                costs.insertion(corpus_symbol).get(),
            ));

            for (query_index, query_symbol) in query.iter().enumerate() {
                // Extend the DP column for string[start..=end]. The direction is
                // always query -> corpus substring: substitution consumes both
                // symbols, deletion consumes only the query symbol, and
                // insertion consumes only the corpus symbol.
                let substitution = add_distance(
                    previous[query_index],
                    costs.substitution(query_symbol, corpus_symbol).get(),
                );
                let deletion =
                    add_distance(current[query_index], costs.deletion(query_symbol).get());
                let insertion = add_distance(
                    previous[query_index + 1],
                    costs.insertion(corpus_symbol).get(),
                );
                current.push(substitution.min(deletion).min(insertion));
            }

            // The final cell is wed(query, string[start..=end]). Convert the
            // inclusive `end` used by this loop to the public end-exclusive
            // range. The internal threshold is the strict upper bound
            // immediately above the public inclusive threshold.
            if current[query.len()] < threshold {
                matches.push(create_match(
                    string_id,
                    start,
                    end + 1,
                    Cost::new(current[query.len()])?,
                )?);
            }
            std::mem::swap(&mut previous, &mut current);
        }
    }
    Ok(())
}
