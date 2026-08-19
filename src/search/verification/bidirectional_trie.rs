//! Anchor-local verification with bidirectional trie caches.
//!
//! Every candidate anchor forces one substitution between a query symbol and a
//! data symbol, and the two directions around it are extended independently.
//! This enumerates exactly the alignments pairing at least one query symbol
//! with one data symbol, and no others.
//!
//! Those are all the alignments that matter when
//! `substitution(from, to) <= deletion(from) + insertion(to)`. An alignment
//! pairing no symbols deletes the whole query string and inserts the whole
//! substring; replacing one such deletion and insertion by a substitution
//! leaves the distance no larger, so an optimal alignment pairing at least one
//! symbol always exists.
//!
//! Cost policies are not required to satisfy that inequality, and this verifier
//! reports a larger distance than the weighted edit distance when they do not.
//! Range search stays exact because it rejects every search that could reach an
//! alignment pairing no symbols: such an alignment deletes the whole query
//! string, so it costs at least the sum of those deletions, and a threshold
//! reaching that sum is exactly what `RangeSearcher::search_with_metrics`
//! declines up front. A caller of this module has to establish the same
//! precondition.

use std::collections::BTreeMap;

use hashbrown::HashMap;

use super::{add_distance, create_match, root_column, step_dp, validated_candidate_string};
use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::search::bound::StrictBound;
use crate::search::{Candidate, Match};
use crate::store::CorpusStore;
use crate::types::{SequenceId, Symbol};

struct TrieNode {
    // `column` is the WED state after consuming the edge labels from the root
    // through this node. A child therefore needs exactly one `step_dp` call.
    column: Vec<f32>,
    // Labels are owned so a disk-backed store may release each decoded
    // data string after processing one candidate.
    children: HashMap<Symbol, TrieNode>,
}

impl TrieNode {
    fn root<C>(query_string: &[Symbol], costs: &C) -> Self
    where
        C: EditCosts<Symbol>,
    {
        Self {
            column: root_column(query_string, costs),
            children: HashMap::new(),
        }
    }
}

#[derive(Default)]
struct TrieForest {
    // Query-string position is part of the cache key because it determines the
    // query-string prefix/suffix represented by every DP row. Direction is
    // represented by separate maps so backward and forward columns can never
    // be mixed.
    backward: BTreeMap<usize, TrieNode>,
    forward: BTreeMap<usize, TrieNode>,
}

struct DirectionalQueryStrings {
    // The backward data-string iterator is also reversed below. Reversing both
    // sides preserves wed(query string, data string), including asymmetric
    // insertion/deletion.
    backward: Vec<Symbol>,
    forward: Vec<Symbol>,
}

impl DirectionalQueryStrings {
    fn new(query_string: &[Symbol], query_position: usize) -> Self {
        Self {
            backward: query_string[..query_position]
                .iter()
                .rev()
                .copied()
                .collect(),
            forward: query_string[query_position + 1..].to_vec(),
        }
    }
}

fn cached_prefix_distances<C, I>(
    query_string: &[Symbol],
    string_symbols: I,
    budget: f32,
    root: &mut TrieNode,
    costs: &C,
) -> Vec<f32>
where
    C: EditCosts<Symbol>,
    I: IntoIterator<Item = Symbol>,
{
    let mut node = root;
    // Index zero denotes the empty string prefix. It must be retained so an
    // answer may begin or end exactly at the candidate anchor.
    let mut distances = vec![node.column[query_string.len()]];

    for string_symbol in string_symbols {
        // The path label identifies the processed string prefix. Following an
        // existing edge reuses its complete DP column; a missing edge advances
        // the parent column once and caches the result for later candidates.
        // Splitting the borrow lets one lookup serve both cases.
        let TrieNode { column, children } = node;
        node = children.entry(string_symbol).or_insert_with(|| TrieNode {
            column: step_dp(query_string, string_symbol, column, costs),
            children: HashMap::new(),
        });

        // A cached DP column is candidate-independent, but the remaining
        // strict budget is not. Recheck the lower bound on every traversal.
        if node.column.iter().copied().fold(f32::INFINITY, f32::min) >= budget {
            break;
        }
        distances.push(node.column[query_string.len()]);
    }

    distances
}

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
    for candidate in candidates {
        validated_candidate_string(query_string, candidate, corpus)?;
    }

    // Both caches are call-local deliberately: a DP column depends on this
    // query string and cost policy, so sharing it across calls would be unsound.
    let mut forest = TrieForest::default();
    let mut directional_query_strings_by_position =
        BTreeMap::<usize, DirectionalQueryStrings>::new();
    let mut substrings = BTreeMap::<(SequenceId, usize, usize), f32>::new();

    for candidate in candidates {
        // Candidates were validated before cache construction, so an
        // error cannot leave misleading partial statistics behind.
        let string = corpus
            .string(candidate.string_id)?
            .ok_or(Error::UnknownString(candidate.string_id.as_usize()))?;
        let query_position = candidate.query_position.as_usize();
        let string_position = candidate.string_position.as_usize();
        let anchor_cost = costs
            .substitution(&query_string[query_position], &string[string_position])
            .get();
        if !bound.admits(anchor_cost) {
            continue;
        }

        let budget = bound.residual(anchor_cost);
        // Building these reference vectors is O(|Q|), so memoize them per
        // query-string position instead of repeating the allocation per
        // candidate.
        let directional_query_strings = directional_query_strings_by_position
            .entry(query_position)
            .or_insert_with(|| DirectionalQueryStrings::new(query_string, query_position));

        let backward_root = match forest.backward.entry(query_position) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(TrieNode::root(&directional_query_strings.backward, costs))
            }
        };
        let backward = cached_prefix_distances(
            &directional_query_strings.backward,
            // Distance from the anchor increases while indices decrease,
            // hence the string prefix must be visited in reverse as well.
            string[..string_position].iter().rev().copied(),
            budget,
            backward_root,
            costs,
        );

        let forward_root = match forest.forward.entry(query_position) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(TrieNode::root(&directional_query_strings.forward, costs))
            }
        };
        let forward = cached_prefix_distances(
            &directional_query_strings.forward,
            string[string_position + 1..].iter().copied(),
            budget,
            forward_root,
            costs,
        );

        for (backward_symbol_count, backward_distance) in backward.iter().copied().enumerate() {
            for (forward_symbol_count, forward_distance) in forward.iter().copied().enumerate() {
                // The two directional edit sequences meet at the forced
                // anchor substitution, so their costs add independently.
                let distance = add_distance(
                    add_distance(anchor_cost, backward_distance),
                    forward_distance,
                );
                if bound.admits(distance) {
                    let symbol_start = string_position - backward_symbol_count;
                    let symbol_end = string_position + forward_symbol_count + 1;
                    substrings
                        .entry((candidate.string_id, symbol_start, symbol_end))
                        .and_modify(|stored| *stored = (*stored).min(distance))
                        .or_insert(distance);
                }
            }
        }
    }

    let matches = substrings
        .into_iter()
        .map(|((string_id, symbol_start, symbol_end), distance)| {
            Ok(create_match(
                string_id,
                symbol_start,
                symbol_end,
                Cost::new(distance)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(matches)
}
