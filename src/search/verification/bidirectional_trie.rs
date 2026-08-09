//! Anchor-local verification with bidirectional trie caches.

use std::collections::BTreeMap;
use std::marker::PhantomData;

use super::{Verifier, add_distance, root_column, step_dp, validated_candidate_data};
use crate::corpus::CorpusStore;
use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::search::{Candidate, Match};
use crate::types::{Position, StringId};

/// Local verification with query-position-specific forward and backward tries.
///
/// The verifier itself is stateless; all caches live for a single
/// verification call.
#[derive(Debug, Clone)]
pub(in crate::search) struct BidirectionalTrieVerifier<Symbol> {
    marker: PhantomData<fn() -> Symbol>,
}

struct TrieNode<Symbol> {
    // `column` is the WED state after consuming the edge labels from the root
    // through this node. A child therefore needs exactly one `step_dp` call.
    column: Vec<f32>,
    // Labels are owned so a disk-backed store may release each decoded
    // document after processing one candidate.
    children: Vec<(Symbol, TrieNode<Symbol>)>,
}

impl<Symbol> TrieNode<Symbol> {
    fn root<Costs>(query: &[&Symbol], costs: &Costs) -> Self
    where
        Costs: EditCosts<Symbol>,
    {
        Self {
            column: root_column(query, costs),
            children: Vec::new(),
        }
    }
}

struct TrieForest<Symbol> {
    // Query position is part of the cache key because it determines the query
    // prefix/suffix represented by every DP row. Direction is represented by
    // separate maps so backward and forward columns can never be mixed.
    backward: BTreeMap<usize, TrieNode<Symbol>>,
    forward: BTreeMap<usize, TrieNode<Symbol>>,
}

impl<Symbol> Default for TrieForest<Symbol> {
    fn default() -> Self {
        Self {
            backward: BTreeMap::new(),
            forward: BTreeMap::new(),
        }
    }
}

struct DirectionalQueries<'query, Symbol> {
    // The backward data iterator is also reversed below. Reversing both sides
    // preserves wed(query, data), including asymmetric insertion/deletion.
    backward: Vec<&'query Symbol>,
    forward: Vec<&'query Symbol>,
}

impl<'query, Symbol> DirectionalQueries<'query, Symbol> {
    fn new(query: &'query [Symbol], query_position: usize) -> Self {
        Self {
            backward: query[..query_position].iter().rev().collect(),
            forward: query[query_position + 1..].iter().collect(),
        }
    }
}

fn cached_prefix_distances<'query, 'data, Symbol, Costs, Data>(
    query: &[&'query Symbol],
    data: Data,
    budget: f32,
    root: &mut TrieNode<Symbol>,
    costs: &Costs,
) -> Vec<f32>
where
    Symbol: Clone + PartialEq + 'query + 'data,
    Costs: EditCosts<Symbol>,
    Data: IntoIterator<Item = &'data Symbol>,
{
    let mut node = root;
    // Index zero denotes the empty data prefix. It must be retained so an
    // answer may begin or end exactly at the candidate anchor.
    let mut distances = vec![node.column[query.len()]];

    for data_symbol in data {
        // The path label identifies the processed data prefix. Following an
        // existing edge reuses its complete DP column; a missing edge advances
        // the parent column once and caches the result for later candidates.
        let child_index = node
            .children
            .iter()
            .position(|(symbol, _)| symbol == data_symbol);

        let index = match child_index {
            Some(index) => index,
            None => {
                let column = step_dp(query, data_symbol, &node.column, costs);
                node.children.push((
                    data_symbol.clone(),
                    TrieNode {
                        column,
                        children: Vec::new(),
                    },
                ));
                node.children.len() - 1
            }
        };

        node = &mut node.children[index].1;

        // A cached DP column is candidate-independent, but the remaining
        // strict budget is not. Recheck the lower bound on every traversal.
        if node.column.iter().copied().fold(f32::INFINITY, f32::min) >= budget {
            break;
        }
        distances.push(node.column[query.len()]);
    }

    distances
}

impl<Symbol> BidirectionalTrieVerifier<Symbol> {
    /// Creates a verifier. Its caches are populated per verification call.
    pub(in crate::search) const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<Symbol> Default for BidirectionalTrieVerifier<Symbol> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Symbol> Verifier<Symbol> for BidirectionalTrieVerifier<Symbol>
where
    Symbol: Clone + PartialEq,
{
    fn verify<Costs, Store>(
        &self,
        query: &[Symbol],
        candidates: &[Candidate],
        corpus: &Store,
        threshold: Cost,
        costs: &Costs,
    ) -> Result<Vec<Match>>
    where
        Costs: EditCosts<Symbol>,
        Store: CorpusStore<Symbol>,
    {
        let threshold = threshold.next_up()?;
        for candidate in candidates {
            validated_candidate_data(query, candidate, corpus)?;
        }

        // Both caches are call-local deliberately: a DP column depends on this
        // query and cost policy, so sharing it across calls would be unsound.
        let mut forest = TrieForest::<Symbol>::default();
        let mut directional_queries = BTreeMap::<usize, DirectionalQueries<Symbol>>::new();
        let mut intervals = BTreeMap::<(StringId, usize, usize), f32>::new();

        for candidate in candidates {
            // Candidates were validated before cache construction, so an
            // error cannot leave misleading partial statistics behind.
            let data = corpus
                .sequence(candidate.string_id)?
                .ok_or(Error::UnknownString(candidate.string_id))?;
            let data = data.as_ref();
            let query_position = candidate.query_position.as_usize();
            let data_position = candidate.data_position.as_usize();
            let anchor_cost = costs
                .substitution(&query[query_position], &data[data_position])
                .get();
            if anchor_cost >= threshold {
                continue;
            }

            let budget = threshold - anchor_cost;
            // Building these reference vectors is O(|Q|), so memoize them per
            // query position instead of repeating the allocation per candidate.
            let directional_query = directional_queries
                .entry(query_position)
                .or_insert_with(|| DirectionalQueries::new(query, query_position));

            let backward_root = match forest.backward.entry(query_position) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(TrieNode::root(&directional_query.backward, costs))
                }
            };
            let backward = cached_prefix_distances(
                &directional_query.backward,
                // Distance from the anchor increases while indices decrease,
                // hence the data prefix must be visited in reverse as well.
                data[..data_position].iter().rev(),
                budget,
                backward_root,
                costs,
            );

            let forward_root = match forest.forward.entry(query_position) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(TrieNode::root(&directional_query.forward, costs))
                }
            };
            let forward = cached_prefix_distances(
                &directional_query.forward,
                data[data_position + 1..].iter(),
                budget,
                forward_root,
                costs,
            );

            for (backward_len, backward_distance) in backward.iter().copied().enumerate() {
                for (forward_len, forward_distance) in forward.iter().copied().enumerate() {
                    // The two directional edit sequences meet at the forced
                    // anchor substitution, so their costs add independently.
                    let distance = add_distance(
                        add_distance(anchor_cost, backward_distance),
                        forward_distance,
                    );
                    if distance < threshold {
                        let start = data_position - backward_len;
                        let end = data_position + forward_len + 1;
                        intervals
                            .entry((candidate.string_id, start, end))
                            .and_modify(|stored| *stored = (*stored).min(distance))
                            .or_insert(distance);
                    }
                }
            }
        }

        let matches = intervals
            .into_iter()
            .map(|((string_id, start, end), distance)| {
                Ok(Match {
                    string_id,
                    range: Position::from_usize(start)?..Position::from_usize(end)?,
                    distance: Cost::new(distance)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(matches)
    }
}
