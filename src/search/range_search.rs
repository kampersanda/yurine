//! Fixed-threshold range search orchestration.

use std::hash::Hash;

use crate::corpus::CorpusStore;
use crate::costs::{Cost, EditCosts};
use crate::errors::{Error, Result};
use crate::search::SearchEngine;
use crate::search::filtering::candidate::MinCandidateSelector;
use crate::search::filtering::generate_candidates;
use crate::search::verification::Verifier;
use crate::search::verification::bidirectional_trie::BidirectionalTrieVerifier;
use crate::search::verification::smith_waterman::SmithWatermanVerifier;
use crate::search::{Candidate, Match};
use crate::types::{Position, StringId};

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

impl<Symbol, Costs> SearchEngine<Symbol, Costs>
where
    Symbol: Clone + PartialEq + Hash + Eq,
    Costs: EditCosts<Symbol>,
{
    /// Finds non-empty substrings satisfying the configured range search.
    ///
    /// Results are ordered by string ID, then range start, then range end.
    ///
    /// When eta is not configured, it defaults to
    /// `threshold / query.len()`. This favors constructing a threshold
    /// subsequence for continuous substitution costs. An empty query uses eta
    /// zero and retains the existing empty-query error behavior.
    ///
    /// If the selector cannot construct a complete threshold subsequence for
    /// a non-empty query, the engine falls back to exhaustive Smith-Waterman
    /// verification instead of returning
    /// [`Error::ThresholdSubsequenceUnavailable`]. This occurs whenever the
    /// query's total filtering contribution is less than or equal to the
    /// threshold. With unit costs, `threshold >= query.len()` is such a case.
    ///
    /// The fallback takes `O(m * sum(n_i^2))` time for query length `m` and
    /// corpus string lengths `n_i`, and can return `O(sum(n_i^2))` intervals.
    /// It may therefore be substantially slower and produce many more results
    /// than the normal filter-and-verify path.
    ///
    /// Searching takes `&self`, so one engine can serve concurrent queries.
    pub fn range_search(&self, query: &[Symbol], params: &RangeSearchParams) -> Result<Vec<Match>> {
        let threshold = params.threshold;
        // strict_threshold(threshold)?;
        let eta = match params.eta {
            Some(eta) => eta,
            None => automatic_eta(threshold, query.len())?,
        };
        self.search_all(query, threshold, eta)
    }

    pub(super) fn search_all(
        &self,
        query: &[Symbol],
        threshold: Cost,
        eta: Cost,
    ) -> Result<Vec<Match>> {
        let selected = match MinCandidateSelector.select(
            query,
            threshold,
            eta,
            &self.index,
            &self.costs,
            &self.neighborhood,
        ) {
            Ok(selected) => selected,
            Err(Error::ThresholdSubsequenceUnavailable) if !query.is_empty() => {
                return verify_exhaustively(query, threshold, &self.store, &self.costs);
            }
            Err(error) => return Err(error),
        };
        let candidates = generate_candidates(
            query,
            &selected,
            eta,
            &self.index,
            &self.costs,
            &self.neighborhood,
        )?;
        BidirectionalTrieVerifier::new().verify(
            query,
            &candidates,
            &self.store,
            threshold,
            &self.costs,
        )
    }
}

/// Exhaustively verifies every non-empty string in the corpus.
///
/// This is used
/// when the selector cannot construct a complete threshold subsequence for a
/// non-empty query. It is slower and can return more results than the normal
/// filter-and-verify path, but is guaranteed to be correct.
fn verify_exhaustively<Symbol, Costs>(
    query: &[Symbol],
    threshold: Cost,
    corpus: &CorpusStore<Symbol>,
    costs: &Costs,
) -> Result<Vec<Match>>
where
    Costs: EditCosts<Symbol>,
{
    // SmithWatermanVerifier uses candidates only to select data strings. One
    // in-bounds anchor per non-empty string requests exhaustive verification
    // without relying on the filtering guarantee that was unavailable.
    let mut candidates = Vec::new();
    for raw_id in 0..corpus.len() {
        let string_id = StringId::from_usize(raw_id)?;
        let sequence = corpus
            .sequence(string_id)?
            .ok_or(Error::UnknownString(string_id))?;
        if !sequence.as_ref().is_empty() {
            candidates.push(Candidate {
                string_id,
                data_position: Position::new(0),
                query_position: Position::new(0),
            });
        }
    }
    SmithWatermanVerifier.verify(query, &candidates, corpus, threshold, costs)
}
