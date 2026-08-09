mod filtering;
mod verification;

use std::ops::Range;

use crate::costs::Cost;
use crate::types::{Position, StringId};

/// A candidate match of a query in a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Candidate {
    string_id: StringId,
    data_position: Position,
    query_position: Position,
}

/// A verified substring satisfying the inclusive distance threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// The data string containing the match.
    pub string_id: StringId,
    /// The matched zero-based, end-exclusive symbol range.
    pub range: Range<Position>,
    /// The weighted edit distance from the query to the substring.
    pub distance: Cost,
}
