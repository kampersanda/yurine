use yurine::costs::{Cost, EditCosts};
use yurine::errors::Error;
use yurine::search::SearchEngineBuilder;
use yurine::search::range_search::RangeSearchParams;
use yurine::tokenization::character::CharacterTokenizer;
use yurine::types::{Position, StringId};

struct CompatibilityCosts;

impl EditCosts<char> for CompatibilityCosts {
    fn substitution(&self, from: &char, to: &char) -> Cost {
        if from == to {
            Cost::ZERO
        } else if (*from, *to) == ('京', '亰') {
            Cost::new_const(0.25)
        } else {
            Cost::ONE
        }
    }

    fn deletion(&self, _token: &char) -> Cost {
        Cost::ONE
    }

    fn insertion(&self, _token: &char) -> Cost {
        Cost::ONE
    }
}

#[test]
fn range_search_result_contract_is_stable() {
    let mut builder = SearchEngineBuilder::new(CharacterTokenizer::new(), CompatibilityCosts);
    for string in ["x東京y", "東京", "東亰", "東京東京", "京都"] {
        builder.add_string(string).unwrap();
    }
    let engine = builder.build().unwrap();

    let matches = engine
        .range_search(
            "東京",
            &RangeSearchParams::new(Cost::new_const(0.25)).with_eta(Cost::new_const(0.25)),
        )
        .unwrap();

    assert_eq!(
        matches,
        [
            yurine::search::Match {
                string_id: StringId::new(0),
                token_range: Position::new(1)..Position::new(3),
                byte_range: 1..7,
                distance: Cost::ZERO,
            },
            yurine::search::Match {
                string_id: StringId::new(1),
                token_range: Position::new(0)..Position::new(2),
                byte_range: 0..6,
                distance: Cost::ZERO,
            },
            yurine::search::Match {
                string_id: StringId::new(2),
                token_range: Position::new(0)..Position::new(2),
                byte_range: 0..6,
                distance: Cost::new_const(0.25),
            },
            yurine::search::Match {
                string_id: StringId::new(3),
                token_range: Position::new(0)..Position::new(2),
                byte_range: 0..6,
                distance: Cost::ZERO,
            },
            yurine::search::Match {
                string_id: StringId::new(3),
                token_range: Position::new(2)..Position::new(4),
                byte_range: 6..12,
                distance: Cost::ZERO,
            },
        ]
    );
}

#[test]
fn repeated_postings_report_candidate_count_without_deduplication() {
    let mut builder = SearchEngineBuilder::new(CharacterTokenizer::new(), CompatibilityCosts);
    for _ in 0..128 {
        builder.add_string("aaaaaaaa").unwrap();
    }
    let engine = builder.build().unwrap();

    let (_, metrics) = engine
        .range_search_with_metrics("aa", &RangeSearchParams::new(Cost::ONE))
        .unwrap();

    assert_eq!(metrics.selected_query_positions, 2);
    assert_eq!(metrics.generated_candidates, 2 * 128 * 8);
}

#[test]
fn exhaustive_fallback_result_contract_is_stable() {
    let mut builder = SearchEngineBuilder::new(CharacterTokenizer::new(), CompatibilityCosts);
    builder.add_string("a").unwrap();
    builder.add_string("").unwrap();
    let engine = builder.build().unwrap();

    let (matches, metrics) = engine
        .range_search_with_metrics("a", &RangeSearchParams::new(Cost::ONE))
        .unwrap();

    assert!(metrics.used_exhaustive_verification);
    assert_eq!(
        matches,
        [yurine::search::Match {
            string_id: StringId::new(0),
            token_range: Position::new(0)..Position::new(1),
            byte_range: 0..1,
            distance: Cost::ZERO,
        }]
    );
}

#[test]
fn empty_query_error_contract_is_stable() {
    let mut builder = SearchEngineBuilder::new(CharacterTokenizer::new(), CompatibilityCosts);
    builder.add_string("a").unwrap();
    let engine = builder.build().unwrap();

    // This intentionally fixes the current error variant. Replacing it with a
    // dedicated empty-query error should be treated as an explicit API change.
    assert_eq!(
        engine.range_search("", &RangeSearchParams::new(Cost::ZERO)),
        Err(Error::ThresholdSubsequenceUnavailable)
    );
}
