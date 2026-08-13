use yurine::costs::EditCosts;
use yurine::errors::Error;
use yurine::{Cost, Match, RangeSearchParams, SearchEngine, SearchEngineBuilder};

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
    assert_range_search_result_contract(build_compatibility_engine());
}

#[cfg(feature = "persist")]
#[test]
fn mmap_range_search_result_contract_is_stable() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("compatibility.yurine");
    build_compatibility_engine()
        .save_with(&path, &yurine::persistence::CharCodec)
        .unwrap();
    assert_range_search_result_contract(
        SearchEngine::open_with(&path, &yurine::persistence::CharCodec).unwrap(),
    );
}

fn build_compatibility_engine() -> SearchEngine<char> {
    let mut builder = SearchEngineBuilder::new();
    for source_text in ["x東京y", "東京", "東亰", "東京東京", "京都"] {
        builder.add_sequence(source_text.chars()).unwrap();
    }
    builder.build().unwrap()
}

fn assert_range_search_result_contract(engine: SearchEngine<char>) {
    let searcher = engine.range_searcher(CompatibilityCosts);

    let matches = searcher
        .search(&['東', '京'], &RangeSearchParams::new(0.25).with_eta(0.25))
        .unwrap();

    assert_eq!(
        matches,
        [
            Match {
                sequence_id: 0,
                token_range: 1..3,
                distance: 0.0,
            },
            Match {
                sequence_id: 1,
                token_range: 0..2,
                distance: 0.0,
            },
            Match {
                sequence_id: 2,
                token_range: 0..2,
                distance: 0.25,
            },
            Match {
                sequence_id: 3,
                token_range: 0..2,
                distance: 0.0,
            },
            Match {
                sequence_id: 3,
                token_range: 2..4,
                distance: 0.0,
            },
        ]
    );
}

#[test]
fn repeated_postings_report_candidate_count_without_deduplication() {
    let mut builder = SearchEngineBuilder::new();
    for _ in 0..128 {
        builder.add_sequence("aaaaaaaa".chars()).unwrap();
    }
    let engine = builder.build().unwrap();
    let searcher = engine.range_searcher(CompatibilityCosts);

    let (_, metrics) = searcher
        .search_with_metrics(&['a', 'a'], &RangeSearchParams::new(1.0))
        .unwrap();

    assert_eq!(metrics.selected_query_positions, 2);
    assert_eq!(metrics.generated_candidates, 2 * 128 * 8);
}

#[test]
fn exhaustive_fallback_result_contract_is_stable() {
    let mut builder = SearchEngineBuilder::new();
    builder.add_sequence(['a']).unwrap();
    builder.add_sequence([]).unwrap();
    let engine = builder.build().unwrap();
    let searcher = engine.range_searcher(CompatibilityCosts);

    let (matches, metrics) = searcher
        .search_with_metrics(&['a'], &RangeSearchParams::new(1.0))
        .unwrap();

    assert!(metrics.used_exhaustive_verification);
    assert_eq!(
        matches,
        [Match {
            sequence_id: 0,
            token_range: 0..1,
            distance: 0.0,
        }]
    );
}

#[test]
fn empty_query_sequence_error_contract_is_stable() {
    let mut builder = SearchEngineBuilder::new();
    builder.add_sequence(['a']).unwrap();
    let engine = builder.build().unwrap();
    let searcher = engine.range_searcher(CompatibilityCosts);

    // This intentionally fixes the current error variant. Replacing it with a
    // dedicated empty-query error should be treated as an explicit API change.
    assert_eq!(
        searcher.search(&[], &RangeSearchParams::new(0.0)),
        Err(Error::ThresholdSubsequenceUnavailable)
    );
}
