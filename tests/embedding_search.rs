use std::num::NonZeroUsize;

use approx::assert_abs_diff_eq;
use yurine::costs::{CosineEmbeddingCosts, EmbeddingStoreBuilder};
use yurine::{RangeSearchParams, SearchEngineBuilder};

#[test]
fn query_only_token_uses_embedding_for_candidate_generation_and_verification() {
    let mut embeddings = EmbeddingStoreBuilder::new(NonZeroUsize::new(2).unwrap());
    embeddings.insert('x', vec![1.0, 0.0]).unwrap();
    embeddings.insert('あ', vec![0.8, 0.6]).unwrap();
    embeddings.insert('b', vec![0.0, 1.0]).unwrap();

    let costs = CosineEmbeddingCosts::new(embeddings.build());
    let mut builder = SearchEngineBuilder::new();
    builder.add_sequence(['あ', 'b']).unwrap();
    let engine = builder.build().unwrap();
    let searcher = engine.range_searcher(costs);

    let matches = searcher
        .search(&['x'], &RangeSearchParams::new(0.25))
        .unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].sequence_id, 0);
    assert_eq!(matches[0].token_range, 0..1);
    assert_abs_diff_eq!(matches[0].distance, 0.2, epsilon = 1e-6);
}

#[cfg(feature = "persist")]
#[test]
fn mapped_embeddings_and_costs_preserve_search_results() {
    use tempfile::tempdir;
    use yurine::costs::EmbeddingStore;
    use yurine::persistence::CharCodec;

    let mut embeddings = EmbeddingStoreBuilder::new(NonZeroUsize::new(2).unwrap());
    embeddings.insert('x', [1.0, 0.0]).unwrap();
    embeddings.insert('あ', [0.8, 0.6]).unwrap();
    let embeddings = embeddings.build();
    let owned_costs = CosineEmbeddingCosts::new(embeddings.clone());

    let directory = tempdir().unwrap();
    let embedding_path = directory.path().join("embeddings.yurine");
    let costs_path = directory.path().join("costs.yurine");
    embeddings.save_with(&embedding_path, &CharCodec).unwrap();
    owned_costs.save(&costs_path).unwrap();
    let mapped = EmbeddingStore::open_with(embedding_path, &CharCodec).unwrap();
    let mapped_costs = CosineEmbeddingCosts::open(costs_path, mapped).unwrap();

    let mut builder = SearchEngineBuilder::new();
    builder.add_sequence(['あ']).unwrap();
    let engine = builder.build().unwrap();
    let params = RangeSearchParams::new(0.25);
    let owned = engine
        .range_searcher(owned_costs)
        .search(&['x'], &params)
        .unwrap();
    let mapped = engine
        .range_searcher(mapped_costs)
        .search(&['x'], &params)
        .unwrap();

    assert_eq!(mapped, owned);
}
