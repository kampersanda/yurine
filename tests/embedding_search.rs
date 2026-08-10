use std::num::NonZeroUsize;

use approx::assert_abs_diff_eq;
use yurine::costs::Cost;
use yurine::costs::embedding::{CosineEmbeddingCosts, EmbeddingStore};
use yurine::search::SearchEngineBuilder;
use yurine::search::range_search::RangeSearchParams;
use yurine::tokenization::character::CharacterTokenizer;
use yurine::types::{Position, StringId};

#[test]
fn query_only_token_uses_embedding_for_candidate_generation_and_verification() {
    let mut embeddings = EmbeddingStore::new(NonZeroUsize::new(2).unwrap());
    embeddings.insert('x', vec![1.0, 0.0]).unwrap();
    embeddings.insert('あ', vec![0.8, 0.6]).unwrap();
    embeddings.insert('b', vec![0.0, 1.0]).unwrap();

    let mut builder = SearchEngineBuilder::new(
        CharacterTokenizer::new(),
        CosineEmbeddingCosts::new(embeddings),
    );
    builder.add_string("あb").unwrap();
    let engine = builder.build().unwrap();

    let matches = engine
        .range_search(
            "x",
            &RangeSearchParams::new(Cost::new_const(0.25)).with_eta(Cost::new_const(0.25)),
        )
        .unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].string_id, StringId::new(0));
    assert_eq!(matches[0].token_range, Position::new(0)..Position::new(1));
    assert_eq!(matches[0].byte_range, 0..3);
    assert_abs_diff_eq!(matches[0].distance.get(), 0.2, epsilon = 1e-6);
}
