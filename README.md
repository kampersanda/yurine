# yurine

Fast, exact search for sequence segments under weighted edit distance.

## What Yurine does

Yurine finds approximate query segments inside longer token sequences. In this
example, brackets mark the returned range:

```text
Indexed: Jinbocho is a [book town     known for curry]
Query:                  book district known for curry
Match:                  book town     known for curry
Edit:                        district -> town (cost 0.25)
```

## Key characteristics

- **Segment search:** finds matching ranges within longer sequences.
- **Weighted edits:** costs can vary by token, operation, and direction.
- **Embedding costs:** cosine distance can make semantically related tokens
  with different spellings, such as `literature` and `books`, inexpensive
  substitutions.
- **Exact results:** returns every segment within the threshold; “exact” means
  complete results, not exact string matching.
- **Reusable indexes:** supports generic token types, multiple cost policies,
  and memory-mapped persistence.

## Usage

```rust
use yurine::costs::CustomCosts;
use yurine::{Cost, RangeSearchParams, SearchEngineBuilder};

fn main() -> yurine::errors::Result<()> {
    let mut builder = SearchEngineBuilder::new();
    let jinbocho = builder.add_sequence([
        "Jinbocho", "is", "a", "book", "town", "known", "for", "curry",
    ])?;

    let engine = builder.build()?;
    let mut costs = CustomCosts::default();
    costs.set_substitution("district", "town", Cost::new_const(0.25));
    let matches = engine
        .range_searcher(costs)
        .search(
            &["book", "district", "known", "for", "curry"],
            &RangeSearchParams::new(0.25),
        )?;

    assert_eq!(matches[0].sequence_id, jinbocho);
    assert_eq!(matches[0].distance, 0.25);
    assert_eq!(matches[0].token_range, 3..8);
    Ok(())
}
```

The query matches the `book town known for curry` segment rather than the whole
sequence. It does not occur verbatim: replacing `district` with `town` costs
`0.25`, less than the default substitution cost.

## Embedding-based search

`CosineEmbeddingCosts` derives substitution costs from token embeddings, so
similar tokens can match without an explicit rule:

```rust
use std::num::NonZeroUsize;
use yurine::costs::{CosineEmbeddingCosts, EmbeddingStoreBuilder};
use yurine::{RangeSearchParams, SearchEngineBuilder};

fn main() -> yurine::errors::Result<()> {
    let mut builder = SearchEngineBuilder::new();
    builder.add_sequence([
        "Visitors", "enjoy", "books", "and", "curry", "in", "Jinbocho",
    ])?;
    let engine = builder.build()?;

    let mut embeddings = EmbeddingStoreBuilder::new(NonZeroUsize::new(2).unwrap());
    embeddings.insert("literature", [1.0, 0.0])?;
    embeddings.insert("books", [0.8, 0.6])?;
    let costs = CosineEmbeddingCosts::new(embeddings.build());

    let matches = engine.range_searcher(costs).search(
        &["literature", "and", "curry"],
        &RangeSearchParams::new(0.2),
    )?;

    assert_eq!(matches[0].token_range, 2..5);
    Ok(())
}
```

Here, `literature` matches `books` by cosine distance, returning the
`books and curry` segment. The crate-level Rust Doc contains the tested
version of this example.

## Optional persistence

The `persist` feature adds immutable, memory-mapped snapshots for search
engines, embedding stores, and edit-cost policies:

```rust
use tempfile::tempdir;
use yurine::persistence::StringCodec;
use yurine::{SearchEngine, SearchEngineBuilder};

fn main() -> yurine::errors::Result<()> {
    let directory = tempdir().expect("create temporary directory");
    let path = directory.path().join("index.yurine");
    let mut builder = SearchEngineBuilder::new();
    builder.add_sequence(["Jinbocho", "book", "town"].map(str::to_owned))?;
    builder.build()?.save_with(&path, &StringCodec)?;

    let engine = SearchEngine::open_with(&path, &StringCodec)?;
    engine.verify()?;
    Ok(())
}
```

Build with `--features persist`. Opened indexes keep their large corpus and
posting arrays memory-mapped. The snapshot must not be modified or truncated
while it is mapped.

## Documentation

The Rust Doc is the primary library documentation. It contains the API
contracts, usage guidance, and executable examples, including embedding-based
search. Generate and open it with:

```console
$ cargo doc --no-deps --open
```

Enable all documented persistence APIs with:

```console
$ cargo doc --no-deps --all-features --open
```

Rust Doc examples are tested with `cargo test --doc`. Run
`cargo test --doc --all-features` to test the README examples as well, including
persistence. This README intentionally keeps its examples introductory and does
not duplicate detailed API guidance. New public behavior and examples should be
documented in `src/lib.rs` or on the relevant public item so they remain close
to the code and can be tested.

## Command-line search

The workspace includes the `yurine` command for searching newline-delimited
text. Run it from the repository with:

```console
$ printf 'Jinbocho is a book town known for curry\n' | \
    cargo run -p yurine-cli -- --tokenizer whitespace --threshold 1 \
    'book district known for curry' -
0	1	14	39	book town known for curry
```

See the [CLI guide](cli/README.md) for installation, input and output formats,
tokenization, and custom or embedding-based edit costs.

## Data preparation

The [`tools/`](tools/) project prepares tokenized corpora and static embeddings
for the command-line interface. See [`tools/README.md`](tools/README.md) for its
setup and commands.

## References

Yurine implements the search algorithm proposed for subtrajectory search in
road networks, applied here to sequences of arbitrary tokens.

Satoshi Koide, Chuan Xiao, and Yoshiharu Ishikawa. Fast Subtrajectory
Similarity Search in Road Networks under Weighted Edit Distance Constraints.
*PVLDB*, 13(11): 2188–2201, 2020.
<https://doi.org/10.14778/3407790.3407818>

## License

Licensed under either of MIT or Apache 2.0 at your option.
