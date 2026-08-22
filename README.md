# yurine

Fast, exact search for sequence segments under weighted edit distance.

## What Yurine does

Yurine finds approximate query segments inside longer token sequences. In this
example, brackets mark the returned segment:

```text
Indexed: Jinbocho is a [book town known for curry] .
                              │     │
                              │     └── ins(known) = 0.2
                              │
                              ├── sub(district, town) = 0.1
                              │
Query:                  book district for curry
```

## Key characteristics

- **Segment search:** finds matching ranges within longer sequences.
- **Weighted edits:** costs can vary by token, operation, and direction; they
  need not be symmetric or satisfy the triangle inequality.
- **Semantic matching:** substitution costs can come from token embeddings, so
  `literature` matches `books` — paraphrases, not just exact spellings.
- **Exact results:** no match within the threshold is missed; “exact” means
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

Feeding embedding-derived costs into the same weighted edit distance search
turns segment search semantic: queries match paraphrases, not just exact
spellings.

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

[SoftMatcha](https://softmatcha.github.io/) is a similar project that
efficiently solves a special case of this weighted edit distance
at trillion-scale corpora; Yurine targets the general problem,
with arbitrary per-token, per-operation costs.

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
text. It indexes a corpus once and searches the saved index any number of
times. Install it with Cargo and run it:

```console
$ cargo install --git https://github.com/kampersanda/yurine yurine-cli
$ printf 'Jinbocho is a book town known for curry\n' > corpus.txt
$ yurine index --tokenizer whitespace corpus.index corpus.txt
$ yurine search --threshold 1 corpus.index 'book district known for curry'
0	1	14	39	book town known for curry
```

See the [CLI guide](cli/README.md) for installation, input and output formats,
tokenization, and custom or embedding-based edit costs.

## Data preparation

The [`tools/`](tools/) project prepares tokenized corpora and static embeddings
for the command-line interface. See [`tools/README.md`](tools/README.md) for its
setup and commands.

## Limitations

Yurine is an early project. These are the limitations of the current release:

- **Indexes are immutable.** Adding, changing, or removing a sequence means
  building a new index.
- **Building an index is expensive.** The whole corpus is held in memory, so
  peak memory reaches roughly ten times the size of the text, and construction
  has not been tuned for speed.
- **Indexes are large.** Nothing is compressed yet: an indexed token costs
  about 12 bytes, making an index a few times the size of the text it indexes.
- **Searches are single-threaded.** One index can answer several queries at
  once, but a single query uses one core.
- **Search time grows with the vocabulary.** A larger vocabulary costs more even
  when the corpus does not change. The growth is mild with table-driven costs
  and steep with embedding-based ones.
- **Loose thresholds are rejected.** A threshold reaching the cost of deleting
  the whole query leaves nothing for the index to filter on, and a search with
  one returns an error instead of results. With unit costs that is a threshold
  at or above the query's token count.
- **One result per match.** Overlapping segments describe one match, so a
  search returns the closest of them rather than each one. Two occurrences that
  overlap, such as `aa` twice in `aaa`, are reported once.
- **No ranking.** A search returns every match within the threshold at once.
  There is no top-k, relevance scoring, or streaming of results.
- **Only static embeddings.** Substitution costs come from a fixed table of
  token vectors; context-dependent embeddings are out of scope.
- **Tokenization is up to the caller.** The library takes token sequences and
  returns token ranges. The command-line interface tokenizes by character or
  whitespace only, without normalization.
- **Saved indexes are not portable.** They load only on little-endian targets
  and only with the version of Yurine that wrote them. The API and the file
  formats may still change before 1.0.

## References

Yurine implements the search algorithm proposed for subtrajectory search in
road networks, applied here to sequences of arbitrary tokens.

> Satoshi Koide, Chuan Xiao, and Yoshiharu Ishikawa. Fast Subtrajectory
> Similarity Search in Road Networks under Weighted Edit Distance Constraints.
> _PVLDB_, 13(11): 2188–2201, 2020.
> <https://doi.org/10.14778/3407790.3407818>

## License

Licensed under either of MIT or Apache 2.0 at your option.
