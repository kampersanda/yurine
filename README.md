# yurine

Fast, exact search for sequence segments under weighted edit distance.

Yurine indexes sequences of generic tokens and returns every non-empty segment
within a chosen edit-distance threshold. Applications retain ownership of
tokenization, source text, and conversion from token ranges to source ranges.
Here, "exact search" means that qualifying approximate matches are not omitted;
it does not mean exact string matching.

## Usage

```rust
use yurine::costs::{Cost, custom::CustomCosts};
use yurine::search::{SearchEngineBuilder, range_search::RangeSearchParams};

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
        &RangeSearchParams::new(Cost::new_const(0.25)),
    )?;

assert_eq!(matches[0].sequence_id, jinbocho);
assert_eq!(matches[0].distance, Cost::new_const(0.25));
assert_eq!(matches[0].token_range.start.get(), 3);
assert_eq!(matches[0].token_range.end.get(), 8);
```

The query matches the `book town known for curry` segment rather than the whole
sequence. It does not occur verbatim: replacing `district` with `town` costs
`0.25`, less than the default substitution cost.

## Documentation

The Rust Doc is the primary library documentation. It contains the API
contracts, usage guidance, and executable examples. Generate and open it with:

```console
$ cargo doc --no-deps --open
```

Enable all documented persistence APIs with:

```console
$ cargo doc --no-deps --all-features --open
```

Rust Doc examples are tested with `cargo test --doc`. This README intentionally
keeps its example introductory and does not duplicate detailed API guidance.
New public behavior and examples should be documented in `src/lib.rs` or on the
relevant public item so they remain close to the code and can be tested.

## Optional persistence

The `persist` feature adds immutable, memory-mapped snapshots for search
engines, embedding stores, and edit-cost policies. File-lifetime requirements
and examples are documented in the `persistence` Rust Doc module.

## Command-line search

The workspace includes the `yurine` command for searching newline-delimited
text. Run it from the repository with:

```console
$ printf 'Jinbocho is a book town known for curry\n' | \
    cargo run -p yurine-cli -- --tokenizer whitespace --threshold 1 \
    'book district known for curry' -
0	1	14	39	book town known for curry
```

Run `cargo run -p yurine-cli -- --help` for tokenization, thresholds, corpora,
and edit-cost configuration options.

## Data preparation

The [`tools/`](tools/) project prepares tokenized corpora and static embeddings
for the command-line interface. See [`tools/README.md`](tools/README.md) for its
setup and commands.
