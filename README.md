# yurine

Fast, exact search for sequence segments under weighted edit distance.

Yurine indexes sequences of generic tokens and returns every non-empty segment
within a chosen edit-distance threshold. Applications retain ownership of
tokenization, source text, and conversion from token ranges to source ranges.

## Usage

```rust
use yurine::costs::{Cost, levenshtein::LevenshteinCosts};
use yurine::search::{SearchEngineBuilder, range_search::RangeSearchParams};

let mut builder = SearchEngineBuilder::new();
builder.add_sequence(['東', '京', '都'])?;

let engine = builder.build()?;
let matches = engine
    .range_searcher(LevenshteinCosts::new())
    .search(
        &['東', '京'],
        &RangeSearchParams::new(Cost::ZERO),
    )?;

assert_eq!(matches[0].token_range.start.get(), 0);
assert_eq!(matches[0].token_range.end.get(), 2);
```

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
$ printf '東京都\n京都市\n' | cargo run -p yurine-cli -- '東京'
0	0	0	6	東京
```

Run `cargo run -p yurine-cli -- --help` for tokenization, thresholds, corpora,
and edit-cost configuration options.

## Data preparation

The [`tools/`](tools/) project prepares tokenized corpora and static embeddings
for the command-line interface. See [`tools/README.md`](tools/README.md) for its
setup and commands.
