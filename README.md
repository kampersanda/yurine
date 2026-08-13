# yurine

Fast and exact search for sequence segments under weighted edit distance

```rust
use yurine::costs::Cost;
use yurine::costs::levenshtein::LevenshteinCosts;
use yurine::errors::Result;
use yurine::search::SearchEngineBuilder;
use yurine::search::range_search::RangeSearchParams;

fn main() -> Result<()> {
    let mut builder = SearchEngineBuilder::new();

    let tokyo = builder.add_sequence(['東', '京'])?;
    builder.add_sequence(['京', '都'])?;

    let engine = builder.build()?;
    let searcher = engine.range_searcher(LevenshteinCosts::new());
    let matches = searcher.search(
        &['東', '京'],
        &RangeSearchParams::new(Cost::ZERO),
    )?;

    assert_eq!(matches[0].sequence_id, tokyo);
    assert_eq!(matches[0].token_range.start.get(), 0);
    assert_eq!(matches[0].token_range.end.get(), 2);
    Ok(())
}
```

Yurine accepts token sequences and encodes them as internal symbol strings.
Callers own source text, tokenization, and conversion from returned token ranges
to source-text byte ranges. Token types remain generic for in-memory search.

## Persistent search indexes

Enable the `persist` feature to save an immutable `SearchEngine` snapshot and
open it in another process without rebuilding the index:

```rust
use yurine::persistence::CharCodec;
use yurine::search::{SearchEngine, SearchEngineBuilder};

# fn example() -> yurine::errors::Result<()> {
let mut builder = SearchEngineBuilder::new();
builder.add_sequence(['東', '京'])?;
let engine = builder.build()?;
engine.save_with("index.yurine", &CharCodec)?;

let mapped = SearchEngine::open_with("index.yurine", &CharCodec)?;
mapped.verify()?;
# Ok(())
# }
```

`StringCodec` is also provided. Other token types implement `TokenCodec`; its
identifier and version must remain stable, and decoding an encoded token must
preserve equality and hashing.

The version 1 file is little-endian and contains a fixed header and section
table followed by vocabulary token offsets/blob, sequence offsets, corpus
symbols, posting offsets, and postings. Vocabulary tokens are decoded into
memory. The four fixed-width corpus and posting arrays remain views into one
read-only mmap. Opening validates metadata and every offset. Search validates
the symbol range of each corpus sequence it accesses, but it does not establish
full corpus/posting consistency. Call `verify` before searching when the snapshot
is not trusted; it scans all corpus symbols and postings. `save_with` performs
this complete, corpus-linear verification before writing.

Saving writes and synchronizes a temporary file in the destination directory,
then atomically renames it. Published snapshots must never be modified or
truncated in place while mapped. On Windows, replacing a snapshot that another
process has mapped can fail; publish a new path or wait for readers to close it.

### Persistent embeddings and built-in costs

`EmbeddingStore` is immutable. Build one with `EmbeddingStoreBuilder`, then
save or use it for cosine costs:

```rust
use std::num::NonZeroUsize;
use yurine::costs::embedding::{
    CosineEmbeddingCosts, EmbeddingStore, EmbeddingStoreBuilder,
};
use yurine::persistence::CharCodec;

# fn example() -> yurine::errors::Result<()> {
let mut builder = EmbeddingStoreBuilder::new(NonZeroUsize::new(2).unwrap());
builder.insert('東', [1.0, 0.0])?;
let embeddings = builder.build();
embeddings.save_with("embeddings.yurine", &CharCodec)?;

let mapped = EmbeddingStore::open_with("embeddings.yurine", &CharCodec)?;
let costs = CosineEmbeddingCosts::new(mapped.clone());
costs.save("cosine-costs.yurine")?;

let costs = CosineEmbeddingCosts::open("cosine-costs.yurine", mapped)?;
costs.verify()?;
# Ok(())
# }
```

This replaces the former mutable `EmbeddingStore::new`/`insert` workflow: call
`EmbeddingStoreBuilder::new`, insert rows, and finish with `build`.

The embedding file stores token offsets/blob, a small dimension record, and a
contiguous little-endian `f32` matrix. Only the token index is rebuilt in heap
memory; the matrix remains mmap-backed. Rows are checked for finite, non-zero,
normalized values on first access, and the result is cached. A damaged row
behaves as a missing embedding in normal search, while `verify` scans all rows
and reports the corruption. Call `verify` immediately after opening an
untrusted embedding snapshot to avoid silently applying the configured
missing-embedding cost.

Saving retains only encoded token metadata and row indices in memory. Vector
rows are streamed directly from the existing owned or mmap backing, so saving
does not materialize a second copy of a multi-gigabyte matrix.

`LevenshteinCosts::save`/`open`, `CustomCosts::save_with`/`open_with`, and
`CosineEmbeddingCosts::save`/`open` use independent files. Cosine-cost files
contain only their three constants: callers explicitly supply the separately
opened embedding store. Custom rules are sorted by encoded token bytes, so the
same logical policy and codec produce identical files. All persisted cost
values are decoded as `f32` and validated before becoming `Cost` values.

## Command-line search

The `yurine-cli` package provides the `yurine` binary. It reads one source text
per line from a file or standard input. Searches use unit Levenshtein
costs, character tokenization, and a default threshold of zero. The CLI owns
the source text and token byte ranges so it can continue to print matched text;
these are not stored by the Yurine library.

Results are headerless, tab-delimited CSV records containing the sequence ID,
distance, byte start, byte end, and matched text. CSV quoting applies when
needed.

```console
$ printf '東京都\n京都市\n' | cargo run -p yurine-cli -- '東京'
0	0	0	6	東京
```

Use `--tokenizer whitespace` to search whitespace-delimited tokens:

```console
$ printf 'new york city\nyork new\n' | cargo run -p yurine-cli -- --tokenizer whitespace 'new york' -
0	0	0	8	new york
```

Use `--threshold` to allow insertions, deletions, or substitutions.

A corpus file can be passed after the query. When it is omitted or is `-`, the
corpus is read from standard input.

### Configuring edit costs

Use `--costs` to load an edit-cost policy from JSON. Without this option,
searches continue to use unit Levenshtein costs.

The following configuration loads static token embeddings from a JSON Lines
file. Relative resource paths are resolved from the directory containing the
configuration file.

```json
{
  "version": 1,
  "type": "embedding",
  "embeddings": {
    "path": "embeddings.jsonl",
    "format": "jsonl"
  },
  "missing_substitution_cost": 1.0,
  "deletion_cost": 1.0,
  "insertion_cost": 1.0
}
```

Each embedding record contains one token and one vector:

```json
{"token":"x","embedding":[1.0,0.0]}
{"token":"a","embedding":[0.8,0.6]}
```

The three cost fields are optional and default to `1.0`. Embeddings must be
non-empty, finite, non-zero vectors with the same dimension. Duplicate tokens
are rejected.

For token-specific edit costs, use a custom configuration:

```json
{
  "version": 1,
  "type": "custom",
  "defaults": {
    "substitution": 1.0,
    "deletion": 1.0,
    "insertion": 1.0
  },
  "rules": {
    "path": "rules.jsonl",
    "format": "jsonl"
  }
}
```

Rules are directed from the query sequence to a data segment and are read one
JSON object per line:

```json
{"operation":"substitution","from":"colour","to":"color","cost":0.25}
{"operation":"deletion","token":"the","cost":0.1}
{"operation":"insertion","token":"a","cost":0.2}
```

The `defaults` object and each of its fields are optional and default to `1.0`.
An empty rules file is allowed. Duplicate rules are rejected, and substitution
between equal tokens always costs zero.

Tokens in embedding and custom-cost files must form exactly one complete token
under the selected `--tokenizer`. For example, `colour` is valid with the
whitespace tokenizer but not with the character tokenizer.

```console
$ cargo run -p yurine-cli -- \
    --tokenizer whitespace \
    --costs costs.json \
    --threshold 0.25 \
    colour corpus.txt
```

Run `cargo run -p yurine-cli -- --help` for the complete command-line reference.

## Preparing embedding-search inputs

The uv project in [`tools/`](tools/) converts word2vec text embeddings to
Yurine JSON Lines and normalizes and tokenizes corpora and queries. Its default
Sudachi configuration is compatible with chiVe v1.3, while its input
normalization, token form, split mode, and whitespace tokenizer can be selected
for other embedding models. See [`tools/README.md`](tools/README.md) for
setup and usage.
