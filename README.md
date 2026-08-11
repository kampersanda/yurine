# yurine

Fast and exact substring search under weighted edit distance

The reproducible performance baseline and compatibility workload for the
persistent-index work are documented in [`benchmarks/`](benchmarks/README.md).

```rust
use yurine::costs::Cost;
use yurine::costs::levenshtein::LevenshteinCosts;
use yurine::errors::Result;
use yurine::search::SearchEngineBuilder;
use yurine::search::range_search::RangeSearchParams;
use yurine::tokenization::character::CharacterTokenizer;

fn main() -> Result<()> {
    let mut builder = SearchEngineBuilder::new(
        CharacterTokenizer::new(),
        LevenshteinCosts::new(),
    );

    let tokyo = builder.add_string("東京")?;
    builder.add_string("京都")?;

    let engine = builder.build()?;
    let matches = engine.range_search("東京", &RangeSearchParams::new(Cost::ZERO))?;

    assert_eq!(matches[0].string_id, tokyo);
    assert_eq!(matches[0].byte_range, 0..6);
    Ok(())
}
```

## Command-line search

The `yurine-cli` package provides the `yurine` binary. It reads one corpus
string per line from a file or standard input. Searches use unit Levenshtein
costs, character tokenization, and a default threshold of zero.

Results are headerless, tab-delimited CSV records containing the string ID,
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
file. Relative data paths are resolved from the directory containing the
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

Rules are directed from the query to the corpus and are read one JSON object
per line:

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

## Preparing embedding-search data

The uv project in [`tools/`](tools/) converts word2vec text embeddings to
Yurine JSON Lines and normalizes and tokenizes corpora and queries. Its default
Sudachi configuration is compatible with chiVe v1.3, while its input
normalization, token form, split mode, and whitespace tokenizer can be selected
for other embedding models. See [`tools/README.md`](tools/README.md) for
setup and usage.
