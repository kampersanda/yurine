# yurine

Fast and exact substring search under weighted edit distance

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

Run `cargo run -p yurine-cli -- --help` for the complete command-line reference.
