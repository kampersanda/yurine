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

The `yurine` binary reads one corpus string per line and prints every matching
substring as CSV-compatible, tab-separated fields: string ID, distance, byte
start, byte end, and matched text.

```console
$ printf '東京都\n京都市\n' | cargo run -p yurine-cli -- '東京'
0	0	0	6	東京
```

The default edit-distance threshold is zero. Use `--threshold` to allow edits,
or `--tokenizer whitespace` to search whitespace-delimited tokens. A corpus
file can be passed after the query; when it is omitted or is `-`, the corpus is
read from standard input.

```console
$ cargo run -p yurine-cli -- --tokenizer whitespace --threshold 1 'new york' corpus.txt
```

Run `cargo run -p yurine-cli -- --help` for the complete command-line reference.
