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
