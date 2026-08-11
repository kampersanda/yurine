# Yurine data preparation tools

This uv project provides separate commands for preparing static word
embeddings and line-oriented corpora for the Yurine command-line interface.

## Setup

```console
$ cd scripts
$ uv sync
```

The project pins SudachiPy 0.6.8 and SudachiDict Core 20240109, matching the
tokenization environment documented for chiVe v1.3.

## Convert embeddings

`convert-embeddings` converts the word2vec text format to the JSON Lines format
accepted by Yurine. Both the header form and the headerless form are accepted.
Input and output compression is detected from `.gz`, `.bz2`, `.xz`, or `.lzma`.
Use `-` for standard input or output. Yurine itself reads uncompressed JSON
Lines, so compressed output is intended for storage or transfer rather than a
generated cost configuration.

```console
$ uv run yurine-convert-embeddings \
    chive-1.3-mc90.txt \
    output/embeddings.jsonl \
    --cost-config output/costs.json
```

The conversion is streamed, so the entire source model is not loaded into
memory. By default tokens are preserved exactly. The `--normalization` option
can apply NFC, NFKC, or NFKC plus case folding for models that require it.
Header detection defaults to automatic; use `--header absent` for a headerless,
one-dimensional model whose first token and vector value are both integers.

The generated records look like this:

```json
{"token":"東京","embedding":[0.1,0.2]}
```

## Preprocess a corpus or query

The default settings use Sudachi mode B and `normalized_form`, which are useful
with chiVe. One input line always produces one output line.

```console
$ uv run yurine-preprocess-corpus corpus.txt output/corpus.txt
$ printf '東京都へ行く\n' | uv run yurine-preprocess-corpus - -
東京 都 へ 行く
```

Sudachi modes A, B, and C and surface, normalized, and dictionary forms are
selectable. `--sudachi-config` can point to another Sudachi configuration and
dictionary:

```console
$ uv run yurine-preprocess-corpus --mode A --form surface input.txt output.txt
```

For already-tokenized or non-Japanese data, use the generic whitespace mode.
It preserves tokens while canonicalizing runs of whitespace:

```console
$ uv run yurine-preprocess-corpus \
    --tokenizer whitespace --normalization nfkc input.txt output.txt
```

Use the same preprocessing options for the corpus and query. Different
Sudachi modes or output forms produce different tokens even when the source
text is identical.

## Search with Yurine

```console
$ cargo run -p yurine-cli -- \
    --tokenizer whitespace \
    --costs scripts/output/costs.json \
    --threshold 0.3 \
    '東京 都' \
    scripts/output/corpus.txt
```

Run each feature command with `--help` for all options.
