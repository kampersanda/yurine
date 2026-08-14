# Yurine data preparation tools

This uv project provides separate commands for fetching evaluation corpora and
for preparing static word embeddings and line-oriented corpora for the Yurine
command-line interface.

## Setup

```console
$ cd tools
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

Embedding records and generated cost configurations are validated and
serialized through Pydantic models before they are written.

## Fetch Japanese Wikipedia passages

`fetch-jawiki` downloads a passage release from
[Wikipedia-Utils](https://github.com/singletongue/wikipedia-utils) and writes
one passage per line, which is what `preprocess-corpus` expects. The two
commands compose directly:

```console
$ uv run yurine-fetch-jawiki output/jawiki.txt --limit 1000000
$ uv run yurine-preprocess-corpus output/jawiki.txt output/corpus.txt
```

The default is `passages-c400` from the `20240401` dump: 5.81M passages of
roughly 400 characters each. Passages keep several sentences per line, so a
query can match a segment shorter than the indexed sequence. Use
`--dataset passages-c300` or `--dataset passages-para` for other chunk sizes,
and `--dump 20230403` for the earlier dump.

The archive is streamed and decompressed as it arrives, so `--limit` stops the
download early. It is the option to use when building a size ladder for
measurements; the complete `passages-c400` archive is about 1.4 GB compressed.

Whitespace inside a passage is collapsed so that one passage always occupies
one line. Yurine reports matches by line position, so `--metadata` writes a
JSON Lines file naming the article and section behind each line:

```console
$ uv run yurine-fetch-jawiki output/jawiki.txt --metadata output/jawiki.jsonl
```

```json
{"line":1,"id":1,"pageid":5,"revid":99347164,"title":"アンパサンド","section":"__LEAD__"}
```

`--cache PATH` keeps the downloaded archive and reuses it on later runs, which
avoids repeated downloads when generating several corpus sizes. A cached run
downloads the whole archive even when `--limit` is small. The download goes to
a temporary file first, so an interrupted run does not leave a partial archive
behind as a cache hit.

The Wikipedia-Utils data is derived from Japanese Wikipedia and is distributed
under CC-BY-SA-3.0 and GFDL.

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

Tokenization runs in a single process by default. For a large corpus,
`--workers` spreads it over several processes, with `0` requesting one per CPU.
The input is streamed either way, and the output is byte-for-byte the same:

```console
$ uv run yurine-preprocess-corpus --workers 0 corpus.txt output/corpus.txt
```

## Search with Yurine

```console
$ cargo run -p yurine-cli -- index \
    --tokenizer whitespace \
    tools/output/corpus.index \
    tools/output/corpus.txt
$ cargo run -p yurine-cli -- costs \
    --tokenizer whitespace \
    tools/output/costs.json \
    tools/output/costs.snapshot
$ cargo run -p yurine-cli -- search \
    --costs tools/output/costs.snapshot \
    --threshold 0.3 \
    tools/output/corpus.index \
    '東京 都'
```

Converted embeddings cover a large vocabulary, and reading them takes far
longer than a search. The `costs` command compiles them once so that every
later search opens the snapshot instead. Passing `costs.json` to `--costs`
still works and reads the embeddings again for each query.

Run each feature command with `--help` for all options.

## Development

Run formatting checks, linting, type checking, and tests from `tools/`:

```console
$ uv run ruff format --check .
$ uv run ruff check .
$ uv run ty check
$ uv run pytest
```

The package separates command-line entry points from reusable processing code:

```text
src/yurine_tools/
├── cli/           # Tap argument definitions and command runners
├── corpus.py      # Corpus normalization and tokenization
├── embeddings.py  # Streaming embedding conversion
├── jawiki.py      # Wikipedia-Utils release download and extraction
├── options.py     # Shared option types
├── schemas.py     # Pydantic output schemas
└── text_io.py     # Standard and compressed text I/O
```
