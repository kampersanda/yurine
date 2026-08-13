# Yurine CLI

The `yurine` command searches newline-delimited text for matching segments under
weighted edit distance. Indexing and searching are separate commands, so a
corpus is indexed once and searched any number of times without rebuilding.

## Quick start

Run the CLI from the repository with Cargo:

```console
$ printf 'Jinbocho is a book town known for curry\n' > corpus.txt
$ cargo run -p yurine-cli -- index --tokenizer whitespace corpus.index corpus.txt
$ cargo run -p yurine-cli -- search --threshold 1 corpus.index 'book district known for curry'
0	1	14	39	book town known for curry
```

The query differs from the result by one token and matches a segment within the
longer input line. Without `--costs`, substitutions, deletions, and insertions
have unit cost.

## Command syntax

```console
yurine index [OPTIONS] <INDEX> [CORPUS]
yurine search [OPTIONS] <INDEX> <QUERY>
```

`INDEX` is the index directory: `index` creates it, and `search` reads it.
`CORPUS` is a file containing one source text per line; when it is omitted or
is `-`, `index` reads standard input. `QUERY` is the text to search for.

To build a standalone binary:

```console
$ cargo build --release -p yurine-cli
$ target/release/yurine --help
```

### `index` options

- `--tokenizer character|whitespace` selects tokenization. It defaults to
  `character`.
- `--timing` reports the elapsed time of each stage on standard error.

### `search` options

- `-t, --threshold <NUMBER>` sets the inclusive maximum edit distance. It
  defaults to `0`.
- `--costs <FILE>` loads a custom or embedding-based cost policy.
- `--eta <NUMBER>` overrides an internal candidate-generation radius. Most
  users should leave it unset; it affects filtering performance, not the
  distance threshold used to verify results.
- `--verify` checks the integrity of the index before searching it. Use it on
  an index from an untrusted source; the check reads the whole index.
- `--timing` reports the elapsed time of each stage on standard error.

There is no `--tokenizer` option on `search`. The query is tokenized with the
strategy recorded in the index, so it always matches the corpus.

Use `--help` for the generated command reference.

## Input and tokenization

Each corpus line receives a zero-based sequence ID in input order.

The `character` tokenizer treats each Unicode scalar value as one token. The
`whitespace` tokenizer treats each non-empty run between Unicode whitespace as
one token. Tokenization does not normalize or lowercase text.

Use `whitespace` for pre-tokenized text or word-level cost policies:

```console
$ cargo run -p yurine-cli -- index --tokenizer whitespace corpus.index corpus.txt
```

Cost rules and embeddings must use exactly one complete token under the
tokenizer of the index they are searched with.

## Index directory

`yurine index` writes four files, and `yurine search` needs all of them:

| File | Contents |
| --- | --- |
| `metadata.json` | Format version, tokenizer, and number of source texts |
| `engine.yurine` | The search index, memory-mapped when searching |
| `sources.txt` | A copy of the corpus lines |
| `sources.idx` | Byte offset of each line in `sources.txt` |

Results report byte offsets into the original text, which tokenization does not
preserve, so the source texts are stored next to the index. The offset table
lets a search read only the lines it matched, so search time does not grow with
the size of the corpus.

An index is an immutable snapshot. Adding or changing source texts requires
building a new index. Rebuilding into an existing directory overwrites all four
files.

## Output

Results are headerless, tab-delimited CSV records with these fields:

| Field | Meaning |
| --- | --- |
| 1 | Zero-based sequence ID, equal to the corpus line number |
| 2 | Weighted edit distance |
| 3 | Inclusive UTF-8 byte offset in the source line |
| 4 | Exclusive UTF-8 byte offset in the source line |
| 5 | Matched source text |

CSV quoting is applied when a field contains a tab, quote, or newline. Results
are ordered by sequence ID, start offset, then end offset.

## Timing

`--timing` reports how long each stage took. The report goes to standard error,
so it never mixes with the results on standard output:

```console
$ cargo run -p yurine-cli --release -- index --timing \
    --tokenizer whitespace corpus.index corpus.txt
timing: read=0.403ms build=0.103ms save=8.422ms total=9.072ms

$ cargo run -p yurine-cli --release -- search --timing \
    --threshold 1 corpus.index 'book district known for curry'
0	1	14	39	book town known for curry
timing: open=0.162ms costs=0.000ms search=0.050ms total=0.254ms
```

| Command | Stage | Covers |
| --- | --- | --- |
| `index` | `read` | Reading the corpus, tokenizing it, and storing the source texts |
| `index` | `build` | Building the index |
| `index` | `save` | Writing the index and its metadata |
| `index` | `total` | The whole run |
| `search` | `open` | Reading the metadata and opening the index, including `--verify` |
| `search` | `costs` | Loading the `--costs` configuration and its data files |
| `search` | `search` | Tokenizing the query and searching |
| `search` | `total` | The whole run, including writing the results |

Every stage is always reported, so the fields are the same with and without
`--costs`. When `--costs` is omitted, the `costs` stage only builds the unit-cost
policy, which rounds to `0.000ms`. Values are always in milliseconds with three
decimal places, and the difference between `total` and the sum of the stages is
output, start-up, and result-reading overhead.

Each stage is measured once, without warm-up or repetition, so the numbers vary
between runs. Build a release binary before comparing them, and use the
[`benchmarks/`](../benchmarks/) crate for rigorous measurement. When the corpus
comes from standard input, `read` includes time spent waiting for the upstream
process.

## Custom edit costs

Custom rules are directed from the query to a corpus segment. For example,
this rule makes query token `district` match corpus token `town` at cost
`0.25`.

Create `costs.json`:

```json
{
  "version": 1,
  "type": "custom",
  "rules": {
    "path": "rules.jsonl",
    "format": "jsonl"
  }
}
```

Create `rules.jsonl`:

```json
{"operation":"substitution","from":"district","to":"town","cost":0.25}
```

Create `corpus.txt`:

```text
Jinbocho is a book town known for curry
```

Then run:

```console
$ cargo run -p yurine-cli -- index --tokenizer whitespace corpus.index corpus.txt
$ cargo run -p yurine-cli -- search \
    --costs costs.json \
    --threshold 0.25 \
    corpus.index \
    'book district known for curry'
0	0.25	14	39	book town known for curry
```

The optional `defaults` object can set `substitution`, `deletion`, and
`insertion`; omitted values default to `1.0`. Rule records have one of these
forms:

```json
{"operation":"substitution","from":"query-token","to":"corpus-token","cost":0.25}
{"operation":"deletion","token":"query-token","cost":0.1}
{"operation":"insertion","token":"corpus-token","cost":0.2}
```

## Embedding-based costs

Embedding-based costs use cosine distance for substitutions. Create
`costs.json`:

```json
{
  "version": 1,
  "type": "embedding",
  "embeddings": {
    "path": "embeddings.jsonl",
    "format": "jsonl"
  }
}
```

Create `embeddings.jsonl` with one token and vector per line:

```json
{"token":"literature","embedding":[1.0,0.0]}
{"token":"books","embedding":[0.8,0.6]}
```

Create `corpus.txt`:

```text
Visitors enjoy books and curry in Jinbocho
```

Then run:

```console
$ cargo run -p yurine-cli -- index --tokenizer whitespace corpus.index corpus.txt
$ cargo run -p yurine-cli -- search \
    --costs costs.json \
    --threshold 0.2 \
    corpus.index \
    'literature and curry'
0	0.19999999	15	30	books and curry
```

All vectors must have the same non-zero dimension and contain finite values.
Vectors are normalized when loaded. Duplicate tokens, zero vectors, and tokens
that do not match the tokenizer of the index are rejected.

The optional `missing_substitution_cost`, `deletion_cost`, and `insertion_cost`
fields default to `1.0`.

Paths in a cost configuration are resolved relative to that configuration
file, not the current working directory.

Cost policies are loaded on every search. Only the index is prebuilt, so a
large embedding file is read again for each query.

## Preparing corpora and embeddings

The [`tools/`](../tools/) project can normalize and tokenize corpora and convert
word2vec text embeddings to the JSON Lines format used above. See the
[data-preparation guide](../tools/README.md).
