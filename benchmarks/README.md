# Baseline benchmark

This package provides the reproducible workload used to compare the in-memory
implementation with the persistent index work tracked by issue #24. Measurements
are observations, not CI pass/fail thresholds. Keep generated corpora and
measurement results outside the repository; only the code and reproduction
procedure are versioned.

Both the corpus and the token embeddings are synthetic. They are built to
compare one implementation against another under identical conditions, not to
predict how the engine performs on real text: a synthetic corpus has no natural
token distribution, and synthetic embeddings have no semantics. Read every
number here as a before-and-after ratio.

Generate the default whitespace-tokenized corpus:

```console
cargo run --release -p yurine-benchmarks -- generate /tmp/yurine-baseline.txt
```

The generator uses a fixed seed, 20,000 data sequences, 32 tokens per sequence,
a 256-token vocabulary, and an eight-token hot set. Every 128th sequence starts
with the default query sequence, so the workload always has known exact matches.
Every setting can be overridden; rerunning the same command produces identical
bytes. The vocabulary may go up to 1,000,000 tokens, because embedding-based
costs scan the whole vocabulary once per query position and their cost therefore
grows with it.

Measure construction and one cold plus five warm searches:

```console
cargo run --release -p yurine-benchmarks -- measure /tmp/yurine-baseline.txt
```

To save the engine, reopen its large arrays through mmap, and run the same
search workload, pass a snapshot path:

```console
cargo run --release -p yurine-benchmarks -- measure /tmp/yurine-baseline.txt \
  --persistent-index /tmp/yurine-baseline.index
```

## Cosine embedding costs

`--costs levenshtein` is the default and keeps the original workload.
`--costs cosine` measures `CosineEmbeddingCosts` instead, over a synthetic
embedding matrix covering the whole corpus vocabulary:

```console
cargo run --release -p yurine-benchmarks -- measure /tmp/yurine-baseline.txt \
  --costs cosine --dimension 300
```

The matrix is generated from the corpus, not read from a file, so no real
embedding data is needed. Tokens are sorted, spread over `--embedding-clusters`
clusters by their sorted position, and each embedding is the same fixed blend of
its cluster center with an independent random direction, L2-normalized by the
store. Two tokens in one cluster then have cosine similarity near
`--embedding-cohesion`, and two tokens in different clusters are
near-orthogonal. The same arguments always produce the same matrix, whatever
order the corpus presents its tokens in.

The clusters are what make the substitution neighborhood non-trivial.
Independently drawn 300-dimensional vectors are all near-orthogonal, so at a
small eta every neighborhood would hold only the query token itself, and any
change whose cost depends on neighborhood size would measure as free.
With the defaults, a 20,000-token vocabulary gives clusters of about 312 tokens
that fall inside an eta of 0.25.

`--embedding-seed` fixes the matrix, as `--seed` fixes the corpus. The embedding
options are ignored under `--costs levenshtein`.

Filtering evaluates substitutions across the vocabulary, while verification
works on candidate anchors. A query of frequent tokens spends most of its time
verifying, which hides the filtering cost this policy is slow in. To make
filtering dominate, query rare tokens instead of the hot ones the default query
uses:

```console
cargo run --release -p yurine-benchmarks -- generate /tmp/yurine-cosine.txt \
  --vocabulary 20000 --hot-vocabulary 8
cargo run --release -p yurine-benchmarks -- measure /tmp/yurine-cosine.txt \
  --costs cosine --threshold 2 --warm-runs 15 \
  --query "t9000 t9001 t9002 t9003 t9004 t9005 t9006 t9007"
```

The matrix stays on the heap, whatever `--persistent-index` does with the
engine; only the engine's own arrays are memory-mapped. It is built after the
snapshot is reopened, so it does not disturb `open_heap_peak`, and it is counted
in `engine_resident_heap`. A 20,000-token vocabulary at 300 dimensions is about
24 MB.

## Vocabulary sweep

To measure how search time responds to the vocabulary alone, generate the same
corpus shape at several vocabulary sizes and keep everything else fixed:

```console
for vocabulary in 256 2000 20000; do
  cargo run --release -p yurine-benchmarks -- generate \
    /tmp/yurine-vocabulary-$vocabulary.txt \
    --sequences 20000 --tokens 20 --vocabulary $vocabulary --hot-vocabulary 8
  cargo run --release -p yurine-benchmarks -- measure \
    /tmp/yurine-vocabulary-$vocabulary.txt --threshold 0 --warm-runs 15
done
```

An eight-token hot set keeps the default query on frequent tokens, so
`generated_candidates` is 37,848 at every vocabulary: the three runs verify the
same number of candidate anchors. They do not verify them at the same cost. A
larger vocabulary leaves those candidates sharing fewer prefixes, so the
verification cache holds more nodes and reuses fewer of its columns. The sweep
therefore isolates what the vocabulary does to the cost of verifying one anchor,
not to how many anchors there are; read `generated_candidates` first to confirm
that premise still holds before reading the timings.

This is the sweep that exposed the verification trie's linear child scan. Its
nodes fan out to the vocabulary size near the root, which charged every step of
every candidate a term the dynamic program never asked for, and warm median
search time rose by an order of magnitude across those three vocabularies.
Keying the children by symbol left the response nearly flat, and the slope left
over is the reduced prefix sharing above, which is the method's own.

Keying costs heap: a map holds a node's children less compactly than a vector
did, which raises `warm_search_heap_peak_growth` by roughly two fifths on this
workload. The trie is call-local, so that is transient search memory rather than
resident engine state, and `engine_resident_heap` does not move.

Raising `--threshold` to one keeps the same property with a larger candidate set
and admits `--costs cosine`, which is worth sweeping separately because its
vocabulary response is much steeper.

## Output

Output is tab-separated `metric`, `value`, and `unit`. It includes:

- the cost policy under measurement,
- source corpus and persistent-index sizes,
- corpus-load, engine-build, cold-search, and warm-search timings,
- snapshot-save and mmap-open timings,
- allocator-observed heap peaks for each phase,
- process peak RSS after build, cold search, and warm search,
- current file-backed RSS after open, cold search, and warm search on Linux,
- engine-resident heap after releasing the input corpus,
- whether eta was adjusted, selected query positions, and the generated
  candidate count,
- the number of calls one search makes into the cost policy,
- the embedding matrix's shape, seed, build time, and heap, under
  `--costs cosine` only.

Warm searches report their mean, median, and minimum. Prefer the median: a mean
over few runs drifts with a single outlier, and comparing two implementations by
means over three runs has already produced a ratio that fifteen runs did not
confirm. Raise `--warm-runs` when comparing changes rather than reading one run.

`substitution_calls` counts every `EditCosts::substitution` call one search
makes, across both phases: filtering evaluates the vocabulary against each
query position, and verification evaluates aligned token pairs. It isolates the
vocabulary scans only on a filtering-dominated workload, such as the rare-token
query above, where the candidate count is small enough that verification
contributes little. Read it against `generated_candidates` to tell the two
apart. A change that removes a redundant scan then shows up here without
needing a timing. The counted search runs after the timed ones and uses a
counting wrapper around the same policy, so counting never enters a reported
duration.

`getrusage` supplies peak RSS on Linux and macOS. It is cumulative within the
process, so search RSS includes any earlier construction peak. Heap peaks are
reset at each phase, and the source text and line vector are released before the
search phases. `engine_resident_heap` is the allocator-observed current heap
after that release; use it rather than peak RSS to compare the resident engine
state. On unsupported operating systems peak RSS is reported as zero. Allocator
metrics exclude file-backed mmap pages. Linux file-backed RSS is read from
`/proc/self/smaps_rollup` as total RSS minus anonymous RSS. It includes other
mapped files as well as the index, so compare it with the in-memory baseline.
On macOS, use `vmmap` externally while a larger run is active.

To stress candidate generation with repeated postings, use a smaller corpus and
threshold one:

```console
cargo run --release -p yurine-benchmarks -- generate /tmp/yurine-overlap.txt \
  --sequences 2000 --tokens 32 --vocabulary 32 --hot-vocabulary 4 --seed 1
cargo run --release -p yurine-benchmarks -- measure /tmp/yurine-overlap.txt \
  --threshold 1
```

When `--eta` is omitted, the benchmark uses the search engine's automatic eta,
matching the public API default. Pass `--eta COST` to measure an explicit radius.

The original implementation kept a `HashSet` beside the candidate `Vec` to
remove duplicates. The baseline analysis established that candidates are
already unique: selected query positions and neighborhood symbols are unique,
each posting list is deduplicated, and posting lists for different symbols do
not overlap. The `HashSet` was therefore removed; allocator-observed search heap
growth remains the implementation-independent memory metric.

`eta_was_adjusted` is emitted as numeric boolean `0` or `1`. An eta too small
to construct a threshold subsequence is raised to the smallest one that can, so
a run reporting `1` filtered at a radius wider than the one it started from: the
automatic eta when `--eta` is omitted, and the explicit radius otherwise. Read it
against `generated_candidates`.

The library compatibility integration fixture separately fixes `SequenceId`,
token range, weighted distance, and result order. CLI tests fix conversion from token
ranges to UTF-8 byte ranges and matched source text. The benchmark package's unit
tests check that corpus and embedding generation are reproducible, and that
clustered embeddings separate near tokens from far ones.
