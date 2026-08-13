# Baseline benchmark

This package provides the reproducible workload used to compare the in-memory
implementation with the persistent index work tracked by issue #24. Measurements
are observations, not CI pass/fail thresholds. Keep generated corpora and
measurement results outside the repository; only the code and reproduction
procedure are versioned.

Generate the default whitespace-tokenized corpus:

```console
cargo run --release -p yurine-benchmarks -- generate /tmp/yurine-baseline.txt
```

The generator uses a fixed seed, 20,000 data sequences, 32 tokens per sequence,
a 256-token vocabulary, and an eight-token hot set. Every 128th sequence starts
with the default query sequence, so the workload always has known exact matches.
Every setting can be overridden; rerunning the same command produces identical
bytes.

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

Output is tab-separated `metric`, `value`, and `unit`. It includes:

- source corpus and persistent-index sizes,
- corpus-load, engine-build, cold-search, and warm-search timings,
- snapshot-save and mmap-open timings,
- allocator-observed heap peaks for each phase,
- process peak RSS after build, cold search, and warm search,
- current file-backed RSS after open, cold search, and warm search on Linux,
- engine-resident heap after releasing the input corpus,
- whether exhaustive verification was used, selected query positions, and the
  generated candidate count.

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

`used_exhaustive_verification` is emitted as numeric boolean `0` or `1`.

The library compatibility integration fixture separately fixes `SequenceId`,
token range, weighted distance, and result order. CLI tests fix conversion from token
ranges to UTF-8 byte ranges and matched source text. The benchmark package's unit
test checks that corpus generation is byte-for-byte reproducible.
