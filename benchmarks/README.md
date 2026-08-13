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

Output is tab-separated `metric`, `value`, and `unit`. It includes:

- source corpus and persistent-index sizes (the baseline has no persistent index),
- corpus-load, engine-build, cold-search, and warm-search timings,
- allocator-observed heap peaks for each phase,
- process peak RSS after build, cold search, and warm search,
- engine-resident heap after releasing the input corpus,
- whether exhaustive verification was used, selected query positions, and the
  generated candidate count.

`getrusage` supplies peak RSS on Linux and macOS. It is cumulative within the
process, so search RSS includes any earlier construction peak. Heap peaks are
reset at each phase, and the source text and line vector are released before the
search phases. `engine_resident_heap` is the allocator-observed current heap
after that release; use it rather than peak RSS to compare the resident engine
state. On unsupported operating systems RSS is reported as zero. The
current implementation has no file-backed mmap pages, so anonymous/file-backed
RSS is not split. On Linux, record that split externally with
`/proc/$PID/smaps_rollup`; on macOS use `vmmap` while a larger run is active.
Later mmap benchmarks should use the same corpus and report the split when the
host exposes it.

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
