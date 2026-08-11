# Baseline benchmark

This package provides the reproducible workload used to compare the in-memory
implementation with the persistent index work tracked by issue #24. Measurements
are observations, not CI pass/fail thresholds.

Generate the default whitespace-tokenized corpus:

```console
cargo run --release -p yurine-benchmarks -- generate /tmp/yurine-baseline.txt
```

The generator uses a fixed seed, 20,000 strings, 32 tokens per string, a
256-token vocabulary, and an eight-token hot set. Every 128th string starts with
the default query, so the workload always has known exact matches. Every setting
can be overridden; rerunning the same command produces identical bytes.

Measure construction and one cold plus five warm searches:

```console
cargo run --release -p yurine-benchmarks -- measure /tmp/yurine-baseline.txt
```

Output is tab-separated `metric`, `value`, and `unit`. It includes:

- source corpus and persistent-index sizes (the baseline has no persistent index),
- corpus-load, engine-build, cold-search, and warm-search timings,
- allocator-observed heap peaks for each phase,
- process peak RSS after build, cold search, and warm search,
- whether exhaustive verification was used, selected query positions, and the
  generated candidate count.

`getrusage` supplies peak RSS on Linux and macOS. It is cumulative within the
process, so search RSS includes any earlier construction peak. Heap peaks are
reset at each phase, and the source text and line vector are released before the
search phases. On unsupported operating systems RSS is reported as zero. The
current implementation has no file-backed mmap pages, so anonymous/file-backed
RSS is not split. On Linux, record that split externally with
`/proc/$PID/smaps_rollup`; on macOS use `vmmap` while a larger run is active.
Later mmap benchmarks should use the same corpus and report the split when the
host exposes it.

To stress candidate generation with repeated postings, use a smaller corpus and
threshold one:

```console
cargo run --release -p yurine-benchmarks -- generate /tmp/yurine-overlap.txt \
  --strings 2000 --tokens 32 --vocabulary 32 --hot-vocabulary 4 --seed 1
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

The compatibility integration fixture separately fixes `StringId`, token range,
UTF-8 byte range, weighted distance, and result order. The benchmark package's
unit test checks that corpus generation is byte-for-byte reproducible.

Host-specific observations are stored in [`results/`](results/). They are
reference snapshots only and are not compared by CI. The initial macOS snapshot
captures the original `Vec + HashSet` path; the `without-candidate-dedup`
snapshot records the same workload after proving candidates unique and removing
the redundant set.
