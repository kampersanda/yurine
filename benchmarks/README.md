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
- raw and unique candidate counts, duplicate rate, and the payload capacities of
  the candidate `Vec` and deduplication `HashSet`.

`getrusage` supplies peak RSS on Linux and macOS. It is cumulative within the
process, while heap peaks are reset at each phase. On unsupported operating
systems RSS is reported as zero. The current implementation has no file-backed
mmap pages, so anonymous/file-backed RSS is not split. On Linux, record that
split externally with `/proc/$PID/smaps_rollup`; on macOS use `vmmap` while a
larger run is active. Later mmap benchmarks should use the same corpus and report
the split when the host exposes it.

To stress overlapping postings and the `Vec + HashSet` candidate path, use a
smaller corpus and threshold one:

```console
cargo run --release -p yurine-benchmarks -- generate /tmp/yurine-overlap.txt \
  --strings 2000 --tokens 32 --vocabulary 32 --hot-vocabulary 4 --seed 1
cargo run --release -p yurine-benchmarks -- measure /tmp/yurine-overlap.txt \
  --threshold 1 --eta 0
```

The compatibility integration fixture separately fixes `StringId`, token range,
UTF-8 byte range, weighted distance, and result order. The benchmark package's
unit test checks that corpus generation is byte-for-byte reproducible.

Host-specific observations are stored in [`results/`](results/). They are
reference snapshots only and are not compared by CI.
