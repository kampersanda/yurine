use std::alloc::{GlobalAlloc, Layout, System};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use yurine::costs::Cost;
use yurine::costs::levenshtein::LevenshteinCosts;
use yurine::search::SearchEngineBuilder;
use yurine::search::range_search::RangeSearchParams;
use yurine_benchmarks::{CorpusConfig, DEFAULT_QUERY_SOURCE_TEXT, write_data_sequences};

struct TrackingAllocator;

static CURRENT_HEAP_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_HEAP_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        CURRENT_HEAP_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            if new_size >= layout.size() {
                record_allocation(new_size - layout.size());
            } else {
                CURRENT_HEAP_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_pointer
    }
}

fn record_allocation(bytes: usize) {
    let current = CURRENT_HEAP_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK_HEAP_BYTES.fetch_max(current, Ordering::Relaxed);
}

fn reset_heap_peak() -> usize {
    let current = CURRENT_HEAP_BYTES.load(Ordering::Relaxed);
    PEAK_HEAP_BYTES.store(current, Ordering::Relaxed);
    current
}

fn heap_peak() -> usize {
    PEAK_HEAP_BYTES.load(Ordering::Relaxed)
}

#[derive(Debug, Parser)]
#[command(name = "yurine-baseline")]
struct Options {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generates a deterministic synthetic corpus.
    Generate(GenerateOptions),
    /// Measures in-memory construction and search.
    Measure(MeasureOptions),
}

#[derive(Debug, Args)]
struct GenerateOptions {
    /// Output corpus path.
    output: PathBuf,

    #[arg(long, default_value_t = CorpusConfig::default().sequences)]
    sequences: usize,

    #[arg(long, default_value_t = CorpusConfig::default().tokens_per_sequence)]
    tokens: usize,

    /// Vocabulary size (4..=10000).
    #[arg(long, default_value_t = CorpusConfig::default().vocabulary)]
    vocabulary: usize,

    #[arg(long, default_value_t = CorpusConfig::default().hot_vocabulary)]
    hot_vocabulary: usize,

    #[arg(long, default_value_t = CorpusConfig::default().seed)]
    seed: u64,
}

#[derive(Debug, Args)]
struct MeasureOptions {
    /// Input corpus path.
    corpus: PathBuf,

    #[arg(long = "query", default_value = DEFAULT_QUERY_SOURCE_TEXT)]
    query_source_text: String,

    #[arg(long, default_value = "0", value_parser = parse_cost)]
    threshold: Cost,

    #[arg(long, value_parser = parse_cost)]
    eta: Option<Cost>,

    #[arg(long, default_value = "5")]
    warm_runs: NonZeroUsize,
}

fn main() -> Result<(), Box<dyn Error>> {
    match Options::parse().command {
        Command::Generate(options) => generate(options),
        Command::Measure(options) => measure(options),
    }
}

fn generate(options: GenerateOptions) -> Result<(), Box<dyn Error>> {
    let config = CorpusConfig {
        sequences: options.sequences,
        tokens_per_sequence: options.tokens,
        vocabulary: options.vocabulary,
        hot_vocabulary: options.hot_vocabulary,
        seed: options.seed,
    };
    config.validate()?;
    let mut writer = BufWriter::new(File::create(&options.output)?);
    write_data_sequences(&mut writer, config)?;
    writer.flush()?;
    println!("generated\t{}\tbytes", fs::metadata(options.output)?.len());
    println!("sequences\t{}\tcount", config.sequences);
    println!("tokens_per_sequence\t{}\tcount", config.tokens_per_sequence);
    println!("vocabulary\t{}\tcount", config.vocabulary);
    println!("hot_vocabulary\t{}\tcount", config.hot_vocabulary);
    println!("seed\t{}\tu64", config.seed);
    Ok(())
}

fn measure(options: MeasureOptions) -> Result<(), Box<dyn Error>> {
    let warm_runs = options.warm_runs.get();
    let load_heap_start = reset_heap_peak();
    let load_start = Instant::now();
    let source_contents = fs::read_to_string(&options.corpus)?;
    let source_texts: Vec<_> = source_contents.lines().map(str::to_owned).collect();
    let data_sequence_count = source_texts.len();
    let load_elapsed = load_start.elapsed();
    let load_heap_peak = heap_peak();

    let build_heap_start = reset_heap_peak();
    let build_start = Instant::now();
    let mut builder = SearchEngineBuilder::new(LevenshteinCosts::new());
    for source_text in &source_texts {
        builder.add_sequence(source_text.split_whitespace().map(str::to_owned))?;
    }
    let engine = builder.build()?;
    let build_elapsed = build_start.elapsed();
    let build_heap_peak = heap_peak();
    drop(source_texts);
    drop(source_contents);
    let peak_rss_after_build = peak_rss_bytes();

    let mut params = RangeSearchParams::new(options.threshold);
    if let Some(eta) = options.eta {
        params = params.with_eta(eta);
    }
    let cold_heap_start = reset_heap_peak();
    let engine_resident_heap = cold_heap_start;
    let cold_start = Instant::now();
    let query_sequence: Vec<_> = options
        .query_source_text
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let (cold_matches, metrics) = engine.range_search_with_metrics(&query_sequence, &params)?;
    let cold_elapsed = cold_start.elapsed();
    let cold_heap_peak = heap_peak();
    let peak_rss_after_cold = peak_rss_bytes();
    let cold_match_count = cold_matches.len();
    drop(cold_matches);

    let warm_heap_start = reset_heap_peak();
    let mut warm_elapsed = Duration::ZERO;
    let mut warm_matches = 0usize;
    for _ in 0..warm_runs {
        let start = Instant::now();
        let matches = engine.range_search(&query_sequence, &params)?;
        warm_elapsed += start.elapsed();
        warm_matches = matches.len();
    }
    let warm_heap_peak = heap_peak();
    let peak_rss_after_warm = peak_rss_bytes();

    metric(
        "source_corpus_bytes",
        fs::metadata(options.corpus)?.len(),
        "bytes",
    );
    // The in-memory baseline has no persistent index. Replace this when the
    // persistent-index benchmark is added.
    metric("persistent_index_bytes", 0, "bytes");
    metric("corpus_strings", data_sequence_count, "count");
    metric("corpus_load_elapsed", load_elapsed.as_nanos(), "ns");
    heap_metrics("corpus_load", load_heap_start, load_heap_peak);
    metric("build_elapsed", build_elapsed.as_nanos(), "ns");
    heap_metrics("build", build_heap_start, build_heap_peak);
    metric("peak_rss_after_build", peak_rss_after_build, "bytes");
    metric("engine_resident_heap", engine_resident_heap, "bytes");
    metric("cold_search_elapsed", cold_elapsed.as_nanos(), "ns");
    heap_metrics("cold_search", cold_heap_start, cold_heap_peak);
    metric("peak_rss_after_cold_search", peak_rss_after_cold, "bytes");
    metric(
        "warm_search_mean_elapsed",
        warm_elapsed.as_nanos() / warm_runs as u128,
        "ns",
    );
    heap_metrics("warm_search", warm_heap_start, warm_heap_peak);
    metric("peak_rss_after_warm_search", peak_rss_after_warm, "bytes");
    metric("cold_match_count", cold_match_count, "count");
    metric("warm_match_count", warm_matches, "count");
    metric(
        "used_exhaustive_verification",
        u8::from(metrics.used_exhaustive_verification),
        "bool",
    );
    metric(
        "selected_query_positions",
        metrics.selected_query_positions,
        "count",
    );
    metric(
        "generated_candidates",
        metrics.generated_candidates,
        "count",
    );
    Ok(())
}

fn parse_cost(text: &str) -> Result<Cost, String> {
    let value = text
        .parse::<f32>()
        .map_err(|_| "must be a non-negative finite number".to_owned())?;
    Cost::new(value).map_err(|_| "must be a non-negative finite number".to_owned())
}

fn heap_metrics(phase: &str, start: usize, peak: usize) {
    metric(&format!("{phase}_heap_start"), start, "bytes");
    metric(&format!("{phase}_heap_peak"), peak, "bytes");
    metric(
        &format!("{phase}_heap_peak_growth"),
        peak.saturating_sub(start),
        "bytes",
    );
}

fn metric(name: &str, value: impl std::fmt::Display, unit: &str) {
    println!("{name}\t{value}\t{unit}");
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return 0;
    }
    let rss = unsafe { usage.assume_init() }.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        rss
    } else {
        rss * 1024
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn peak_rss_bytes() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use std::fs;

    use clap::{CommandFactory, Parser};

    use super::{Command, GenerateOptions, Options, generate};
    use yurine::costs::Cost;
    use yurine_benchmarks::{CorpusConfig, DEFAULT_QUERY_SOURCE_TEXT};

    #[test]
    fn command_definition_is_valid() {
        Options::command().debug_assert();
    }

    #[test]
    fn parses_generate_defaults() {
        let options =
            Options::try_parse_from(["yurine-baseline", "generate", "corpus.txt"]).unwrap();
        let Command::Generate(options) = options.command else {
            panic!("expected generate command");
        };
        let defaults = CorpusConfig::default();

        assert_eq!(options.output.to_string_lossy(), "corpus.txt");
        assert_eq!(options.sequences, defaults.sequences);
        assert_eq!(options.tokens, defaults.tokens_per_sequence);
        assert_eq!(options.vocabulary, defaults.vocabulary);
        assert_eq!(options.hot_vocabulary, defaults.hot_vocabulary);
        assert_eq!(options.seed, defaults.seed);
    }

    #[test]
    fn parses_measure_options_and_rejects_zero_warm_runs() {
        let options = Options::try_parse_from([
            "yurine-baseline",
            "measure",
            "corpus.txt",
            "--threshold",
            "1.5",
            "--eta",
            "0.25",
            "--warm-runs",
            "3",
        ])
        .unwrap();
        let Command::Measure(options) = options.command else {
            panic!("expected measure command");
        };

        assert_eq!(options.query_source_text, DEFAULT_QUERY_SOURCE_TEXT);
        assert_eq!(options.threshold, Cost::new_const(1.5));
        assert_eq!(options.eta, Some(Cost::new_const(0.25)));
        assert_eq!(options.warm_runs.get(), 3);
        assert!(
            Options::try_parse_from([
                "yurine-baseline",
                "measure",
                "corpus.txt",
                "--warm-runs",
                "0",
            ])
            .is_err()
        );

        let automatic = Options::try_parse_from([
            "yurine-baseline",
            "measure",
            "corpus.txt",
            "--threshold",
            "1",
        ])
        .unwrap();
        let Command::Measure(automatic) = automatic.command else {
            panic!("expected measure command");
        };
        assert_eq!(automatic.eta, None);
    }

    #[test]
    fn invalid_config_does_not_truncate_existing_output() {
        let output =
            std::env::temp_dir().join(format!("yurine-invalid-config-{}.txt", std::process::id()));
        fs::write(&output, "keep me").unwrap();

        let result = generate(GenerateOptions {
            output: output.clone(),
            sequences: 1,
            tokens: 1,
            vocabulary: 1,
            hot_vocabulary: 1,
            seed: 0,
        });

        assert!(result.is_err());
        assert_eq!(fs::read_to_string(&output).unwrap(), "keep me");
        fs::remove_file(output).unwrap();
    }
}
