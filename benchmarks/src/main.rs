use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeSet;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use yurine::costs::{CosineEmbeddingCosts, EditCosts, LevenshteinCosts};
use yurine::persistence::StringCodec;
use yurine::search::RangeSearchMetrics;
use yurine::{Cost, RangeSearchParams, SearchEngine, SearchEngineBuilder};
use yurine_benchmarks::{
    CorpusConfig, DEFAULT_QUERY_SOURCE_TEXT, EmbeddingConfig, build_embedding_store,
    write_data_sequences,
};

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

    /// Vocabulary size (4..=1000000).
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

    #[arg(long, default_value = "5")]
    warm_runs: NonZeroUsize,

    /// Edit-cost policy under measurement.
    #[arg(long, value_enum, default_value_t = CostsPolicy::Levenshtein)]
    costs: CostsPolicy,

    /// Synthetic embedding dimension, for cosine costs.
    #[arg(long, default_value_t = EmbeddingConfig::default().dimension)]
    dimension: NonZeroUsize,

    /// Number of synthetic embedding clusters, for cosine costs.
    #[arg(long, default_value_t = EmbeddingConfig::default().clusters)]
    embedding_clusters: NonZeroUsize,

    /// Cosine similarity within a cluster, for cosine costs.
    #[arg(long, default_value_t = EmbeddingConfig::default().cohesion)]
    embedding_cohesion: f32,

    #[arg(long, default_value_t = EmbeddingConfig::default().seed)]
    embedding_seed: u64,

    /// Save and reopen the engine from this mmap-backed snapshot path.
    #[arg(long)]
    persistent_index: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CostsPolicy {
    Levenshtein,
    Cosine,
}

impl CostsPolicy {
    const fn name(self) -> &'static str {
        match self {
            Self::Levenshtein => "levenshtein",
            Self::Cosine => "cosine",
        }
    }
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
    let mut builder = SearchEngineBuilder::new();
    for source_text in &source_texts {
        builder.add_sequence(source_text.split_whitespace().map(str::to_owned))?;
    }
    let owned_engine = builder.build()?;
    let build_elapsed = build_start.elapsed();
    let build_heap_peak = heap_peak();
    let peak_rss_after_build = peak_rss_bytes();
    let mut save_elapsed = Duration::ZERO;
    let mut open_elapsed = Duration::ZERO;
    let mut open_heap_start = 0;
    let mut open_heap_peak = 0;
    let persistent_index_bytes;
    let engine: SearchEngine<String> = if let Some(path) = &options.persistent_index {
        let save_start = Instant::now();
        owned_engine.save_with(path, &StringCodec)?;
        save_elapsed = save_start.elapsed();
        persistent_index_bytes = fs::metadata(path)?.len();
        drop(owned_engine);
        open_heap_start = reset_heap_peak();
        let open_start = Instant::now();
        let engine = SearchEngine::open_with(path, &StringCodec)?;
        open_elapsed = open_start.elapsed();
        open_heap_peak = heap_peak();
        engine
    } else {
        persistent_index_bytes = 0;
        owned_engine
    };
    let embedding_config = EmbeddingConfig {
        dimension: options.dimension,
        clusters: options.embedding_clusters,
        cohesion: options.embedding_cohesion,
        seed: options.embedding_seed,
    };
    let mut embedding_elapsed = Duration::ZERO;
    let mut embedding_heap_start = 0;
    let mut embedding_heap_peak = 0;
    let mut embedding_tokens = 0;
    let embeddings = if options.costs == CostsPolicy::Cosine {
        embedding_heap_start = reset_heap_peak();
        let embedding_start = Instant::now();
        let tokens = distinct_tokens(&source_texts);
        embedding_tokens = tokens.len();
        let store = build_embedding_store(&tokens, embedding_config)?;
        embedding_elapsed = embedding_start.elapsed();
        embedding_heap_peak = heap_peak();
        Some(store)
    } else {
        None
    };
    drop(source_texts);
    drop(source_contents);
    let file_backed_rss_after_open = file_backed_rss_bytes();
    // The engine, plus the embedding matrix under cosine costs, is all that
    // stays resident here. Sample it before the query sequence and the timing
    // buffer add anything, so the metric does not grow with `--warm-runs`.
    let engine_resident_heap = reset_heap_peak();

    let params = RangeSearchParams::new(options.threshold.into());
    let query_sequence: Vec<_> = options
        .query_source_text
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let searches = match embeddings {
        Some(embeddings) => run_searches(
            &engine,
            &CosineEmbeddingCosts::new(embeddings),
            &query_sequence,
            &params,
            warm_runs,
        )?,
        None => run_searches(
            &engine,
            &LevenshteinCosts::new(),
            &query_sequence,
            &params,
            warm_runs,
        )?,
    };
    let SearchOutcome {
        cold_elapsed,
        cold_heap_start,
        cold_heap_peak,
        peak_rss_after_cold,
        file_backed_rss_after_cold,
        cold_match_count,
        warm_mean_elapsed,
        warm_median_elapsed,
        warm_min_elapsed,
        warm_heap_start,
        warm_heap_peak,
        peak_rss_after_warm,
        file_backed_rss_after_warm,
        warm_matches,
        substitution_calls,
        metrics,
    } = searches;

    metric("costs_policy", options.costs.name(), "name");
    metric(
        "source_corpus_bytes",
        fs::metadata(options.corpus)?.len(),
        "bytes",
    );
    metric("persistent_index_bytes", persistent_index_bytes, "bytes");
    metric("corpus_strings", data_sequence_count, "count");
    metric("corpus_load_elapsed", load_elapsed.as_nanos(), "ns");
    heap_metrics("corpus_load", load_heap_start, load_heap_peak);
    metric("build_elapsed", build_elapsed.as_nanos(), "ns");
    heap_metrics("build", build_heap_start, build_heap_peak);
    metric("peak_rss_after_build", peak_rss_after_build, "bytes");
    metric("engine_resident_heap", engine_resident_heap, "bytes");
    metric("save_elapsed", save_elapsed.as_nanos(), "ns");
    metric("open_elapsed", open_elapsed.as_nanos(), "ns");
    heap_metrics("open", open_heap_start, open_heap_peak);
    metric(
        "file_backed_rss_after_open",
        file_backed_rss_after_open,
        "bytes",
    );
    if options.costs == CostsPolicy::Cosine {
        metric("embedding_tokens", embedding_tokens, "count");
        metric("embedding_dimension", embedding_config.dimension, "count");
        metric("embedding_clusters", embedding_config.clusters, "count");
        metric(
            "embedding_cohesion",
            embedding_config.cohesion,
            "similarity",
        );
        metric("embedding_seed", embedding_config.seed, "u64");
        metric("embedding_elapsed", embedding_elapsed.as_nanos(), "ns");
        heap_metrics("embedding", embedding_heap_start, embedding_heap_peak);
    }
    metric("cold_search_elapsed", cold_elapsed.as_nanos(), "ns");
    heap_metrics("cold_search", cold_heap_start, cold_heap_peak);
    metric("peak_rss_after_cold_search", peak_rss_after_cold, "bytes");
    metric(
        "file_backed_rss_after_cold_search",
        file_backed_rss_after_cold,
        "bytes",
    );
    metric("warm_search_runs", warm_runs, "count");
    metric(
        "warm_search_mean_elapsed",
        warm_mean_elapsed.as_nanos(),
        "ns",
    );
    metric(
        "warm_search_median_elapsed",
        warm_median_elapsed.as_nanos(),
        "ns",
    );
    metric("warm_search_min_elapsed", warm_min_elapsed.as_nanos(), "ns");
    heap_metrics("warm_search", warm_heap_start, warm_heap_peak);
    metric("peak_rss_after_warm_search", peak_rss_after_warm, "bytes");
    metric(
        "file_backed_rss_after_warm_search",
        file_backed_rss_after_warm,
        "bytes",
    );
    metric("cold_match_count", cold_match_count, "count");
    metric("warm_match_count", warm_matches, "count");
    metric(
        "eta_was_adjusted",
        u8::from(metrics.adjusted_eta.is_some()),
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
    metric("substitution_calls", substitution_calls, "count");
    Ok(())
}

/// Collects the corpus vocabulary in sorted order.
fn distinct_tokens(source_texts: &[String]) -> Vec<String> {
    source_texts
        .iter()
        .flat_map(|source_text| source_text.split_whitespace())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// Measurements of one cold search, `warm_runs` warm searches, and one counted
/// search.
struct SearchOutcome {
    cold_elapsed: Duration,
    cold_heap_start: usize,
    cold_heap_peak: usize,
    peak_rss_after_cold: u64,
    file_backed_rss_after_cold: u64,
    cold_match_count: usize,
    warm_mean_elapsed: Duration,
    warm_median_elapsed: Duration,
    warm_min_elapsed: Duration,
    warm_heap_start: usize,
    warm_heap_peak: usize,
    peak_rss_after_warm: u64,
    file_backed_rss_after_warm: u64,
    warm_matches: usize,
    substitution_calls: usize,
    metrics: RangeSearchMetrics,
}

/// Runs the search workload under one edit-cost policy.
///
/// The policy is borrowed rather than moved so the same value serves both the
/// timed searchers and the counted one; a cosine policy owns its whole
/// embedding matrix, and cloning it would measure a workload with twice the
/// resident state.
fn run_searches<C>(
    engine: &SearchEngine<String>,
    costs: &C,
    query_sequence: &[String],
    params: &RangeSearchParams,
    warm_runs: usize,
) -> Result<SearchOutcome, Box<dyn Error>>
where
    C: EditCosts<String>,
{
    let mut warm_samples = Vec::with_capacity(warm_runs);
    let searcher = engine.range_searcher(BorrowedCosts(costs));

    let cold_heap_start = reset_heap_peak();
    let cold_start = Instant::now();
    let (cold_matches, metrics) = searcher.search_with_metrics(query_sequence, params)?;
    let cold_elapsed = cold_start.elapsed();
    let cold_heap_peak = heap_peak();
    let peak_rss_after_cold = peak_rss_bytes();
    let file_backed_rss_after_cold = file_backed_rss_bytes();
    let cold_match_count = cold_matches.len();
    drop(cold_matches);

    let warm_heap_start = reset_heap_peak();
    let mut warm_matches = 0usize;
    for _ in 0..warm_runs {
        let start = Instant::now();
        let matches = searcher.search(query_sequence, params)?;
        warm_samples.push(start.elapsed());
        warm_matches = matches.len();
    }
    let warm_heap_peak = heap_peak();
    let peak_rss_after_warm = peak_rss_bytes();
    let file_backed_rss_after_warm = file_backed_rss_bytes();

    // Counting runs last so that its extra work per substitution stays outside
    // every reported duration and heap phase.
    let substitution_calls = Cell::new(0);
    let counting_searcher = engine.range_searcher(CountingCosts {
        costs,
        substitution_calls: &substitution_calls,
    });
    counting_searcher.search(query_sequence, params)?;

    let warm_total: Duration = warm_samples.iter().sum();
    warm_samples.sort_unstable();
    Ok(SearchOutcome {
        cold_elapsed,
        cold_heap_start,
        cold_heap_peak,
        peak_rss_after_cold,
        file_backed_rss_after_cold,
        cold_match_count,
        warm_mean_elapsed: warm_total / warm_runs as u32,
        warm_median_elapsed: median(&warm_samples),
        warm_min_elapsed: warm_samples[0],
        warm_heap_start,
        warm_heap_peak,
        peak_rss_after_warm,
        file_backed_rss_after_warm,
        warm_matches,
        substitution_calls: substitution_calls.get(),
        metrics,
    })
}

/// Returns the median of ascending `samples`, averaging the two central ones
/// for an even count.
fn median(samples: &[Duration]) -> Duration {
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2
    } else {
        samples[middle]
    }
}

/// Lends one cost policy to a searcher, which owns whatever it is given.
struct BorrowedCosts<'a, C>(&'a C);

impl<T, C> EditCosts<T> for BorrowedCosts<'_, C>
where
    C: EditCosts<T>,
{
    fn substitution(&self, from: &T, to: &T) -> Cost {
        self.0.substitution(from, to)
    }

    fn deletion(&self, token: &T) -> Cost {
        self.0.deletion(token)
    }

    fn insertion(&self, token: &T) -> Cost {
        self.0.insertion(token)
    }
}

/// Counts substitution evaluations, the work filtering spends on the
/// vocabulary.
///
/// Counting here rather than inside the library keeps the count available
/// without widening the public search API.
struct CountingCosts<'a, C> {
    costs: &'a C,
    substitution_calls: &'a Cell<usize>,
}

impl<T, C> EditCosts<T> for CountingCosts<'_, C>
where
    C: EditCosts<T>,
{
    fn substitution(&self, from: &T, to: &T) -> Cost {
        self.substitution_calls
            .set(self.substitution_calls.get() + 1);
        self.costs.substitution(from, to)
    }

    fn deletion(&self, token: &T) -> Cost {
        self.costs.deletion(token)
    }

    fn insertion(&self, token: &T) -> Cost {
        self.costs.insertion(token)
    }
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

#[cfg(target_os = "linux")]
fn file_backed_rss_bytes() -> u64 {
    let Ok(status) = fs::read_to_string("/proc/self/smaps_rollup") else {
        return 0;
    };
    let kilobytes = |name: &str| {
        status
            .lines()
            .find_map(|line| {
                line.strip_prefix(name)?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()
            })
            .unwrap_or(0)
    };
    kilobytes("Rss:")
        .saturating_sub(kilobytes("Anonymous:"))
        .saturating_mul(1024)
}

#[cfg(not(target_os = "linux"))]
fn file_backed_rss_bytes() -> u64 {
    0
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn peak_rss_bytes() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::Duration;

    use clap::{CommandFactory, Parser};

    use super::{Command, CostsPolicy, GenerateOptions, Options, generate, median};
    use yurine::costs::Cost;
    use yurine_benchmarks::{CorpusConfig, DEFAULT_QUERY_SOURCE_TEXT, EmbeddingConfig};

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
            "--warm-runs",
            "3",
        ])
        .unwrap();
        let Command::Measure(options) = options.command else {
            panic!("expected measure command");
        };

        assert_eq!(options.query_source_text, DEFAULT_QUERY_SOURCE_TEXT);
        assert_eq!(options.threshold, Cost::new_const(1.5));
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
    }

    #[test]
    fn median_averages_the_two_central_samples_of_an_even_count() {
        let samples: Vec<_> = [10, 20, 30, 50].map(Duration::from_nanos).to_vec();

        assert_eq!(median(&samples), Duration::from_nanos(25));
        assert_eq!(median(&samples[..3]), Duration::from_nanos(20));
        assert_eq!(median(&samples[..1]), Duration::from_nanos(10));
    }

    #[test]
    fn measures_levenshtein_costs_unless_asked_for_cosine() {
        let defaults =
            Options::try_parse_from(["yurine-baseline", "measure", "corpus.txt"]).unwrap();
        let Command::Measure(defaults) = defaults.command else {
            panic!("expected measure command");
        };

        assert_eq!(defaults.costs, CostsPolicy::Levenshtein);
        assert_eq!(defaults.dimension, EmbeddingConfig::default().dimension);

        let cosine = Options::try_parse_from([
            "yurine-baseline",
            "measure",
            "corpus.txt",
            "--costs",
            "cosine",
            "--dimension",
            "128",
            "--embedding-clusters",
            "16",
            "--embedding-cohesion",
            "0.9",
        ])
        .unwrap();
        let Command::Measure(cosine) = cosine.command else {
            panic!("expected measure command");
        };

        assert_eq!(cosine.costs, CostsPolicy::Cosine);
        assert_eq!(cosine.dimension.get(), 128);
        assert_eq!(cosine.embedding_clusters.get(), 16);
        assert_eq!(cosine.embedding_cohesion, 0.9);
        assert!(
            Options::try_parse_from([
                "yurine-baseline",
                "measure",
                "corpus.txt",
                "--costs",
                "jaccard",
            ])
            .is_err()
        );
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
