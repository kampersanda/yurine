use std::alloc::{GlobalAlloc, Layout, System};
use std::error::Error;
use std::fs::{self, File};
use std::io::BufWriter;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use yurine::costs::Cost;
use yurine::costs::levenshtein::LevenshteinCosts;
use yurine::search::SearchEngineBuilder;
use yurine::search::range_search::RangeSearchParams;
use yurine::tokenization::whitespace::WhitespaceTokenizer;
use yurine_benchmarks::{CorpusConfig, DEFAULT_QUERY, write_corpus};

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

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    match arguments.next().as_deref() {
        Some("generate") => generate(arguments.collect()),
        Some("measure") => measure(arguments.collect()),
        _ => Err(usage().into()),
    }
}

fn generate(arguments: Vec<String>) -> Result<(), Box<dyn Error>> {
    let output = arguments.first().ok_or_else(usage)?;
    let mut config = CorpusConfig::default();
    let mut index = 1;
    while index < arguments.len() {
        let option = &arguments[index];
        let value = arguments.get(index + 1).ok_or_else(usage)?;
        match option.as_str() {
            "--strings" => config.strings = value.parse()?,
            "--tokens" => config.tokens_per_string = value.parse()?,
            "--vocabulary" => config.vocabulary = value.parse()?,
            "--hot-vocabulary" => config.hot_vocabulary = value.parse()?,
            "--seed" => config.seed = value.parse()?,
            _ => return Err(format!("unknown option: {option}\n{}", usage()).into()),
        }
        index += 2;
    }

    let file = File::create(output)?;
    write_corpus(BufWriter::new(file), config)?;
    println!("generated\t{}\tbytes", fs::metadata(output)?.len());
    println!("strings\t{}\tcount", config.strings);
    println!("tokens_per_string\t{}\tcount", config.tokens_per_string);
    println!("vocabulary\t{}\tcount", config.vocabulary);
    println!("hot_vocabulary\t{}\tcount", config.hot_vocabulary);
    println!("seed\t{}\tu64", config.seed);
    Ok(())
}

fn measure(arguments: Vec<String>) -> Result<(), Box<dyn Error>> {
    let corpus_path = arguments.first().ok_or_else(usage)?;
    let mut query = DEFAULT_QUERY.to_owned();
    let mut threshold = Cost::ZERO;
    let mut eta = Cost::ZERO;
    let mut warm_runs = 5usize;
    let mut index = 1;
    while index < arguments.len() {
        let option = &arguments[index];
        let value = arguments.get(index + 1).ok_or_else(usage)?;
        match option.as_str() {
            "--query" => query = value.to_owned(),
            "--threshold" => threshold = Cost::new(value.parse()?)?,
            "--eta" => eta = Cost::new(value.parse()?)?,
            "--warm-runs" => warm_runs = value.parse()?,
            _ => return Err(format!("unknown option: {option}\n{}", usage()).into()),
        }
        index += 2;
    }
    if warm_runs == 0 {
        return Err("--warm-runs must be greater than zero".into());
    }

    let load_heap_start = reset_heap_peak();
    let load_start = Instant::now();
    let contents = fs::read_to_string(corpus_path)?;
    let corpus: Vec<_> = contents.lines().map(str::to_owned).collect();
    let load_elapsed = load_start.elapsed();
    let load_heap_peak = heap_peak();

    let build_heap_start = reset_heap_peak();
    let build_start = Instant::now();
    let mut builder = SearchEngineBuilder::new(WhitespaceTokenizer::new(), LevenshteinCosts::new());
    for string in &corpus {
        builder.add_string(string)?;
    }
    let engine = builder.build()?;
    let build_elapsed = build_start.elapsed();
    let build_heap_peak = heap_peak();
    let peak_rss_after_build = peak_rss_bytes();

    let params = RangeSearchParams::new(threshold).with_eta(eta);
    let cold_heap_start = reset_heap_peak();
    let cold_start = Instant::now();
    let (cold_matches, metrics) = engine.range_search_with_metrics(&query, &params)?;
    let cold_elapsed = cold_start.elapsed();
    let cold_heap_peak = heap_peak();
    let peak_rss_after_cold = peak_rss_bytes();

    let warm_heap_start = reset_heap_peak();
    let mut warm_elapsed = Duration::ZERO;
    let mut warm_matches = 0usize;
    for _ in 0..warm_runs {
        let start = Instant::now();
        warm_matches = engine.range_search(&query, &params)?.len();
        warm_elapsed += start.elapsed();
    }
    let warm_heap_peak = heap_peak();
    let peak_rss_after_warm = peak_rss_bytes();

    metric(
        "source_corpus_bytes",
        fs::metadata(corpus_path)?.len(),
        "bytes",
    );
    metric("persistent_index_bytes", 0, "bytes");
    metric("corpus_strings", corpus.len(), "count");
    metric("corpus_load_elapsed", load_elapsed.as_nanos(), "ns");
    heap_metrics("corpus_load", load_heap_start, load_heap_peak);
    metric("build_elapsed", build_elapsed.as_nanos(), "ns");
    heap_metrics("build", build_heap_start, build_heap_peak);
    metric("peak_rss_after_build", peak_rss_after_build, "bytes");
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
    metric("cold_match_count", cold_matches.len(), "count");
    metric("warm_match_count", warm_matches, "count");
    metric(
        "selected_query_positions",
        metrics.selected_query_positions,
        "count",
    );
    metric("raw_candidates", metrics.raw_candidates, "count");
    metric("unique_candidates", metrics.unique_candidates, "count");
    metric(
        "candidate_duplicate_rate",
        metrics.duplicate_rate(),
        "ratio",
    );
    metric(
        "candidate_vec_payload_capacity",
        metrics.candidate_vec_payload_bytes(),
        "bytes",
    );
    metric(
        "dedup_set_key_capacity",
        metrics.dedup_set_key_capacity_bytes(),
        "bytes",
    );
    Ok(())
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

fn usage() -> String {
    format!(
        "usage:\n  yurine-baseline generate CORPUS [--strings N] [--tokens N] [--vocabulary N] [--hot-vocabulary N] [--seed N]\n  yurine-baseline measure CORPUS [--query QUERY] [--threshold COST] [--eta COST] [--warm-runs N]\n\ndefault query: {DEFAULT_QUERY}"
    )
}
