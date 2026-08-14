use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufRead, BufReader, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use csv::{Terminator, WriterBuilder};
use yurine::persistence::{CharCodec, StringCodec, TokenCodec};
use yurine::{Cost, Match, RangeSearchParams, SearchEngine, SearchEngineBuilder};

mod cost_config;
mod cost_snapshot;
mod index;
mod tokenization;

use cost_config::RuntimeCosts;
use index::{SourceReader, SourceWriter};
use tokenization::{CharacterTokenizer, Tokenizer, TokenizerKind, WhitespaceTokenizer};

/// Search newline-delimited source texts with edit distance using Yurine.
#[derive(Debug, Parser, PartialEq)]
#[command(version)]
struct Options {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand, PartialEq)]
enum Command {
    /// Build a reusable index from newline-delimited source texts.
    Index(IndexOptions),
    /// Compile an edit-cost configuration into a reusable snapshot.
    Costs(CostsOptions),
    /// Search a prebuilt index.
    Search(SearchOptions),
}

#[derive(Debug, Args, PartialEq)]
struct IndexOptions {
    /// Directory to write the index to; created if it does not exist.
    #[arg(value_name = "INDEX")]
    index: PathBuf,

    /// File containing one source text per line; reads standard input if omitted or '-'.
    corpus: Option<PathBuf>,

    /// Tokenization strategy; queries reuse it through the index.
    #[arg(long, value_enum, default_value_t = TokenizerKind::Character)]
    tokenizer: TokenizerKind,

    /// Report the elapsed time of each stage on standard error.
    ///
    /// Each stage is measured once, without warm-up or repetition. Reading the
    /// corpus from standard input includes waiting for the upstream process.
    #[arg(long)]
    timing: bool,
}

#[derive(Debug, Args, PartialEq)]
struct CostsOptions {
    /// JSON file describing the edit-cost policy.
    #[arg(value_name = "COSTS")]
    costs: PathBuf,

    /// Directory to write the snapshot to; created if it does not exist.
    #[arg(value_name = "SNAPSHOT")]
    snapshot: PathBuf,

    /// Tokenization of the indexes the snapshot is searched with.
    #[arg(long, value_enum, default_value_t = TokenizerKind::Character)]
    tokenizer: TokenizerKind,

    /// Report the elapsed time of each stage on standard error.
    ///
    /// Each stage is measured once, without warm-up or repetition.
    #[arg(long)]
    timing: bool,
}

#[derive(Debug, Args, PartialEq)]
struct SearchOptions {
    /// Directory holding an index built by the 'index' command.
    #[arg(value_name = "INDEX")]
    index: PathBuf,

    /// Query source text.
    #[arg(value_name = "QUERY")]
    query_source_text: String,

    /// Maximum edit distance.
    #[arg(short, long, default_value = "0", value_parser = parse_threshold)]
    threshold: Cost,

    /// Substitution-neighborhood radius; calculated automatically if omitted.
    #[arg(long, value_parser = parse_cost)]
    eta: Option<Cost>,

    /// JSON file describing the edit-cost policy, or a snapshot directory
    /// built by the 'costs' command.
    #[arg(long)]
    costs: Option<PathBuf>,

    /// Check the internal integrity of the search index and the edit costs
    /// before searching.
    ///
    /// The stored source texts are not checked, so a match is reported as
    /// stored even if they no longer agree with the search index.
    #[arg(long)]
    verify: bool,

    /// Report the elapsed time of each stage on standard error.
    ///
    /// Each stage is measured once, without warm-up or repetition.
    #[arg(long)]
    timing: bool,
}

fn main() -> ExitCode {
    let result = match Options::parse().command {
        Command::Index(options) => run_index(&options),
        Command::Costs(options) => run_costs(&options),
        Command::Search(options) => run_search(&options),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_index(options: &IndexOptions) -> Result<()> {
    let started = Instant::now();
    let mut timings = match options.tokenizer {
        TokenizerKind::Character => build_index(options, CharacterTokenizer, &CharCodec),
        TokenizerKind::Whitespace => build_index(options, WhitespaceTokenizer, &StringCodec),
    }
    .context("failed to build the index")?;

    if options.timing {
        timings.total = started.elapsed();
        write_index_timings(io::stderr().lock(), &timings).context("failed to write timings")?;
    }
    Ok(())
}

fn run_costs(options: &CostsOptions) -> Result<()> {
    let started = Instant::now();
    let mut timings = match options.tokenizer {
        TokenizerKind::Character => build_costs(options, CharacterTokenizer, &CharCodec),
        TokenizerKind::Whitespace => build_costs(options, WhitespaceTokenizer, &StringCodec),
    }
    .context("failed to build the cost snapshot")?;

    if options.timing {
        timings.total = started.elapsed();
        write_costs_timings(io::stderr().lock(), &timings).context("failed to write timings")?;
    }
    Ok(())
}

fn run_search(options: &SearchOptions) -> Result<()> {
    let started = Instant::now();
    let (matches, mut timings) = find_matches(options).context("search failed")?;

    write_matches(io::stdout().lock(), &matches).context("failed to write results")?;

    if options.timing {
        timings.total = started.elapsed();
        write_search_timings(io::stderr().lock(), &timings).context("failed to write timings")?;
    }
    Ok(())
}

fn parse_cost(text: &str) -> Result<Cost, String> {
    let value = text
        .parse::<f32>()
        .map_err(|_| "must be a non-negative finite number".to_owned())?;
    Cost::new(value).map_err(|_| "must be a non-negative finite number".to_owned())
}

fn parse_threshold(text: &str) -> Result<Cost, String> {
    let threshold = parse_cost(text)?;
    if threshold == Cost::MAX {
        Err("must be less than f32::MAX".to_owned())
    } else {
        Ok(threshold)
    }
}

fn build_index<T, C>(
    options: &IndexOptions,
    tokenizer: impl Tokenizer<Token = T>,
    codec: &C,
) -> Result<IndexTimings>
where
    T: Clone + Eq + Hash,
    C: TokenCodec<T>,
{
    let directory = options.index.as_path();
    std::fs::create_dir_all(directory)
        .with_context(|| format!("failed to create index directory '{}'", directory.display()))?;

    let read_start = Instant::now();
    let mut timings = IndexTimings::default();
    let mut sources = SourceWriter::create(directory)?;
    let mut builder = SearchEngineBuilder::new();
    for source_text in read_corpus(options.corpus.as_deref())?.lines() {
        let source_text = source_text.context("failed to read the corpus")?;
        let sequence: Vec<_> = tokenizer
            .tokenize(&source_text)
            .into_iter()
            .map(|token| token.value)
            .collect();
        builder.add_sequence(sequence)?;
        sources.push(&source_text)?;
    }
    let sequence_count = sources.finish()?;
    timings.read = read_start.elapsed();

    let build_start = Instant::now();
    let engine = builder.build()?;
    timings.build = build_start.elapsed();

    // Publish the new index only once every stage has succeeded, so a failed
    // run leaves the index of a previous one usable.
    let save_start = Instant::now();
    engine.save_with(index::engine_path(directory), codec)?;
    sources.publish()?;
    index::write_metadata(directory, options.tokenizer, sequence_count)?;
    timings.save = save_start.elapsed();

    Ok(timings)
}

fn build_costs<T, C>(
    options: &CostsOptions,
    tokenizer: impl Tokenizer<Token = T>,
    codec: &C,
) -> Result<CostsTimings>
where
    T: Clone + Eq + Hash,
    C: TokenCodec<T>,
{
    let directory = options.snapshot.as_path();
    std::fs::create_dir_all(directory).with_context(|| {
        format!(
            "failed to create cost snapshot directory '{}'",
            directory.display()
        )
    })?;

    let read_start = Instant::now();
    let mut timings = CostsTimings::default();
    let costs = cost_config::load(&options.costs, &tokenizer)?;
    timings.read = read_start.elapsed();

    let save_start = Instant::now();
    cost_snapshot::save(directory, &costs, options.tokenizer, codec)?;
    timings.save = save_start.elapsed();

    Ok(timings)
}

fn read_corpus(path: Option<&Path>) -> Result<Box<dyn BufRead>> {
    match path {
        Some(path) if path != Path::new("-") => {
            let file = File::open(path)
                .with_context(|| format!("failed to open corpus file '{}'", path.display()))?;
            Ok(Box::new(BufReader::new(file)))
        }
        _ => Ok(Box::new(io::stdin().lock())),
    }
}

fn find_matches(options: &SearchOptions) -> Result<(Vec<LocatedMatch>, SearchTimings)> {
    let metadata_start = Instant::now();
    let metadata = index::read_metadata(&options.index)?;
    let metadata_elapsed = metadata_start.elapsed();

    let (matches, mut timings) = match metadata.tokenizer {
        TokenizerKind::Character => search_index(options, metadata, CharacterTokenizer, &CharCodec),
        TokenizerKind::Whitespace => {
            search_index(options, metadata, WhitespaceTokenizer, &StringCodec)
        }
    }?;
    timings.open += metadata_elapsed;
    Ok((matches, timings))
}

fn search_index<T, C>(
    options: &SearchOptions,
    metadata: index::Metadata,
    tokenizer: impl Tokenizer<Token = T>,
    codec: &C,
) -> Result<(Vec<LocatedMatch>, SearchTimings)>
where
    T: Clone + Eq + Hash,
    C: TokenCodec<T>,
{
    let directory = options.index.as_path();
    let mut timings = SearchTimings::default();

    let open_start = Instant::now();
    let engine = SearchEngine::open_with(index::engine_path(directory), codec)?;
    if options.verify {
        engine.verify()?;
    }
    let mut sources = SourceReader::open(directory, metadata.sequence_count)?;
    timings.open = open_start.elapsed();

    // A snapshot is a directory, a configuration a file, so the two forms of
    // '--costs' are told apart without a second option.
    let costs_start = Instant::now();
    let costs = match &options.costs {
        Some(path) if path.is_dir() => cost_snapshot::open(path, metadata.tokenizer, codec)?,
        Some(path) => cost_config::load(path, &tokenizer)?,
        None => RuntimeCosts::levenshtein(),
    };
    if options.verify {
        costs.verify()?;
    }
    timings.costs = costs_start.elapsed();

    let search_start = Instant::now();
    let mut params = RangeSearchParams::new(options.threshold.into());
    if let Some(eta) = options.eta {
        params = params.with_eta(eta.into());
    }
    let query_sequence: Vec<_> = tokenizer
        .tokenize(&options.query_source_text)
        .into_iter()
        .map(|token| token.value)
        .collect();
    let matches = engine
        .range_searcher(costs)
        .search(&query_sequence, &params)?;
    timings.search = search_start.elapsed();

    let located = locate_matches(matches, &mut sources, &tokenizer)?;
    Ok((located, timings))
}

/// Elapsed time of each stage of building an index.
///
/// `total` is measured by the caller because it covers work outside the build
/// itself.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct IndexTimings {
    read: Duration,
    build: Duration,
    save: Duration,
    total: Duration,
}

/// Elapsed time of each stage of building a cost snapshot.
///
/// `total` is measured by the caller because it covers work outside the build
/// itself.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CostsTimings {
    read: Duration,
    save: Duration,
    total: Duration,
}

/// Elapsed time of each stage of a single search.
///
/// `total` is measured by the caller because it covers work outside the search
/// itself.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SearchTimings {
    open: Duration,
    costs: Duration,
    search: Duration,
    total: Duration,
}

fn write_index_timings(mut output: impl Write, timings: &IndexTimings) -> io::Result<()> {
    writeln!(
        output,
        "timing: read={} build={} save={} total={}",
        format_milliseconds(timings.read),
        format_milliseconds(timings.build),
        format_milliseconds(timings.save),
        format_milliseconds(timings.total),
    )
}

fn write_costs_timings(mut output: impl Write, timings: &CostsTimings) -> io::Result<()> {
    writeln!(
        output,
        "timing: read={} save={} total={}",
        format_milliseconds(timings.read),
        format_milliseconds(timings.save),
        format_milliseconds(timings.total),
    )
}

fn write_search_timings(mut output: impl Write, timings: &SearchTimings) -> io::Result<()> {
    writeln!(
        output,
        "timing: open={} costs={} search={} total={}",
        format_milliseconds(timings.open),
        format_milliseconds(timings.costs),
        format_milliseconds(timings.search),
        format_milliseconds(timings.total),
    )
}

fn format_milliseconds(elapsed: Duration) -> String {
    format!("{:.3}ms", elapsed.as_secs_f64() * 1e3)
}

#[derive(Debug, Clone, PartialEq)]
struct LocatedMatch {
    sequence_id: usize,
    byte_range: Range<usize>,
    distance: f32,
    matched_text: String,
}

/// Maps token ranges back to byte ranges of the stored source texts.
///
/// Matches are ordered by sequence ID, so caching the source text of the
/// previous match is enough to tokenize each matched source text once.
fn locate_matches<T>(
    matches: Vec<Match>,
    sources: &mut SourceReader,
    tokenizer: &impl Tokenizer<Token = T>,
) -> Result<Vec<LocatedMatch>> {
    let mut located = Vec::with_capacity(matches.len());
    let mut cached: Option<(usize, String, Vec<Range<usize>>)> = None;

    for matched in matches {
        if cached
            .as_ref()
            .is_none_or(|(sequence_id, _, _)| *sequence_id != matched.sequence_id)
        {
            let source_text = sources.read(matched.sequence_id)?;
            let ranges = tokenizer
                .tokenize(&source_text)
                .into_iter()
                .map(|token| token.byte_range)
                .collect();
            cached = Some((matched.sequence_id, source_text, ranges));
        }
        let (_, source_text, ranges) = cached.as_ref().expect("the source text was just read");

        // Matches are always non-empty token ranges, but the stored source text
        // only tokenizes to them if it still agrees with the index.
        let bounds = (
            ranges.get(matched.token_range.start),
            ranges.get(matched.token_range.end - 1),
        );
        let (Some(start), Some(end)) = bounds else {
            bail!("index does not match its stored source texts");
        };
        let byte_range = start.start..end.end;
        located.push(LocatedMatch {
            sequence_id: matched.sequence_id,
            matched_text: source_text[byte_range.clone()].to_owned(),
            byte_range,
            distance: matched.distance,
        });
    }
    Ok(located)
}

fn write_matches(output: impl Write, matches: &[LocatedMatch]) -> csv::Result<()> {
    let mut writer = WriterBuilder::new()
        .delimiter(b'\t')
        .terminator(Terminator::Any(b'\n'))
        .has_headers(false)
        .from_writer(output);

    for matched in matches {
        writer.write_record([
            matched.sequence_id.to_string(),
            matched.distance.to_string(),
            matched.byte_range.start.to_string(),
            matched.byte_range.end.to_string(),
            matched.matched_text.clone(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use clap::{CommandFactory, Parser};
    use yurine::costs::Cost;

    use super::{
        Command, CostsOptions, CostsTimings, IndexOptions, IndexTimings, LocatedMatch, Options,
        SearchOptions, SearchTimings, TokenizerKind, find_matches, run_costs, run_index,
        write_costs_timings, write_index_timings, write_matches, write_search_timings,
    };

    static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    pub(crate) struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        pub(crate) fn new() -> Self {
            let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("yurine-cli-search-{}-{id}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self { path }
        }

        pub(crate) fn path(&self) -> &Path {
            &self.path
        }

        pub(crate) fn write(&self, name: &str, contents: &str) -> PathBuf {
            let path = self.path.join(name);
            fs::write(&path, contents).unwrap();
            path
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.path).unwrap();
        }
    }

    /// Builds an index over one source text per element and returns its path.
    fn build_index(
        directory: &TestDirectory,
        tokenizer: TokenizerKind,
        source_texts: &[&str],
    ) -> PathBuf {
        let corpus = directory.write("corpus.txt", &format!("{}\n", source_texts.join("\n")));
        let index = directory.path().join("index");
        run_index(&IndexOptions {
            index: index.clone(),
            corpus: Some(corpus),
            tokenizer,
            timing: false,
        })
        .unwrap();
        index
    }

    /// Compiles a cost configuration into a snapshot and returns its path.
    fn build_costs(directory: &TestDirectory, tokenizer: TokenizerKind, costs: PathBuf) -> PathBuf {
        let snapshot = directory.path().join("costs-snapshot");
        run_costs(&CostsOptions {
            costs,
            snapshot: snapshot.clone(),
            tokenizer,
            timing: false,
        })
        .unwrap();
        snapshot
    }

    fn search_options(index: PathBuf, query_source_text: &str, threshold: Cost) -> SearchOptions {
        SearchOptions {
            index,
            query_source_text: query_source_text.to_owned(),
            threshold,
            eta: None,
            costs: None,
            verify: false,
            timing: false,
        }
    }

    #[test]
    fn command_definition_is_valid() {
        Options::command().debug_assert();
    }

    #[test]
    fn parses_index_options() {
        let options = Options::try_parse_from([
            "yurine",
            "index",
            "--tokenizer",
            "whitespace",
            "--timing",
            "index-directory",
            "corpus.txt",
        ])
        .unwrap();

        assert_eq!(
            options.command,
            Command::Index(IndexOptions {
                index: PathBuf::from("index-directory"),
                corpus: Some(PathBuf::from("corpus.txt")),
                tokenizer: TokenizerKind::Whitespace,
                timing: true,
            })
        );
    }

    #[test]
    fn parses_search_options() {
        let options = Options::try_parse_from([
            "yurine",
            "search",
            "--threshold",
            "1.5",
            "--eta",
            "0.25",
            "--costs",
            "costs.json",
            "--verify",
            "--timing",
            "index-directory",
            "hello world",
        ])
        .unwrap();

        assert_eq!(
            options.command,
            Command::Search(SearchOptions {
                index: PathBuf::from("index-directory"),
                query_source_text: "hello world".to_owned(),
                threshold: Cost::new_const(1.5),
                eta: Some(Cost::new_const(0.25)),
                costs: Some(PathBuf::from("costs.json")),
                verify: true,
                timing: true,
            })
        );
    }

    #[test]
    fn parses_costs_options() {
        let options = Options::try_parse_from([
            "yurine",
            "costs",
            "--tokenizer",
            "whitespace",
            "--timing",
            "costs.json",
            "costs-snapshot",
        ])
        .unwrap();

        assert_eq!(
            options.command,
            Command::Costs(CostsOptions {
                costs: PathBuf::from("costs.json"),
                snapshot: PathBuf::from("costs-snapshot"),
                tokenizer: TokenizerKind::Whitespace,
                timing: true,
            })
        );
    }

    #[test]
    fn optional_flags_default_to_disabled() {
        let options = Options::try_parse_from(["yurine", "search", "index-directory", "query"])
            .unwrap()
            .command;
        assert_eq!(
            options,
            Command::Search(SearchOptions {
                index: PathBuf::from("index-directory"),
                query_source_text: "query".to_owned(),
                threshold: Cost::ZERO,
                eta: None,
                costs: None,
                verify: false,
                timing: false,
            })
        );
    }

    #[test]
    fn clap_rejects_invalid_arguments() {
        assert!(Options::try_parse_from(["yurine"]).is_err());
        assert!(Options::try_parse_from(["yurine", "query"]).is_err());
        assert!(Options::try_parse_from(["yurine", "index"]).is_err());
        assert!(Options::try_parse_from(["yurine", "costs"]).is_err());
        assert!(Options::try_parse_from(["yurine", "costs", "costs.json"]).is_err());
        assert!(Options::try_parse_from(["yurine", "search", "index-directory"]).is_err());
        assert!(
            Options::try_parse_from(["yurine", "search", "--tokenizer", "character", "i", "q"])
                .is_err()
        );
        assert!(
            Options::try_parse_from(["yurine", "index", "--tokenizer", "bytes", "index-directory"])
                .is_err()
        );
        assert!(
            Options::try_parse_from(["yurine", "search", "--threshold", "-1", "i", "q"]).is_err()
        );
        assert!(Options::try_parse_from(["yurine", "search", "i", "first", "second"]).is_err());
    }

    #[test]
    fn threshold_rejects_maximum_cost_but_eta_accepts_it() {
        let maximum = Cost::MAX.to_string();

        let error =
            Options::try_parse_from(["yurine", "search", "--threshold", &maximum, "i", "q"])
                .unwrap_err();
        assert!(error.to_string().contains("must be less than f32::MAX"));

        let options =
            Options::try_parse_from(["yurine", "search", "--eta", &maximum, "i", "q"]).unwrap();
        assert_eq!(
            options.command,
            Command::Search(SearchOptions {
                index: PathBuf::from("i"),
                query_source_text: "q".to_owned(),
                threshold: Cost::ZERO,
                eta: Some(Cost::MAX),
                costs: None,
                verify: false,
                timing: false,
            })
        );
    }

    #[test]
    fn searches_with_character_tokenization() {
        let directory = TestDirectory::new();
        let index = build_index(&directory, TokenizerKind::Character, &["東京都", "京都市"]);

        let (matches, _) = find_matches(&search_options(index, "東京", Cost::ZERO)).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].sequence_id, 0);
        assert_eq!(matches[0].byte_range, 0..6);
        assert_eq!(matches[0].matched_text, "東京");
    }

    #[test]
    fn searches_with_whitespace_tokenization() {
        let directory = TestDirectory::new();
        let index = build_index(
            &directory,
            TokenizerKind::Whitespace,
            &["new york city", "york new"],
        );

        let (matches, _) = find_matches(&search_options(index, "new york", Cost::ZERO)).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].byte_range, 0..8);
        assert_eq!(matches[0].matched_text, "new york");
    }

    #[test]
    fn an_index_serves_repeated_searches() {
        let directory = TestDirectory::new();
        let index = build_index(&directory, TokenizerKind::Character, &["東京都", "京都市"]);

        let (first, _) = find_matches(&search_options(index.clone(), "東京", Cost::ZERO)).unwrap();
        let (second, _) = find_matches(&search_options(index.clone(), "京都", Cost::ZERO)).unwrap();
        let (verified, _) = find_matches(&SearchOptions {
            verify: true,
            ..search_options(index, "京都", Cost::ZERO)
        })
        .unwrap();

        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 2);
        assert_eq!(verified, second);
    }

    #[test]
    fn a_failed_rebuild_leaves_the_previous_index_usable() {
        let directory = TestDirectory::new();
        let index = build_index(&directory, TokenizerKind::Character, &["東京都", "京都市"]);
        // The corpus stops being valid UTF-8 after the first line.
        let corpus = directory.path().join("broken.txt");
        fs::write(&corpus, b"\xe6\x9d\xb1\n\xff\n").unwrap();

        let error = run_index(&IndexOptions {
            index: index.clone(),
            corpus: Some(corpus),
            tokenizer: TokenizerKind::Character,
            timing: false,
        })
        .unwrap_err();

        assert!(error.to_string().contains("failed to build the index"));
        let (matches, _) = find_matches(&search_options(index, "東京", Cost::ZERO)).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_text, "東京");
    }

    #[test]
    fn source_texts_keep_their_original_spacing() {
        let directory = TestDirectory::new();
        let index = build_index(
            &directory,
            TokenizerKind::Whitespace,
            &["  new   york  city"],
        );

        let (matches, _) = find_matches(&search_options(index, "york city", Cost::ZERO)).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].byte_range, 8..18);
        assert_eq!(matches[0].matched_text, "york  city");
    }

    #[test]
    fn searching_a_missing_index_fails() {
        let directory = TestDirectory::new();

        let error = find_matches(&search_options(
            directory.path().join("absent"),
            "query",
            Cost::ZERO,
        ))
        .unwrap_err();

        assert!(error.to_string().contains("failed to read index metadata"));
    }

    #[test]
    fn searches_with_character_embedding_costs() {
        let directory = TestDirectory::new();
        directory.write(
            "embeddings.jsonl",
            concat!(
                "{\"token\":\"x\",\"embedding\":[1.0,0.0]}\n",
                "{\"token\":\"あ\",\"embedding\":[0.8,0.6]}\n"
            ),
        );
        let costs = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "embedding",
                "embeddings": {"path": "embeddings.jsonl", "format": "jsonl"}
            }"#,
        );
        let index = build_index(&directory, TokenizerKind::Character, &["あ"]);

        let (matches, _) = find_matches(&SearchOptions {
            eta: Some(Cost::new_const(0.25)),
            costs: Some(costs),
            ..search_options(index, "x", Cost::new_const(0.25))
        })
        .unwrap();

        assert_eq!(matches.len(), 1);
        assert!((matches[0].distance - 0.2).abs() < 1e-6);
    }

    #[test]
    fn searches_with_whitespace_embedding_costs() {
        let directory = TestDirectory::new();
        directory.write(
            "embeddings.jsonl",
            concat!(
                "{\"token\":\"colour\",\"embedding\":[1.0,0.0]}\n",
                "{\"token\":\"color\",\"embedding\":[0.8,0.6]}\n"
            ),
        );
        let costs = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "embedding",
                "embeddings": {"path": "embeddings.jsonl", "format": "jsonl"}
            }"#,
        );
        let index = build_index(&directory, TokenizerKind::Whitespace, &["color palette"]);

        let (matches, _) = find_matches(&SearchOptions {
            eta: Some(Cost::new_const(0.25)),
            costs: Some(costs),
            ..search_options(index, "colour", Cost::new_const(0.25))
        })
        .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].byte_range, 0..5);
        assert!((matches[0].distance - 0.2).abs() < 1e-6);
    }

    #[test]
    fn searches_with_character_custom_costs() {
        let directory = TestDirectory::new();
        directory.write(
            "rules.jsonl",
            "{\"operation\":\"substitution\",\"from\":\"x\",\"to\":\"a\",\"cost\":0.25}\n",
        );
        let costs = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "custom",
                "rules": {"path": "rules.jsonl", "format": "jsonl"}
            }"#,
        );
        let index = build_index(&directory, TokenizerKind::Character, &["a"]);

        let (matches, _) = find_matches(&SearchOptions {
            eta: Some(Cost::new_const(0.25)),
            costs: Some(costs),
            ..search_options(index, "x", Cost::new_const(0.25))
        })
        .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].byte_range, 0..1);
        assert_eq!(matches[0].distance, 0.25);
    }

    #[test]
    fn searches_with_whitespace_custom_costs() {
        let directory = TestDirectory::new();
        directory.write(
            "rules.jsonl",
            "{\"operation\":\"substitution\",\"from\":\"colour\",\"to\":\"color\",\"cost\":0.25}\n",
        );
        let costs = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "custom",
                "rules": {"path": "rules.jsonl", "format": "jsonl"}
            }"#,
        );
        let index = build_index(&directory, TokenizerKind::Whitespace, &["color palette"]);

        let (matches, _) = find_matches(&SearchOptions {
            eta: Some(Cost::new_const(0.25)),
            costs: Some(costs),
            ..search_options(index, "colour", Cost::new_const(0.25))
        })
        .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].byte_range, 0..5);
        assert_eq!(matches[0].distance, 0.25);
    }

    #[test]
    fn a_cost_snapshot_searches_like_its_configuration() {
        let directory = TestDirectory::new();
        directory.write(
            "embeddings.jsonl",
            concat!(
                "{\"token\":\"x\",\"embedding\":[1.0,0.0]}\n",
                "{\"token\":\"あ\",\"embedding\":[0.8,0.6]}\n"
            ),
        );
        let costs = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "embedding",
                "embeddings": {"path": "embeddings.jsonl", "format": "jsonl"}
            }"#,
        );
        let index = build_index(&directory, TokenizerKind::Character, &["あ"]);
        let snapshot = build_costs(&directory, TokenizerKind::Character, costs.clone());

        let (configured, _) = find_matches(&SearchOptions {
            eta: Some(Cost::new_const(0.25)),
            costs: Some(costs),
            ..search_options(index.clone(), "x", Cost::new_const(0.25))
        })
        .unwrap();
        let (compiled, _) = find_matches(&SearchOptions {
            eta: Some(Cost::new_const(0.25)),
            costs: Some(snapshot),
            verify: true,
            ..search_options(index, "x", Cost::new_const(0.25))
        })
        .unwrap();

        assert_eq!(configured.len(), 1);
        assert_eq!(compiled, configured);
    }

    #[test]
    fn a_custom_cost_snapshot_searches_like_its_configuration() {
        let directory = TestDirectory::new();
        directory.write(
            "rules.jsonl",
            "{\"operation\":\"substitution\",\"from\":\"colour\",\"to\":\"color\",\"cost\":0.25}\n",
        );
        let costs = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "custom",
                "rules": {"path": "rules.jsonl", "format": "jsonl"}
            }"#,
        );
        let index = build_index(&directory, TokenizerKind::Whitespace, &["color palette"]);
        let snapshot = build_costs(&directory, TokenizerKind::Whitespace, costs);

        let (matches, _) = find_matches(&SearchOptions {
            eta: Some(Cost::new_const(0.25)),
            costs: Some(snapshot),
            ..search_options(index, "colour", Cost::new_const(0.25))
        })
        .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].byte_range, 0..5);
        assert_eq!(matches[0].distance, 0.25);
    }

    #[test]
    fn searching_rejects_a_snapshot_tokenized_unlike_the_index() {
        let directory = TestDirectory::new();
        directory.write(
            "rules.jsonl",
            "{\"operation\":\"deletion\",\"token\":\"a\",\"cost\":0.25}\n",
        );
        let costs = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "custom",
                "rules": {"path": "rules.jsonl", "format": "jsonl"}
            }"#,
        );
        let index = build_index(&directory, TokenizerKind::Whitespace, &["a b"]);
        let snapshot = build_costs(&directory, TokenizerKind::Character, costs);

        let error = find_matches(&SearchOptions {
            costs: Some(snapshot),
            ..search_options(index, "a", Cost::ZERO)
        })
        .unwrap_err();

        let message = format!("{error:#}");
        assert!(message.contains("uses the character tokenizer"));
        assert!(message.contains("the index uses whitespace"));
    }

    #[test]
    fn building_a_snapshot_reports_an_invalid_configuration() {
        let directory = TestDirectory::new();
        let costs = directory.write("costs.json", "not json");

        let error = run_costs(&CostsOptions {
            costs,
            snapshot: directory.path().join("costs-snapshot"),
            tokenizer: TokenizerKind::Character,
            timing: false,
        })
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("failed to build the cost snapshot")
        );
    }

    #[test]
    fn csv_writer_quotes_tabs_in_output_fields() {
        let matches = vec![LocatedMatch {
            sequence_id: 0,
            byte_range: 0..3,
            distance: 0.0,
            matched_text: "a\tb".to_owned(),
        }];
        let mut output = Vec::new();

        write_matches(&mut output, &matches).unwrap();

        assert_eq!(String::from_utf8(output).unwrap(), "0\t0\t0\t3\t\"a\tb\"\n");
    }

    #[test]
    fn index_timings_are_reported_in_milliseconds() {
        let timings = IndexTimings {
            read: Duration::from_micros(12_345),
            build: Duration::from_millis(234),
            save: Duration::from_micros(1_234),
            total: Duration::from_secs(2),
        };
        let mut output = Vec::new();

        write_index_timings(&mut output, &timings).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "timing: read=12.345ms build=234.000ms save=1.234ms total=2000.000ms\n"
        );
    }

    #[test]
    fn costs_timings_are_reported_in_milliseconds() {
        let timings = CostsTimings {
            read: Duration::from_micros(12_345),
            save: Duration::from_millis(234),
            total: Duration::from_secs(2),
        };
        let mut output = Vec::new();

        write_costs_timings(&mut output, &timings).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "timing: read=12.345ms save=234.000ms total=2000.000ms\n"
        );
    }

    #[test]
    fn search_timings_are_reported_in_milliseconds() {
        let timings = SearchTimings {
            open: Duration::from_micros(12_345),
            costs: Duration::ZERO,
            search: Duration::from_micros(1_234),
            total: Duration::from_secs(2),
        };
        let mut output = Vec::new();

        write_search_timings(&mut output, &timings).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "timing: open=12.345ms costs=0.000ms search=1.234ms total=2000.000ms\n"
        );
    }
}
