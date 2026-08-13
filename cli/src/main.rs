use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufRead, BufReader, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use csv::{Terminator, WriterBuilder};
use yurine::costs::Cost;
use yurine::search::range_search::RangeSearchParams;
use yurine::search::{Match, SearchEngineBuilder};
use yurine::types::StringId;

mod cost_config;
mod tokenization;

use cost_config::RuntimeCosts;
use tokenization::{CharacterTokenizer, Tokenizer, WhitespaceTokenizer};

/// Search newline-delimited source texts with edit distance using Yurine.
#[derive(Debug, Parser, PartialEq)]
#[command(version)]
struct Options {
    /// Query source text.
    #[arg(value_name = "QUERY")]
    query_source_text: String,

    /// File containing one source text per line; reads standard input if omitted or '-'.
    corpus: Option<PathBuf>,

    /// Maximum edit distance.
    #[arg(short, long, default_value = "0", value_parser = parse_threshold)]
    threshold: Cost,

    /// Substitution-neighborhood radius; calculated automatically if omitted.
    #[arg(long, value_parser = parse_cost)]
    eta: Option<Cost>,

    /// Tokenization strategy.
    #[arg(long, value_enum, default_value_t = TokenizerKind::Character)]
    tokenizer: TokenizerKind,

    /// JSON file describing the edit-cost policy.
    #[arg(long)]
    costs: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
enum TokenizerKind {
    #[default]
    Character,
    Whitespace,
}

fn main() -> ExitCode {
    match run(Options::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(options: Options) -> Result<()> {
    let source_texts = read_source_texts(options.corpus.as_deref())?;
    let matches = match options.tokenizer {
        TokenizerKind::Character => {
            search_source_texts(&source_texts, &options, CharacterTokenizer)
        }
        TokenizerKind::Whitespace => {
            search_source_texts(&source_texts, &options, WhitespaceTokenizer)
        }
    }
    .context("search failed")?;

    write_matches(io::stdout().lock(), &source_texts, &matches).context("failed to write results")
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

fn read_source_texts(path: Option<&Path>) -> Result<Vec<String>> {
    match path {
        Some(path) if path != Path::new("-") => {
            let file = File::open(path)
                .with_context(|| format!("failed to open corpus file '{}'", path.display()))?;
            read_lines(BufReader::new(file))
                .with_context(|| format!("failed to read corpus file '{}'", path.display()))
        }
        _ => read_lines(io::stdin().lock()).context("failed to read corpus from standard input"),
    }
}

fn read_lines(reader: impl BufRead) -> io::Result<Vec<String>> {
    reader.lines().collect()
}

fn search_source_texts<T>(
    source_texts: &[String],
    options: &Options,
    tokenizer: impl Tokenizer<Token = T>,
) -> Result<Vec<LocatedMatch>>
where
    T: Clone + Eq + Hash,
{
    let costs = match &options.costs {
        Some(path) => cost_config::load(path, &tokenizer)?,
        None => RuntimeCosts::levenshtein(),
    };
    let mut builder = SearchEngineBuilder::new(costs);
    let mut source_token_ranges = Vec::with_capacity(source_texts.len());
    for source_text in source_texts {
        let (sequence, ranges): (Vec<_>, Vec<_>) = tokenizer
            .tokenize(source_text)
            .into_iter()
            .map(|token| (token.value, token.byte_range))
            .unzip();
        builder.add_sequence(sequence)?;
        source_token_ranges.push(ranges);
    }
    let engine = builder.build()?;
    let mut params = RangeSearchParams::new(options.threshold);
    if let Some(eta) = options.eta {
        params = params.with_eta(eta);
    }
    let query_sequence: Vec<_> = tokenizer
        .tokenize(&options.query_source_text)
        .into_iter()
        .map(|token| token.value)
        .collect();
    let matches = engine.range_search(&query_sequence, &params)?;
    Ok(matches
        .into_iter()
        .map(|matched| locate_match(matched, &source_token_ranges))
        .collect())
}

#[derive(Debug, Clone, PartialEq)]
struct LocatedMatch {
    string_id: StringId,
    byte_range: Range<usize>,
    distance: Cost,
}

fn locate_match(matched: Match, source_token_ranges: &[Vec<Range<usize>>]) -> LocatedMatch {
    // Every source text is passed to `add_sequence` in order, so `StringId`
    // indexes `source_token_ranges`. Matches are always non-empty token ranges.
    let ranges = &source_token_ranges[matched.sequence_id.as_usize()];
    let token_start = matched.token_range.start.as_usize();
    let token_end = matched.token_range.end.as_usize();
    LocatedMatch {
        string_id: matched.sequence_id,
        byte_range: ranges[token_start].start..ranges[token_end - 1].end,
        distance: matched.distance,
    }
}

fn write_matches(
    output: impl Write,
    source_texts: &[String],
    matches: &[LocatedMatch],
) -> csv::Result<()> {
    let mut writer = WriterBuilder::new()
        .delimiter(b'\t')
        .terminator(Terminator::Any(b'\n'))
        .has_headers(false)
        .from_writer(output);

    for matched in matches {
        let source_text = &source_texts[matched.string_id.as_usize()];
        let matched_text = &source_text[matched.byte_range.clone()];
        writer.write_record([
            matched.string_id.to_string(),
            matched.distance.to_string(),
            matched.byte_range.start.to_string(),
            matched.byte_range.end.to_string(),
            matched_text.to_owned(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use clap::{CommandFactory, Parser};
    use yurine::costs::Cost;
    use yurine::types::StringId;

    use super::{
        LocatedMatch, Options, TokenizerKind, read_lines, search_source_texts, write_matches,
    };
    use crate::tokenization::{CharacterTokenizer, WhitespaceTokenizer};

    static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new() -> Self {
            let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("yurine-cli-search-{}-{id}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self { path }
        }

        fn write(&self, name: &str, contents: &str) -> PathBuf {
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

    #[test]
    fn command_definition_is_valid() {
        Options::command().debug_assert();
    }

    #[test]
    fn parses_search_options() {
        let options = Options::try_parse_from([
            "yurine",
            "--threshold",
            "1.5",
            "--eta",
            "0.25",
            "--tokenizer",
            "whitespace",
            "--costs",
            "costs.json",
            "hello world",
            "corpus.txt",
        ])
        .unwrap();

        assert_eq!(
            options,
            Options {
                query_source_text: "hello world".to_owned(),
                corpus: Some(PathBuf::from("corpus.txt")),
                threshold: Cost::new_const(1.5),
                eta: Some(Cost::new_const(0.25)),
                tokenizer: TokenizerKind::Whitespace,
                costs: Some(PathBuf::from("costs.json")),
            }
        );
    }

    #[test]
    fn clap_rejects_invalid_arguments() {
        assert!(Options::try_parse_from(["yurine"]).is_err());
        assert!(Options::try_parse_from(["yurine", "--threshold", "-1", "query"]).is_err());
        assert!(Options::try_parse_from(["yurine", "--tokenizer", "bytes", "query"]).is_err());
        assert!(Options::try_parse_from(["yurine", "query", "first", "second"]).is_err());
    }

    #[test]
    fn threshold_rejects_maximum_cost_but_eta_accepts_it() {
        let maximum = Cost::MAX.to_string();

        let error =
            Options::try_parse_from(["yurine", "--threshold", &maximum, "query"]).unwrap_err();
        assert!(error.to_string().contains("must be less than f32::MAX"));

        let options = Options::try_parse_from(["yurine", "--eta", &maximum, "query"]).unwrap();
        assert_eq!(options.eta, Some(Cost::MAX));
    }

    #[test]
    fn reads_one_source_text_per_line() {
        let source_texts = read_lines(Cursor::new("東京\r\n京都\n\n")).unwrap();
        assert_eq!(source_texts, ["東京", "京都", ""]);
    }

    #[test]
    fn searches_with_character_tokenization() {
        let source_texts = vec!["東京都".to_owned(), "京都市".to_owned()];
        let options = Options {
            query_source_text: "東京".to_owned(),
            corpus: None,
            threshold: Cost::ZERO,
            eta: None,
            tokenizer: TokenizerKind::Character,
            costs: None,
        };

        let matches = search_source_texts(&source_texts, &options, CharacterTokenizer).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].string_id.get(), 0);
        assert_eq!(matches[0].byte_range, 0..6);
    }

    #[test]
    fn searches_with_whitespace_tokenization() {
        let source_texts = vec!["new york city".to_owned(), "york new".to_owned()];
        let options = Options {
            query_source_text: "new york".to_owned(),
            corpus: None,
            threshold: Cost::ZERO,
            eta: None,
            tokenizer: TokenizerKind::Whitespace,
            costs: None,
        };

        let matches = search_source_texts(&source_texts, &options, WhitespaceTokenizer).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].byte_range, 0..8);
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
        let config = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "embedding",
                "embeddings": {"path": "embeddings.jsonl", "format": "jsonl"}
            }"#,
        );
        let options = Options {
            query_source_text: "x".to_owned(),
            corpus: None,
            threshold: Cost::new_const(0.25),
            eta: Some(Cost::new_const(0.25)),
            tokenizer: TokenizerKind::Character,
            costs: Some(config),
        };

        let matches =
            search_source_texts(&["あ".to_owned()], &options, CharacterTokenizer).unwrap();

        assert_eq!(matches.len(), 1);
        assert!((matches[0].distance.get() - 0.2).abs() < 1e-6);
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
        let config = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "embedding",
                "embeddings": {"path": "embeddings.jsonl", "format": "jsonl"}
            }"#,
        );
        let options = Options {
            query_source_text: "colour".to_owned(),
            corpus: None,
            threshold: Cost::new_const(0.25),
            eta: Some(Cost::new_const(0.25)),
            tokenizer: TokenizerKind::Whitespace,
            costs: Some(config),
        };

        let matches =
            search_source_texts(&["color palette".to_owned()], &options, WhitespaceTokenizer)
                .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].byte_range, 0..5);
        assert!((matches[0].distance.get() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn searches_with_character_custom_costs() {
        let directory = TestDirectory::new();
        directory.write(
            "rules.jsonl",
            "{\"operation\":\"substitution\",\"from\":\"x\",\"to\":\"a\",\"cost\":0.25}\n",
        );
        let config = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "custom",
                "rules": {"path": "rules.jsonl", "format": "jsonl"}
            }"#,
        );
        let options = Options {
            query_source_text: "x".to_owned(),
            corpus: None,
            threshold: Cost::new_const(0.25),
            eta: Some(Cost::new_const(0.25)),
            tokenizer: TokenizerKind::Character,
            costs: Some(config),
        };

        let matches = search_source_texts(&["a".to_owned()], &options, CharacterTokenizer).unwrap();

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
        let config = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "custom",
                "rules": {"path": "rules.jsonl", "format": "jsonl"}
            }"#,
        );
        let options = Options {
            query_source_text: "colour".to_owned(),
            corpus: None,
            threshold: Cost::new_const(0.25),
            eta: Some(Cost::new_const(0.25)),
            tokenizer: TokenizerKind::Whitespace,
            costs: Some(config),
        };

        let matches =
            search_source_texts(&["color palette".to_owned()], &options, WhitespaceTokenizer)
                .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].byte_range, 0..5);
        assert_eq!(matches[0].distance, 0.25);
    }

    #[test]
    fn csv_writer_quotes_tabs_in_output_fields() {
        let source_texts = vec!["a\tb".to_owned()];
        let matches = vec![LocatedMatch {
            string_id: StringId::new(0),
            byte_range: 0..3,
            distance: Cost::ZERO,
        }];
        let mut output = Vec::new();

        write_matches(&mut output, &source_texts, &matches).unwrap();

        assert_eq!(String::from_utf8(output).unwrap(), "0\t0\t0\t3\t\"a\tb\"\n");
    }
}
