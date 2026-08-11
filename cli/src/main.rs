use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use csv::{Terminator, WriterBuilder};
use yurine::costs::Cost;
use yurine::costs::levenshtein::LevenshteinCosts;
use yurine::search::range_search::RangeSearchParams;
use yurine::search::{Match, SearchEngineBuilder};
use yurine::tokenization::Tokenizer;
use yurine::tokenization::character::CharacterTokenizer;
use yurine::tokenization::whitespace::WhitespaceTokenizer;

/// Search newline-delimited strings with edit distance using Yurine.
#[derive(Debug, Parser, PartialEq)]
#[command(version)]
struct Options {
    /// Query string.
    query: String,

    /// File containing one string per line; reads standard input if omitted or '-'.
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
    let corpus = read_corpus(options.corpus.as_deref())?;
    let matches = match options.tokenizer {
        TokenizerKind::Character => search(&corpus, &options, CharacterTokenizer::new()),
        TokenizerKind::Whitespace => search(&corpus, &options, WhitespaceTokenizer::new()),
    }
    .context("search failed")?;

    write_matches(io::stdout().lock(), &corpus, &matches).context("failed to write results")
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

fn read_corpus(path: Option<&Path>) -> Result<Vec<String>> {
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

fn search<T>(
    corpus: &[String],
    options: &Options,
    tokenizer: T,
) -> yurine::errors::Result<Vec<Match>>
where
    T: Tokenizer,
    T::Token: Clone + Eq + Hash,
{
    let mut builder = SearchEngineBuilder::new(tokenizer, LevenshteinCosts::new());
    for string in corpus {
        builder.add_string(string)?;
    }
    let engine = builder.build()?;
    let mut params = RangeSearchParams::new(options.threshold);
    if let Some(eta) = options.eta {
        params = params.with_eta(eta);
    }
    engine.range_search(&options.query, &params)
}

fn write_matches(output: impl Write, corpus: &[String], matches: &[Match]) -> csv::Result<()> {
    let mut writer = WriterBuilder::new()
        .delimiter(b'\t')
        .terminator(Terminator::Any(b'\n'))
        .has_headers(false)
        .from_writer(output);

    for matched in matches {
        let source = &corpus[matched.string_id.as_usize()];
        let text = &source[matched.byte_range.clone()];
        writer.write_record([
            matched.string_id.to_string(),
            matched.distance.to_string(),
            matched.byte_range.start.to_string(),
            matched.byte_range.end.to_string(),
            text.to_owned(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::path::PathBuf;

    use clap::{CommandFactory, Parser};
    use yurine::costs::Cost;
    use yurine::search::Match;
    use yurine::tokenization::character::CharacterTokenizer;
    use yurine::tokenization::whitespace::WhitespaceTokenizer;
    use yurine::types::{Position, StringId};

    use super::{Options, TokenizerKind, read_lines, search, write_matches};

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
            "hello world",
            "corpus.txt",
        ])
        .unwrap();

        assert_eq!(
            options,
            Options {
                query: "hello world".to_owned(),
                corpus: Some(PathBuf::from("corpus.txt")),
                threshold: Cost::new_const(1.5),
                eta: Some(Cost::new_const(0.25)),
                tokenizer: TokenizerKind::Whitespace,
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
    fn reads_one_string_per_line() {
        let corpus = read_lines(Cursor::new("東京\r\n京都\n\n")).unwrap();
        assert_eq!(corpus, ["東京", "京都", ""]);
    }

    #[test]
    fn searches_with_character_tokenization() {
        let corpus = vec!["東京都".to_owned(), "京都市".to_owned()];
        let options = Options {
            query: "東京".to_owned(),
            corpus: None,
            threshold: Cost::ZERO,
            eta: None,
            tokenizer: TokenizerKind::Character,
        };

        let matches = search(&corpus, &options, CharacterTokenizer::new()).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].string_id.get(), 0);
        assert_eq!(matches[0].byte_range, 0..6);
    }

    #[test]
    fn searches_with_whitespace_tokenization() {
        let corpus = vec!["new york city".to_owned(), "york new".to_owned()];
        let options = Options {
            query: "new york".to_owned(),
            corpus: None,
            threshold: Cost::ZERO,
            eta: None,
            tokenizer: TokenizerKind::Whitespace,
        };

        let matches = search(&corpus, &options, WhitespaceTokenizer::new()).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].byte_range, 0..8);
    }

    #[test]
    fn csv_writer_quotes_tabs_in_output_fields() {
        let corpus = vec!["a\tb".to_owned()];
        let matches = vec![Match {
            string_id: StringId::new(0),
            token_range: Position::new(0)..Position::new(3),
            byte_range: 0..3,
            distance: Cost::ZERO,
        }];
        let mut output = Vec::new();

        write_matches(&mut output, &corpus, &matches).unwrap();

        assert_eq!(String::from_utf8(output).unwrap(), "0\t0\t0\t3\t\"a\tb\"\n");
    }
}
