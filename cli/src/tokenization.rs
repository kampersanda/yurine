use std::fmt;
use std::ops::Range;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Tokenization strategy of a corpus and of the queries searching it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TokenizerKind {
    #[default]
    Character,
    Whitespace,
}

impl fmt::Display for TokenizerKind {
    /// Writes the name the command line and the metadata files use.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Character => "character",
            Self::Whitespace => "whitespace",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Tokenized<T> {
    pub(crate) value: T,
    pub(crate) byte_range: Range<usize>,
}

impl<T> Tokenized<T> {
    fn new(value: T, byte_range: Range<usize>) -> Self {
        Self { value, byte_range }
    }
}

pub(crate) trait Tokenizer {
    type Token;

    fn tokenize(&self, source_text: &str) -> Vec<Tokenized<Self::Token>>;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CharacterTokenizer;

impl Tokenizer for CharacterTokenizer {
    type Token = char;

    fn tokenize(&self, source_text: &str) -> Vec<Tokenized<Self::Token>> {
        source_text
            .char_indices()
            .map(|(byte_start, token)| {
                let byte_end = byte_start + token.len_utf8();
                Tokenized::new(token, byte_start..byte_end)
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct WhitespaceTokenizer;

impl Tokenizer for WhitespaceTokenizer {
    type Token = String;

    fn tokenize(&self, source_text: &str) -> Vec<Tokenized<Self::Token>> {
        let mut tokens = Vec::new();
        let mut token_byte_start = None;

        for (byte_index, character) in source_text.char_indices() {
            if character.is_whitespace() {
                if let Some(byte_start) = token_byte_start.take() {
                    tokens.push(Tokenized::new(
                        source_text[byte_start..byte_index].to_owned(),
                        byte_start..byte_index,
                    ));
                }
            } else if token_byte_start.is_none() {
                token_byte_start = Some(byte_index);
            }
        }

        if let Some(byte_start) = token_byte_start {
            tokens.push(Tokenized::new(
                source_text[byte_start..].to_owned(),
                byte_start..source_text.len(),
            ));
        }

        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::{CharacterTokenizer, Tokenized, Tokenizer, WhitespaceTokenizer};

    #[test]
    fn character_tokenizer_records_utf8_byte_ranges() {
        assert_eq!(
            CharacterTokenizer.tokenize("aあ🦀"),
            [
                Tokenized::new('a', 0..1),
                Tokenized::new('あ', 1..4),
                Tokenized::new('🦀', 4..8),
            ]
        );
    }

    #[test]
    fn character_tokenizer_handles_empty_input() {
        assert!(CharacterTokenizer.tokenize("").is_empty());
    }

    #[test]
    fn whitespace_tokenizer_records_utf8_byte_ranges() {
        assert_eq!(
            WhitespaceTokenizer.tokenize("東京\u{3000}京都 大阪"),
            [
                Tokenized::new("東京".to_owned(), 0..6),
                Tokenized::new("京都".to_owned(), 9..15),
                Tokenized::new("大阪".to_owned(), 16..22),
            ]
        );
    }

    #[test]
    fn whitespace_tokenizer_splits_on_consecutive_whitespace() {
        assert_eq!(
            WhitespaceTokenizer.tokenize("  one\ttwo\nthree  "),
            [
                Tokenized::new("one".to_owned(), 2..5),
                Tokenized::new("two".to_owned(), 6..9),
                Tokenized::new("three".to_owned(), 10..15),
            ]
        );
    }

    #[test]
    fn whitespace_tokenizer_handles_input_without_tokens() {
        assert!(WhitespaceTokenizer.tokenize("").is_empty());
        assert!(WhitespaceTokenizer.tokenize(" \t\n").is_empty());
    }
}
