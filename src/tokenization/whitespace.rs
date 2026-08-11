//! Whitespace-delimited tokenization.

use super::{Tokenized, Tokenizer};

/// Splits a string at Unicode whitespace.
#[derive(Debug, Clone, Copy, Default)]
pub struct WhitespaceTokenizer;

impl WhitespaceTokenizer {
    /// Creates a whitespace tokenizer.
    pub const fn new() -> Self {
        Self
    }
}

impl Tokenizer for WhitespaceTokenizer {
    type Token = String;

    fn tokenize(&self, input: &str) -> Vec<Tokenized<Self::Token>> {
        let mut tokens = Vec::new();
        let mut token_start = None;

        for (index, character) in input.char_indices() {
            if character.is_whitespace() {
                if let Some(start) = token_start.take() {
                    tokens.push(Tokenized::new(input[start..index].to_owned(), start..index));
                }
            } else if token_start.is_none() {
                token_start = Some(index);
            }
        }

        if let Some(start) = token_start {
            tokens.push(Tokenized::new(
                input[start..].to_owned(),
                start..input.len(),
            ));
        }

        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::WhitespaceTokenizer;
    use crate::tokenization::{Tokenized, Tokenizer};

    #[test]
    fn splits_on_consecutive_whitespace() {
        let tokenizer = WhitespaceTokenizer::new();

        assert_eq!(
            tokenizer.tokenize("  one\ttwo\nthree  "),
            [
                Tokenized::new("one".to_owned(), 2..5),
                Tokenized::new("two".to_owned(), 6..9),
                Tokenized::new("three".to_owned(), 10..15),
            ]
        );
    }

    #[test]
    fn records_utf8_byte_ranges() {
        let tokenizer = WhitespaceTokenizer::new();

        assert_eq!(
            tokenizer.tokenize("東京\u{3000}京都 大阪"),
            [
                Tokenized::new("東京".to_owned(), 0..6),
                Tokenized::new("京都".to_owned(), 9..15),
                Tokenized::new("大阪".to_owned(), 16..22),
            ]
        );
    }

    #[test]
    fn handles_input_without_tokens() {
        let tokenizer = WhitespaceTokenizer::new();

        assert!(tokenizer.tokenize("").is_empty());
        assert!(tokenizer.tokenize(" \t\n").is_empty());
    }
}
