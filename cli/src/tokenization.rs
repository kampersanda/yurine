use std::ops::Range;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Tokenized {
    pub(crate) value: String,
    pub(crate) byte_range: Range<usize>,
}

impl Tokenized {
    fn new(value: String, byte_range: Range<usize>) -> Self {
        Self { value, byte_range }
    }
}

pub(crate) trait Tokenizer {
    fn tokenize(&self, source_text: &str) -> Vec<Tokenized>;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CharacterTokenizer;

impl Tokenizer for CharacterTokenizer {
    fn tokenize(&self, source_text: &str) -> Vec<Tokenized> {
        source_text
            .char_indices()
            .map(|(start, token)| {
                let end = start + token.len_utf8();
                Tokenized::new(token.to_string(), start..end)
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct WhitespaceTokenizer;

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, source_text: &str) -> Vec<Tokenized> {
        let mut tokens = Vec::new();
        let mut token_start = None;

        for (index, character) in source_text.char_indices() {
            if character.is_whitespace() {
                if let Some(start) = token_start.take() {
                    tokens.push(Tokenized::new(
                        source_text[start..index].to_owned(),
                        start..index,
                    ));
                }
            } else if token_start.is_none() {
                token_start = Some(index);
            }
        }

        if let Some(start) = token_start {
            tokens.push(Tokenized::new(
                source_text[start..].to_owned(),
                start..source_text.len(),
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
                Tokenized::new("a".to_owned(), 0..1),
                Tokenized::new("あ".to_owned(), 1..4),
                Tokenized::new("🦀".to_owned(), 4..8),
            ]
        );
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
}
