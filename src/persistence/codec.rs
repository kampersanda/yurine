use crate::errors::{Error, Result};

/// Encodes application tokens for a persisted index.
///
/// The identifier and version are recorded in the file and checked when it is
/// opened. Each call encodes exactly one token; framing is handled by the file
/// format.
pub trait TokenCodec<T> {
    /// Returns the stable identifier written to persisted files.
    fn id(&self) -> &str;

    /// Returns the codec format version.
    fn version(&self) -> u32 {
        1
    }

    /// Appends the encoded token to `output`.
    fn encode(&self, token: &T, output: &mut Vec<u8>) -> Result<()>;

    /// Decodes one token from its encoded bytes.
    fn decode(&self, bytes: &[u8]) -> Result<T>;
}

/// The standard UTF-8 codec for [`String`] tokens.
#[derive(Debug, Clone, Copy, Default)]
pub struct StringCodec;

impl TokenCodec<String> for StringCodec {
    fn id(&self) -> &str {
        "yurine:string:utf8"
    }

    fn encode(&self, token: &String, output: &mut Vec<u8>) -> Result<()> {
        output.extend_from_slice(token.as_bytes());
        Ok(())
    }

    fn decode(&self, bytes: &[u8]) -> Result<String> {
        String::from_utf8(bytes.to_vec())
            .map_err(|error| Error::InvalidTokenEncoding(error.to_string()))
    }
}

/// The standard little-endian Unicode scalar codec for [`char`] tokens.
#[derive(Debug, Clone, Copy, Default)]
pub struct CharCodec;

impl TokenCodec<char> for CharCodec {
    fn id(&self) -> &str {
        "yurine:char:u32le"
    }

    fn encode(&self, token: &char, output: &mut Vec<u8>) -> Result<()> {
        output.extend_from_slice(&u32::from(*token).to_le_bytes());
        Ok(())
    }

    fn decode(&self, bytes: &[u8]) -> Result<char> {
        let raw: [u8; 4] = bytes
            .try_into()
            .map_err(|_| Error::InvalidTokenEncoding("char must contain four bytes".into()))?;
        char::from_u32(u32::from_le_bytes(raw))
            .ok_or_else(|| Error::InvalidTokenEncoding("char is not a Unicode scalar value".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::{CharCodec, StringCodec, TokenCodec};
    use crate::errors::Error;

    #[test]
    fn string_codec_round_trips_utf8() {
        let codec = StringCodec;
        let mut bytes = Vec::new();
        codec.encode(&"東京".to_owned(), &mut bytes).unwrap();

        assert_eq!(codec.decode(&bytes).unwrap(), "東京");
    }

    #[test]
    fn char_codec_uses_little_endian_u32() {
        let codec = CharCodec;
        let mut bytes = Vec::new();
        codec.encode(&'東', &mut bytes).unwrap();

        assert_eq!(bytes, u32::from('東').to_le_bytes());
        assert_eq!(codec.decode(&bytes).unwrap(), '東');
    }

    #[test]
    fn codecs_reject_invalid_input() {
        assert!(matches!(
            StringCodec.decode(&[0xff]),
            Err(Error::InvalidTokenEncoding(_))
        ));
        assert!(matches!(
            CharCodec.decode(&[0; 3]),
            Err(Error::InvalidTokenEncoding(_))
        ));
        assert!(matches!(
            CharCodec.decode(&0xd800_u32.to_le_bytes()),
            Err(Error::InvalidTokenEncoding(_))
        ));
    }
}
