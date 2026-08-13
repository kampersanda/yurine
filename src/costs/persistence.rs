use crate::errors::{Error, Result};
use crate::persistence::TokenCodec;

pub(crate) struct NoTokenCodec;

impl TokenCodec<()> for NoTokenCodec {
    fn id(&self) -> &str {
        "yurine:none"
    }

    fn encode(&self, _: &(), _: &mut Vec<u8>) -> Result<()> {
        Err(Error::InvalidTokenEncoding(
            "this file kind does not contain tokens".into(),
        ))
    }

    fn decode(&self, _: &[u8]) -> Result<()> {
        Err(Error::InvalidTokenEncoding(
            "this file kind does not contain tokens".into(),
        ))
    }
}

pub(crate) struct MetadataReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> MetadataReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    pub(crate) fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.array()?))
    }

    pub(crate) fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = usize::try_from(self.u64()?).map_err(|_| Error::PlatformSizeOverflow)?;
        let end = self
            .position
            .checked_add(len)
            .ok_or(Error::InvalidFile("metadata range overflows"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or(Error::InvalidFile("cost metadata is truncated"))?;
        self.position = end;
        Ok(value)
    }

    pub(crate) fn finish(self) -> Result<()> {
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::InvalidFile("cost metadata has trailing bytes"))
        }
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .position
            .checked_add(N)
            .ok_or(Error::InvalidFile("metadata range overflows"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or(Error::InvalidFile("cost metadata is truncated"))?
            .try_into()
            .unwrap();
        self.position = end;
        Ok(value)
    }
}

pub(crate) fn push_bytes(output: &mut Vec<u8>, bytes: &[u8]) {
    output.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    output.extend_from_slice(bytes);
}
