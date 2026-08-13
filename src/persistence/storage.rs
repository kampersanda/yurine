use std::fs::File;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use zerocopy::{FromBytes, Immutable, KnownLayout};

use crate::errors::{Error, Result};

/// A typed slice backed by one immutable, read-only file mapping.
pub(crate) struct MappedSlice<T> {
    mmap: Arc<Mmap>,
    pointer: NonNull<T>,
    len: usize,
    marker: PhantomData<T>,
}

// The pointer refers to an immutable mapping kept alive by `mmap`.
unsafe impl<T: Sync> Send for MappedSlice<T> {}
unsafe impl<T: Sync> Sync for MappedSlice<T> {}

impl<T> Clone for MappedSlice<T> {
    fn clone(&self) -> Self {
        Self {
            mmap: Arc::clone(&self.mmap),
            pointer: self.pointer,
            len: self.len,
            marker: PhantomData,
        }
    }
}

impl<T> MappedSlice<T>
where
    T: FromBytes + Immutable + KnownLayout,
{
    /// Creates a typed view after validating its range, length, and alignment.
    pub(crate) fn new(
        mmap: Arc<Mmap>,
        offset: u64,
        byte_len: u64,
        element_count: u64,
    ) -> Result<Self> {
        let expected_len = element_count
            .checked_mul(u64::try_from(size_of::<T>()).unwrap())
            .ok_or(Error::InvalidFile("section byte length overflows"))?;
        if byte_len != expected_len {
            return Err(Error::InvalidFile(
                "section byte length does not match element count",
            ));
        }

        let offset = usize::try_from(offset).map_err(|_| Error::PlatformSizeOverflow)?;
        let byte_len = usize::try_from(byte_len).map_err(|_| Error::PlatformSizeOverflow)?;
        let end = offset
            .checked_add(byte_len)
            .ok_or(Error::InvalidFile("section range overflows"))?;
        let bytes = mmap
            .get(offset..end)
            .ok_or(Error::InvalidFile("section lies outside the file"))?;
        let values = <[T]>::ref_from_bytes(bytes)
            .map_err(|_| Error::InvalidFile("section has invalid length or alignment"))?;
        let len = values.len();
        let pointer = NonNull::new(values.as_ptr().cast_mut()).unwrap_or_else(NonNull::dangling);

        Ok(Self {
            mmap,
            pointer,
            len,
            marker: PhantomData,
        })
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: `new` validated the typed slice and stored its pointer and
        // length. The mapping is immutable, has a stable address, and is kept
        // alive by `self.mmap` for the returned borrow.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }
}

impl<T> Deref for MappedSlice<T>
where
    T: FromBytes + Immutable + KnownLayout,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Either an in-memory array or a typed view into an immutable mapping.
pub(crate) enum Storage<T> {
    Owned(Box<[T]>),
    Mapped(MappedSlice<T>),
}

impl<T> From<MappedSlice<T>> for Storage<T> {
    fn from(values: MappedSlice<T>) -> Self {
        Self::Mapped(values)
    }
}

impl<T> Storage<T>
where
    T: FromBytes + Immutable + KnownLayout,
{
    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            Self::Owned(values) => values,
            Self::Mapped(values) => values,
        }
    }
}

impl<T> Deref for Storage<T>
where
    T: FromBytes + Immutable + KnownLayout,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Maps an immutable snapshot file once for all of its section views.
pub(crate) fn map_file(file: &File) -> Result<Arc<Mmap>> {
    if file.metadata()?.len() == 0 {
        return Err(Error::InvalidFile("file is empty"));
    }

    // SAFETY: Persisted files are immutable snapshots. Yurine never modifies
    // or truncates a published file in place; replacements are published by
    // atomic rename. Callers must uphold the same rule, documented on the
    // persistence module and future open APIs.
    let mmap = unsafe { MmapOptions::new().map(file)? };
    Ok(Arc::new(mmap))
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::sync::Arc;

    use memmap2::MmapOptions;

    use super::{MappedSlice, Storage};
    use crate::errors::Error;

    fn test_map(bytes: &[u8]) -> Arc<memmap2::Mmap> {
        let mut mmap = MmapOptions::new().len(bytes.len()).map_anon().unwrap();
        mmap.copy_from_slice(bytes);
        Arc::new(mmap.make_read_only().unwrap())
    }

    #[test]
    fn owned_and_mapped_storage_have_the_same_slice_interface() {
        let values = [11_u32, 22, 33];
        let bytes = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let mapped = MappedSlice::new(test_map(&bytes), 0, 12, 3).unwrap();
        let owned = Storage::Owned(values.into());
        let mapped = Storage::Mapped(mapped);

        assert_eq!(owned.as_slice(), mapped.as_slice());
    }

    #[test]
    fn mapped_slice_rejects_bad_length_and_range() {
        let mmap = test_map(&[0; 8]);
        assert!(matches!(
            MappedSlice::<u32>::new(Arc::clone(&mmap), 0, 7, 2),
            Err(Error::InvalidFile(_))
        ));
        assert!(matches!(
            MappedSlice::<u32>::new(mmap, 4, 8, 2),
            Err(Error::InvalidFile(_))
        ));
    }

    #[test]
    fn empty_file_is_rejected_before_mapping() {
        let path = std::env::temp_dir().join(format!(
            "yurine-empty-map-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::write(&path, []).unwrap();
        let file = File::open(&path).unwrap();
        let result = super::map_file(&file);
        fs::remove_file(path).unwrap();

        assert!(matches!(result, Err(Error::InvalidFile("file is empty"))));
    }
}
