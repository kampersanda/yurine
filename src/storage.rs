use std::ops::Deref;

#[cfg(feature = "persist")]
use crate::persistence::storage::MappedSlice;

/// Either an in-memory array or a typed view into an immutable mapping.
#[derive(Debug)]
pub(crate) enum Storage<T> {
    Owned(Box<[T]>),
    #[cfg(feature = "persist")]
    Mapped(MappedSlice<T>),
}

impl<T> Storage<T> {
    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            Self::Owned(values) => values,
            #[cfg(feature = "persist")]
            Self::Mapped(values) => values,
        }
    }
}

impl<T> Deref for Storage<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::Storage;
    use crate::types::Symbol;

    #[test]
    fn owned_storage_does_not_require_zerocopy_traits() {
        let values = Storage::Owned(vec![Symbol::new(7)].into_boxed_slice());

        assert_eq!(values[0].get(), 7);
        assert!(format!("{values:?}").contains("Symbol"));
    }
}
