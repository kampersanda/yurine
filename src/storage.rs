use std::ops::Deref;

#[cfg(feature = "persist")]
use crate::persistence::storage::MappedSlice;

/// Either an in-memory array or a typed view into an immutable mapping.
#[derive(Debug)]
pub(crate) enum Storage<T> {
    /// Values produced and validated by an in-memory builder.
    Owned(Box<[T]>),
    /// Values read from an external snapshot whose payload is validated lazily.
    #[cfg(feature = "persist")]
    Mapped(MappedSlice<T>),
}

impl<T> Storage<T> {
    /// Returns the values independently of their backing storage.
    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            Self::Owned(values) => values,
            #[cfg(feature = "persist")]
            Self::Mapped(values) => values,
        }
    }

    /// Returns whether the values originate from an external mapped snapshot.
    ///
    /// Callers use this distinction to avoid repeating semantic checks for
    /// owned values that were already validated during construction.
    pub(crate) fn is_mapped(&self) -> bool {
        match self {
            Self::Owned(_) => false,
            #[cfg(feature = "persist")]
            Self::Mapped(_) => true,
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
        assert!(!values.is_mapped());
    }
}
