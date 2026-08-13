use std::path::Path;

use super::LevenshteinCosts;
use crate::costs::persistence::NoTokenCodec;
use crate::errors::Result;
use crate::persistence::format::{FileKind, PersistedFile, SectionData, SectionKind, write_file};

impl LevenshteinCosts {
    /// Saves the built-in cost kind and persistence format version.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let sections: [(SectionKind, SectionData<'_>); 0] = [];
        write_file(
            path.as_ref(),
            FileKind::LevenshteinCosts,
            &NoTokenCodec,
            &sections,
        )
    }

    /// Opens a persisted Levenshtein cost policy.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        PersistedFile::open(path.as_ref(), FileKind::LevenshteinCosts, &NoTokenCodec)?;
        Ok(Self)
    }

    /// Fully validates this cost policy.
    pub fn verify(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn levenshtein_costs_round_trip_deterministically() {
        let directory = tempdir().unwrap();
        let first = directory.path().join("first.yurine");
        let second = directory.path().join("second.yurine");
        LevenshteinCosts.save(&first).unwrap();
        LevenshteinCosts.save(&second).unwrap();

        LevenshteinCosts::open(&first).unwrap().verify().unwrap();
        assert_eq!(fs::read(second).unwrap(), fs::read(first).unwrap());
    }
}
