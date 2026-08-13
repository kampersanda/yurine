use std::collections::HashMap;
use std::hash::Hash;
use std::path::Path;

use super::CustomCosts;
use crate::costs::Cost;
use crate::costs::persistence::{MetadataReader, push_bytes};
use crate::errors::{Error, Result};
use crate::persistence::TokenCodec;
use crate::persistence::format::{FileKind, PersistedFile, SectionData, SectionKind, write_file};

const METADATA_VERSION: u32 = 1;

impl<T> CustomCosts<T>
where
    T: Eq + Hash,
{
    /// Saves defaults and token-specific rules in deterministic codec order.
    pub fn save_with<C: TokenCodec<T>>(&self, path: impl AsRef<Path>, codec: &C) -> Result<()> {
        let mut substitutions = Vec::new();
        for (from, targets) in &self.substitutions {
            let from = encode_token(from, codec)?;
            for (to, cost) in targets {
                substitutions.push((from.clone(), encode_token(to, codec)?, *cost));
            }
        }
        substitutions.sort_unstable_by(|left, right| (&left.0, &left.1).cmp(&(&right.0, &right.1)));

        let mut deletions = self
            .deletions
            .iter()
            .map(|(token, cost)| Ok((encode_token(token, codec)?, *cost)))
            .collect::<Result<Vec<_>>>()?;
        deletions.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let mut insertions = self
            .insertions
            .iter()
            .map(|(token, cost)| Ok((encode_token(token, codec)?, *cost)))
            .collect::<Result<Vec<_>>>()?;
        insertions.sort_unstable_by(|left, right| left.0.cmp(&right.0));

        reject_duplicate_encodings(&deletions)?;
        reject_duplicate_encodings(&insertions)?;
        if substitutions
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1)
        {
            return Err(Error::InvalidTokenEncoding(
                "token codec maps distinct substitution rules to the same bytes".into(),
            ));
        }

        let mut metadata = Vec::new();
        metadata.extend_from_slice(&METADATA_VERSION.to_le_bytes());
        for cost in [
            self.default_substitution,
            self.default_deletion,
            self.default_insertion,
        ] {
            metadata.extend_from_slice(&cost.get().to_le_bytes());
        }
        metadata.extend_from_slice(&(substitutions.len() as u64).to_le_bytes());
        metadata.extend_from_slice(&(deletions.len() as u64).to_le_bytes());
        metadata.extend_from_slice(&(insertions.len() as u64).to_le_bytes());
        for (from, to, cost) in substitutions {
            push_bytes(&mut metadata, &from);
            push_bytes(&mut metadata, &to);
            metadata.extend_from_slice(&cost.get().to_le_bytes());
        }
        for (token, cost) in deletions.into_iter().chain(insertions) {
            push_bytes(&mut metadata, &token);
            metadata.extend_from_slice(&cost.get().to_le_bytes());
        }

        let sections = [(
            SectionKind::CostMetadata,
            SectionData::Bytes {
                bytes: &metadata,
                element_count: 1,
            },
        )];
        write_file(path.as_ref(), FileKind::CustomCosts, codec, &sections)
    }

    /// Opens a saved custom cost policy into memory.
    pub fn open_with<C: TokenCodec<T>>(path: impl AsRef<Path>, codec: &C) -> Result<Self> {
        let file = PersistedFile::open(path.as_ref(), FileKind::CustomCosts, codec)?;
        let mut reader = MetadataReader::new(file.bytes(SectionKind::CostMetadata)?);
        if reader.u32()? != METADATA_VERSION {
            return Err(Error::InvalidFile(
                "unsupported custom-cost metadata version",
            ));
        }
        let default_substitution = Cost::new(reader.f32()?)?;
        let default_deletion = Cost::new(reader.f32()?)?;
        let default_insertion = Cost::new(reader.f32()?)?;
        let substitution_count = count(reader.u64()?)?;
        let deletion_count = count(reader.u64()?)?;
        let insertion_count = count(reader.u64()?)?;
        validate_rule_counts(
            reader.remaining(),
            substitution_count,
            deletion_count,
            insertion_count,
        )?;

        let mut substitutions: HashMap<T, HashMap<T, Cost>> = HashMap::new();
        for _ in 0..substitution_count {
            let from = codec.decode(reader.bytes()?)?;
            let to = codec.decode(reader.bytes()?)?;
            let cost = Cost::new(reader.f32()?)?;
            if from == to {
                return Err(Error::InvalidFile(
                    "equal-token substitution override is not allowed",
                ));
            }
            if substitutions
                .entry(from)
                .or_default()
                .insert(to, cost)
                .is_some()
            {
                return Err(Error::InvalidFile("substitution rule is duplicated"));
            }
        }
        let mut deletions = HashMap::with_capacity(deletion_count);
        for _ in 0..deletion_count {
            let token = codec.decode(reader.bytes()?)?;
            let cost = Cost::new(reader.f32()?)?;
            if deletions.insert(token, cost).is_some() {
                return Err(Error::InvalidFile("deletion rule is duplicated"));
            }
        }
        let mut insertions = HashMap::with_capacity(insertion_count);
        for _ in 0..insertion_count {
            let token = codec.decode(reader.bytes()?)?;
            let cost = Cost::new(reader.f32()?)?;
            if insertions.insert(token, cost).is_some() {
                return Err(Error::InvalidFile("insertion rule is duplicated"));
            }
        }
        reader.finish()?;
        Ok(Self {
            default_substitution,
            default_deletion,
            default_insertion,
            substitutions,
            deletions,
            insertions,
        })
    }

    /// Validates all stored cost values.
    pub fn verify(&self) -> Result<()> {
        for cost in self
            .substitutions
            .values()
            .flat_map(HashMap::values)
            .chain(self.deletions.values())
            .chain(self.insertions.values())
            .copied()
            .chain([
                self.default_substitution,
                self.default_deletion,
                self.default_insertion,
            ])
        {
            Cost::new(cost.get())?;
        }
        Ok(())
    }
}

fn encode_token<T: Eq, C: TokenCodec<T>>(token: &T, codec: &C) -> Result<Vec<u8>> {
    let mut encoded = Vec::new();
    codec.encode(token, &mut encoded)?;
    if codec.decode(&encoded)? != *token {
        return Err(Error::InvalidTokenEncoding(
            "token codec does not round-trip a custom-cost token".into(),
        ));
    }
    Ok(encoded)
}

fn reject_duplicate_encodings(rows: &[(Vec<u8>, Cost)]) -> Result<()> {
    if rows.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        Err(Error::InvalidTokenEncoding(
            "token codec maps distinct custom-cost tokens to the same bytes".into(),
        ))
    } else {
        Ok(())
    }
}

fn count(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::PlatformSizeOverflow)
}

fn validate_rule_counts(
    remaining: usize,
    substitutions: usize,
    deletions: usize,
    insertions: usize,
) -> Result<()> {
    let minimum_len = substitutions
        .checked_mul(20)
        .and_then(|len| deletions.checked_mul(12)?.checked_add(len))
        .and_then(|len| insertions.checked_mul(12)?.checked_add(len));
    if minimum_len.is_none_or(|minimum_len| minimum_len > remaining) {
        return Err(Error::InvalidFile(
            "custom-cost rule counts exceed metadata size",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;
    use crate::costs::EditCosts;
    use crate::persistence::CharCodec;

    fn costs() -> CustomCosts<char> {
        let mut costs = CustomCosts::new(
            Cost::new_const(2.0),
            Cost::new_const(3.0),
            Cost::new_const(4.0),
        );
        costs.set_substitution('a', 'A', Cost::new_const(0.25));
        costs.set_substitution('b', 'B', Cost::new_const(0.3));
        costs.set_deletion('.', Cost::new_const(0.5));
        costs.set_deletion('?', Cost::new_const(0.6));
        costs.set_insertion('-', Cost::new_const(0.75));
        costs.set_insertion('_', Cost::new_const(0.8));
        costs
    }

    fn costs_reordered() -> CustomCosts<char> {
        let mut costs = CustomCosts::new(
            Cost::new_const(2.0),
            Cost::new_const(3.0),
            Cost::new_const(4.0),
        );
        costs.set_insertion('_', Cost::new_const(0.8));
        costs.set_insertion('-', Cost::new_const(0.75));
        costs.set_deletion('?', Cost::new_const(0.6));
        costs.set_deletion('.', Cost::new_const(0.5));
        costs.set_substitution('b', 'B', Cost::new_const(0.3));
        costs.set_substitution('a', 'A', Cost::new_const(0.25));
        costs
    }

    #[test]
    fn custom_costs_round_trip() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("custom.yurine");
        costs().save_with(&path, &CharCodec).unwrap();
        let opened = CustomCosts::open_with(path, &CharCodec).unwrap();

        assert_eq!(opened.substitution(&'x', &'y'), Cost::new_const(2.0));
        assert_eq!(opened.substitution(&'a', &'A'), Cost::new_const(0.25));
        assert_eq!(opened.deletion(&'.'), Cost::new_const(0.5));
        assert_eq!(opened.insertion(&'-'), Cost::new_const(0.75));
        opened.verify().unwrap();
    }

    #[test]
    fn custom_cost_files_are_byte_deterministic() {
        let directory = tempdir().unwrap();
        let first = directory.path().join("first.yurine");
        let second = directory.path().join("second.yurine");
        costs().save_with(&first, &CharCodec).unwrap();
        costs_reordered().save_with(&second, &CharCodec).unwrap();
        assert_eq!(fs::read(first).unwrap(), fs::read(second).unwrap());
    }

    #[test]
    fn corrupt_default_cost_is_rejected_without_panicking() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("corrupt.yurine");
        costs().save_with(&path, &CharCodec).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let metadata = section_offset(&bytes, SectionKind::CostMetadata as u32);
        bytes[metadata + 4..metadata + 8].copy_from_slice(&f32::NAN.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let result = CustomCosts::<char>::open_with(path, &CharCodec);
        assert!(matches!(result, Err(Error::InvalidCost(value)) if value.is_nan()));
    }

    #[test]
    fn corrupt_rule_count_is_rejected_before_allocation() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("corrupt-count.yurine");
        costs().save_with(&path, &CharCodec).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let metadata = section_offset(&bytes, SectionKind::CostMetadata as u32);
        bytes[metadata + 24..metadata + 32].copy_from_slice(&u64::MAX.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        assert!(matches!(
            CustomCosts::<char>::open_with(path, &CharCodec),
            Err(Error::InvalidFile(
                "custom-cost rule counts exceed metadata size"
            ))
        ));
    }

    fn section_offset(bytes: &[u8], wanted_kind: u32) -> usize {
        let section_count = u32::from_le_bytes(bytes[28..32].try_into().unwrap()) as usize;
        let table = u64::from_le_bytes(bytes[48..56].try_into().unwrap()) as usize;
        (0..section_count)
            .find_map(|index| {
                let entry = table + index * 32;
                let kind = u32::from_le_bytes(bytes[entry..entry + 4].try_into().unwrap());
                (kind == wanted_kind).then(|| {
                    u64::from_le_bytes(bytes[entry + 8..entry + 16].try_into().unwrap()) as usize
                })
            })
            .unwrap()
    }
}
