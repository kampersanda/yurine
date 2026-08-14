//! On-disk layout of a compiled edit-cost snapshot directory.
//!
//! A snapshot holds the cost policy of a JSON configuration in the persisted
//! form of the library, so a search opens it instead of parsing the
//! configuration and its data files again. Parsing dominates the run time of a
//! search whenever the policy covers a large vocabulary.
//!
//! Snapshots are independent of an index: one policy can be used with several
//! indexes, and one index with several policies. Only the tokenizer has to
//! agree, because it decides how the tokens of a configuration are read.

use std::fs;
use std::hash::Hash;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use yurine::costs::{CosineEmbeddingCosts, CustomCosts, EmbeddingStore, LevenshteinCosts};
use yurine::persistence::TokenCodec;

use crate::cost_config::RuntimeCosts;
use crate::tokenization::TokenizerKind;

const FORMAT_VERSION: u32 = 1;
const METADATA_FILE: &str = "metadata.json";
const COSTS_FILE: &str = "costs.yurine";
const STORE_FILE: &str = "store.yurine";
const TEMPORARY_SUFFIX: &str = ".tmp";

/// Description of a snapshot directory, stored as `metadata.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Metadata {
    pub(crate) version: u32,
    pub(crate) kind: CostKind,
    pub(crate) tokenizer: TokenizerKind,
}

/// Which cost policy a snapshot holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CostKind {
    Levenshtein,
    Embedding,
    Custom,
}

/// Writes `costs` to `directory`, replacing the snapshot of a previous run.
///
/// The cost files are written under temporary names and renamed into place
/// only once all of them have been written, so a failed run leaves the
/// snapshot of a previous one usable.
pub(crate) fn save<T, C>(
    directory: &Path,
    costs: &RuntimeCosts<T>,
    tokenizer: TokenizerKind,
    codec: &C,
) -> Result<()>
where
    T: Eq + Hash,
    C: TokenCodec<T>,
{
    let policy = temporary(directory, COSTS_FILE);
    let (kind, files): (CostKind, &[&str]) = match costs {
        RuntimeCosts::Levenshtein(costs) => {
            costs.save(&policy).with_context(|| written(&policy))?;
            (CostKind::Levenshtein, &[COSTS_FILE])
        }
        RuntimeCosts::Embedding(costs) => {
            let store = temporary(directory, STORE_FILE);
            costs
                .embeddings()
                .save_with(&store, codec)
                .with_context(|| written(&store))?;
            costs.save(&policy).with_context(|| written(&policy))?;
            (CostKind::Embedding, &[COSTS_FILE, STORE_FILE])
        }
        RuntimeCosts::Custom(costs) => {
            costs
                .save_with(&policy, codec)
                .with_context(|| written(&policy))?;
            (CostKind::Custom, &[COSTS_FILE])
        }
    };

    for name in files {
        let path = directory.join(name);
        fs::rename(temporary(directory, name), &path)
            .with_context(|| format!("failed to publish '{}'", path.display()))?;
    }
    // A policy without embeddings never reads a store left by a previous run.
    if kind != CostKind::Embedding {
        remove_if_present(&directory.join(STORE_FILE))?;
    }
    write_metadata(directory, kind, tokenizer)
}

/// Opens the snapshot in `directory` for an index tokenized with `tokenizer`.
pub(crate) fn open<T, C>(
    directory: &Path,
    tokenizer: TokenizerKind,
    codec: &C,
) -> Result<RuntimeCosts<T>>
where
    T: Eq + Hash,
    C: TokenCodec<T>,
{
    let metadata = read_metadata(directory)?;
    if metadata.tokenizer != tokenizer {
        bail!(
            "cost snapshot '{}' uses the {} tokenizer, but the index uses {}",
            directory.display(),
            metadata.tokenizer,
            tokenizer
        );
    }

    let policy = directory.join(COSTS_FILE);
    let costs = match metadata.kind {
        CostKind::Levenshtein => RuntimeCosts::Levenshtein(
            LevenshteinCosts::open(&policy).with_context(|| opened(&policy))?,
        ),
        CostKind::Embedding => {
            let store = directory.join(STORE_FILE);
            let embeddings =
                EmbeddingStore::open_with(&store, codec).with_context(|| opened(&store))?;
            RuntimeCosts::Embedding(
                CosineEmbeddingCosts::open(&policy, embeddings).with_context(|| opened(&policy))?,
            )
        }
        CostKind::Custom => RuntimeCosts::Custom(
            CustomCosts::open_with(&policy, codec).with_context(|| opened(&policy))?,
        ),
    };
    Ok(costs)
}

fn write_metadata(directory: &Path, kind: CostKind, tokenizer: TokenizerKind) -> Result<()> {
    let metadata = Metadata {
        version: FORMAT_VERSION,
        kind,
        tokenizer,
    };
    let path = directory.join(METADATA_FILE);
    let contents = serde_json::to_string_pretty(&metadata)?;
    fs::write(&path, format!("{contents}\n"))
        .with_context(|| format!("failed to write '{}'", path.display()))
}

pub(crate) fn read_metadata(directory: &Path) -> Result<Metadata> {
    let path = directory.join(METADATA_FILE);
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("failed to read cost snapshot metadata '{}'", path.display()))?;
    let metadata: Metadata = serde_json::from_str(&contents).with_context(|| {
        format!(
            "failed to parse cost snapshot metadata '{}'",
            path.display()
        )
    })?;
    if metadata.version != FORMAT_VERSION {
        bail!(
            "cost snapshot '{}' has unsupported format version {} (expected {FORMAT_VERSION})",
            directory.display(),
            metadata.version
        );
    }
    Ok(metadata)
}

fn temporary(directory: &Path, name: &str) -> PathBuf {
    directory.join(format!("{name}{TEMPORARY_SUFFIX}"))
}

fn remove_if_present(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        result => result.with_context(|| format!("failed to remove '{}'", path.display())),
    }
}

fn written(path: &Path) -> String {
    format!("failed to write '{}'", path.display())
}

fn opened(path: &Path) -> String {
    format!("failed to open '{}'", path.display())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use yurine::costs::EditCosts;
    use yurine::persistence::{CharCodec, StringCodec};

    use super::{CostKind, Metadata, open, read_metadata, save};
    use crate::cost_config::{self, RuntimeCosts};
    use crate::tests::TestDirectory;
    use crate::tokenization::{CharacterTokenizer, TokenizerKind, WhitespaceTokenizer};

    /// Writes a cost configuration and compiles it into `snapshot`.
    fn compile_custom(directory: &TestDirectory) -> std::path::PathBuf {
        directory.write(
            "rules.jsonl",
            "{\"operation\":\"substitution\",\"from\":\"x\",\"to\":\"a\",\"cost\":0.25}\n",
        );
        let config = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "custom",
                "defaults": {"deletion": 0.5},
                "rules": {"path": "rules.jsonl", "format": "jsonl"}
            }"#,
        );
        let costs = cost_config::load(&config, &CharacterTokenizer).unwrap();
        let snapshot = directory.path().join("snapshot");
        fs::create_dir(&snapshot).unwrap();
        save(&snapshot, &costs, TokenizerKind::Character, &CharCodec).unwrap();
        snapshot
    }

    #[test]
    fn custom_costs_round_trip_through_a_snapshot() {
        let directory = TestDirectory::new();
        let snapshot = compile_custom(&directory);

        let costs: RuntimeCosts<char> =
            open(&snapshot, TokenizerKind::Character, &CharCodec).unwrap();

        assert_eq!(costs.substitution(&'x', &'a'), 0.25);
        assert_eq!(costs.substitution(&'a', &'x'), 1.0);
        assert_eq!(costs.deletion(&'x'), 0.5);
        assert_eq!(
            read_metadata(&snapshot).unwrap(),
            Metadata {
                version: 1,
                kind: CostKind::Custom,
                tokenizer: TokenizerKind::Character,
            }
        );
    }

    #[test]
    fn embedding_costs_round_trip_through_a_snapshot() {
        let directory = TestDirectory::new();
        directory.write(
            "embeddings.jsonl",
            concat!(
                "{\"token\":\"colour\",\"embedding\":[1.0,0.0]}\n",
                "{\"token\":\"color\",\"embedding\":[0.8,0.6]}\n"
            ),
        );
        let config = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "embedding",
                "embeddings": {"path": "embeddings.jsonl", "format": "jsonl"},
                "deletion_cost": 0.5
            }"#,
        );
        let compiled = cost_config::load(&config, &WhitespaceTokenizer).unwrap();
        let snapshot = directory.path().join("snapshot");
        fs::create_dir(&snapshot).unwrap();
        save(
            &snapshot,
            &compiled,
            TokenizerKind::Whitespace,
            &StringCodec,
        )
        .unwrap();

        let costs: RuntimeCosts<String> =
            open(&snapshot, TokenizerKind::Whitespace, &StringCodec).unwrap();

        let distance = costs
            .substitution(&"colour".to_owned(), &"color".to_owned())
            .get();
        assert!((distance - 0.2).abs() < 1e-6);
        assert_eq!(costs.deletion(&"colour".to_owned()), 0.5);
        assert_eq!(read_metadata(&snapshot).unwrap().kind, CostKind::Embedding);
        costs.verify().unwrap();
    }

    #[test]
    fn levenshtein_costs_round_trip_through_a_snapshot() {
        let directory = TestDirectory::new();
        let snapshot = directory.path().join("snapshot");
        fs::create_dir(&snapshot).unwrap();
        save(
            &snapshot,
            &RuntimeCosts::<char>::levenshtein(),
            TokenizerKind::Character,
            &CharCodec,
        )
        .unwrap();

        let costs: RuntimeCosts<char> =
            open(&snapshot, TokenizerKind::Character, &CharCodec).unwrap();

        assert_eq!(costs.substitution(&'x', &'a'), 1.0);
        assert!(!snapshot.join("store.yurine").exists());
    }

    #[test]
    fn a_replaced_snapshot_does_not_keep_the_previous_embedding_store() {
        let directory = TestDirectory::new();
        directory.write(
            "embeddings.jsonl",
            "{\"token\":\"a\",\"embedding\":[1.0]}\n",
        );
        let config = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "embedding",
                "embeddings": {"path": "embeddings.jsonl", "format": "jsonl"}
            }"#,
        );
        let embedding = cost_config::load(&config, &CharacterTokenizer).unwrap();
        let snapshot = directory.path().join("snapshot");
        fs::create_dir(&snapshot).unwrap();
        save(&snapshot, &embedding, TokenizerKind::Character, &CharCodec).unwrap();
        assert!(snapshot.join("store.yurine").exists());

        save(
            &snapshot,
            &RuntimeCosts::<char>::levenshtein(),
            TokenizerKind::Character,
            &CharCodec,
        )
        .unwrap();

        assert!(!snapshot.join("store.yurine").exists());
        assert_eq!(
            read_metadata(&snapshot).unwrap().kind,
            CostKind::Levenshtein
        );
    }

    #[test]
    fn opening_rejects_a_tokenizer_that_does_not_match_the_index() {
        let directory = TestDirectory::new();
        let snapshot = compile_custom(&directory);

        let error =
            open::<String, _>(&snapshot, TokenizerKind::Whitespace, &StringCodec).unwrap_err();

        let message = error.to_string();
        assert!(message.contains("uses the character tokenizer"));
        assert!(message.contains("the index uses whitespace"));
    }

    #[test]
    fn opening_reports_a_missing_or_unsupported_snapshot() {
        let directory = TestDirectory::new();
        let missing = directory.path().join("absent");
        let unsupported = directory.path().join("unsupported");
        fs::create_dir(&unsupported).unwrap();
        fs::write(
            unsupported.join("metadata.json"),
            r#"{"version": 2, "kind": "custom", "tokenizer": "character"}"#,
        )
        .unwrap();

        let missing_error = open::<char, _>(&missing, TokenizerKind::Character, &CharCodec)
            .unwrap_err()
            .to_string();
        let unsupported_error = open::<char, _>(&unsupported, TokenizerKind::Character, &CharCodec)
            .unwrap_err()
            .to_string();

        assert!(missing_error.contains("failed to read cost snapshot metadata"));
        assert!(unsupported_error.contains("unsupported format version 2"));
    }

    #[test]
    fn opening_reports_a_snapshot_whose_files_are_missing() {
        let directory = TestDirectory::new();
        let snapshot = compile_custom(&directory);
        fs::remove_file(snapshot.join("costs.yurine")).unwrap();

        let error = open::<char, _>(&snapshot, TokenizerKind::Character, &CharCodec).unwrap_err();

        assert!(error.to_string().contains("failed to open"));
    }
}
