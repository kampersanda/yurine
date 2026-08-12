use std::collections::HashSet;
use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use yurine::costs::custom::CustomCosts;
use yurine::costs::embedding::{CosineEmbeddingCosts, EmbeddingStore};
use yurine::costs::levenshtein::LevenshteinCosts;
use yurine::costs::{Cost, EditCosts};

use crate::tokenization::Tokenizer;

#[derive(Debug)]
pub(crate) enum RuntimeCosts<T> {
    Levenshtein(LevenshteinCosts),
    Embedding(CosineEmbeddingCosts<T>),
    Custom(CustomCosts<T>),
}

impl<T> RuntimeCosts<T> {
    pub(crate) const fn levenshtein() -> Self {
        Self::Levenshtein(LevenshteinCosts::new())
    }
}

impl<T> EditCosts<T> for RuntimeCosts<T>
where
    T: Eq + Hash,
{
    fn substitution(&self, from: &T, to: &T) -> Cost {
        match self {
            Self::Levenshtein(costs) => costs.substitution(from, to),
            Self::Embedding(costs) => costs.substitution(from, to),
            Self::Custom(costs) => costs.substitution(from, to),
        }
    }

    fn deletion(&self, token: &T) -> Cost {
        match self {
            Self::Levenshtein(costs) => costs.deletion(token),
            Self::Embedding(costs) => costs.deletion(token),
            Self::Custom(costs) => costs.deletion(token),
        }
    }

    fn insertion(&self, token: &T) -> Cost {
        match self {
            Self::Levenshtein(costs) => costs.insertion(token),
            Self::Embedding(costs) => costs.insertion(token),
            Self::Custom(costs) => costs.insertion(token),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum CostConfig {
    Embedding {
        version: u32,
        embeddings: DataSource,
        #[serde(default = "unit_cost")]
        missing_substitution_cost: f32,
        #[serde(default = "unit_cost")]
        deletion_cost: f32,
        #[serde(default = "unit_cost")]
        insertion_cost: f32,
    },
    Custom {
        version: u32,
        #[serde(default)]
        defaults: CustomDefaults,
        rules: DataSource,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DataSource {
    path: PathBuf,
    format: DataFormat,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum DataFormat {
    Jsonl,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CustomDefaults {
    substitution: f32,
    deletion: f32,
    insertion: f32,
}

impl Default for CustomDefaults {
    fn default() -> Self {
        Self {
            substitution: unit_cost(),
            deletion: unit_cost(),
            insertion: unit_cost(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingRecord {
    token: String,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum CustomRule {
    Substitution { from: String, to: String, cost: f32 },
    Deletion { token: String, cost: f32 },
    Insertion { token: String, cost: f32 },
}

pub(crate) fn load(path: &Path, tokenizer: &impl Tokenizer) -> Result<RuntimeCosts<String>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open cost configuration '{}'", path.display()))?;
    let config: CostConfig = serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("failed to parse cost configuration '{}'", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));

    let costs = (|| -> Result<RuntimeCosts<String>> {
        match config {
            CostConfig::Embedding {
                version,
                embeddings,
                missing_substitution_cost,
                deletion_cost,
                insertion_cost,
            } => {
                validate_version(version)?;
                let store =
                    load_embeddings(&base.join(embeddings.path), embeddings.format, tokenizer)?;
                let costs = CosineEmbeddingCosts::new(store)
                    .with_missing_substitution_cost(parse_cost(
                        missing_substitution_cost,
                        "missing_substitution_cost",
                    )?)
                    .with_deletion_cost(parse_cost(deletion_cost, "deletion_cost")?)
                    .with_insertion_cost(parse_cost(insertion_cost, "insertion_cost")?);
                Ok(RuntimeCosts::Embedding(costs))
            }
            CostConfig::Custom {
                version,
                defaults,
                rules,
            } => {
                validate_version(version)?;
                let costs =
                    load_custom_costs(&base.join(rules.path), rules.format, defaults, tokenizer)?;
                Ok(RuntimeCosts::Custom(costs))
            }
        }
    })();
    costs.with_context(|| format!("invalid cost configuration '{}'", path.display()))
}

fn validate_version(version: u32) -> Result<()> {
    if version != 1 {
        bail!("unsupported cost configuration version {version}; expected 1");
    }
    Ok(())
}

const fn unit_cost() -> f32 {
    1.0
}

fn parse_cost(value: f32, field: &str) -> Result<Cost> {
    Cost::new(value).with_context(|| format!("invalid cost configuration field '{field}'"))
}

fn load_embeddings(
    path: &Path,
    _format: DataFormat,
    tokenizer: &impl Tokenizer,
) -> Result<EmbeddingStore<String>> {
    let reader = open_jsonl(path, "embedding")?;
    let mut store = None;

    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line.with_context(|| {
            format!(
                "failed to read embedding file '{}' at line {line_number}",
                path.display()
            )
        })?;
        let record: EmbeddingRecord = serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse embedding file '{}' at line {line_number}",
                path.display()
            )
        })?;
        let token = single_token(tokenizer, &record.token).with_context(|| {
            format!(
                "invalid token in embedding file '{}' at line {line_number}",
                path.display()
            )
        })?;

        if store.is_none() {
            let dimension = NonZeroUsize::new(record.embedding.len()).with_context(|| {
                format!(
                    "embedding file '{}' has an empty embedding at line {line_number}",
                    path.display()
                )
            })?;
            store = Some(EmbeddingStore::new(dimension));
        }

        let previous = store
            .as_mut()
            .expect("embedding store is initialized above")
            .insert(token, record.embedding)
            .with_context(|| {
                format!(
                    "invalid embedding in '{}' at line {line_number}",
                    path.display()
                )
            })?;
        if previous.is_some() {
            bail!(
                "duplicate token in embedding file '{}' at line {line_number}",
                path.display()
            );
        }
    }

    store.with_context(|| format!("embedding file '{}' is empty", path.display()))
}

fn load_custom_costs(
    path: &Path,
    _format: DataFormat,
    defaults: CustomDefaults,
    tokenizer: &impl Tokenizer,
) -> Result<CustomCosts<String>> {
    let mut costs = CustomCosts::new(
        parse_cost(defaults.substitution, "defaults.substitution")?,
        parse_cost(defaults.deletion, "defaults.deletion")?,
        parse_cost(defaults.insertion, "defaults.insertion")?,
    );
    let reader = open_jsonl(path, "custom cost rule")?;
    let mut substitutions = HashSet::new();
    let mut deletions = HashSet::new();
    let mut insertions = HashSet::new();

    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line.with_context(|| {
            format!(
                "failed to read custom cost rule file '{}' at line {line_number}",
                path.display()
            )
        })?;
        let rule: CustomRule = serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse custom cost rule file '{}' at line {line_number}",
                path.display()
            )
        })?;

        match rule {
            CustomRule::Substitution { from, to, cost } => {
                let from = single_token(tokenizer, &from).with_context(|| {
                    format!(
                        "invalid 'from' token in custom cost rule file '{}' at line {line_number}",
                        path.display()
                    )
                })?;
                let to = single_token(tokenizer, &to).with_context(|| {
                    format!(
                        "invalid 'to' token in custom cost rule file '{}' at line {line_number}",
                        path.display()
                    )
                })?;
                if !substitutions.insert((from.clone(), to.clone())) {
                    bail!(
                        "duplicate substitution in custom cost rule file '{}' at line {line_number}",
                        path.display()
                    );
                }
                costs.set_substitution(from, to, parse_rule_cost(cost, path, line_number)?);
            }
            CustomRule::Deletion { token, cost } => {
                let token = single_token(tokenizer, &token).with_context(|| {
                    format!(
                        "invalid deletion token in custom cost rule file '{}' at line {line_number}",
                        path.display()
                    )
                })?;
                if !deletions.insert(token.clone()) {
                    bail!(
                        "duplicate deletion in custom cost rule file '{}' at line {line_number}",
                        path.display()
                    );
                }
                costs.set_deletion(token, parse_rule_cost(cost, path, line_number)?);
            }
            CustomRule::Insertion { token, cost } => {
                let token = single_token(tokenizer, &token).with_context(|| {
                    format!(
                        "invalid insertion token in custom cost rule file '{}' at line {line_number}",
                        path.display()
                    )
                })?;
                if !insertions.insert(token.clone()) {
                    bail!(
                        "duplicate insertion in custom cost rule file '{}' at line {line_number}",
                        path.display()
                    );
                }
                costs.set_insertion(token, parse_rule_cost(cost, path, line_number)?);
            }
        }
    }

    Ok(costs)
}

fn open_jsonl(path: &Path, kind: &str) -> Result<BufReader<File>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open {kind} file '{}'", path.display()))?;
    Ok(BufReader::new(file))
}

fn parse_rule_cost(value: f32, path: &Path, line_number: usize) -> Result<Cost> {
    Cost::new(value).with_context(|| {
        format!(
            "invalid cost in custom cost rule file '{}' at line {line_number}",
            path.display()
        )
    })
}

fn single_token(tokenizer: &impl Tokenizer, text: &str) -> Result<String> {
    let mut tokens = tokenizer.tokenize(text);
    if tokens.len() != 1 || tokens[0].byte_range != (0..text.len()) {
        bail!("value must contain exactly one complete token");
    }
    Ok(tokens.remove(0).value)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::load;
    use crate::tokenization::{CharacterTokenizer, WhitespaceTokenizer};
    use yurine::costs::EditCosts;

    static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory {
        path: std::path::PathBuf,
    }

    impl TestDirectory {
        fn new() -> Self {
            let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "yurine-cli-cost-config-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self { path }
        }

        fn write(&self, name: &str, contents: &str) -> std::path::PathBuf {
            let path = self.path.join(name);
            fs::write(&path, contents).unwrap();
            path
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.path).unwrap();
        }
    }

    #[test]
    fn loads_embedding_costs_and_resolves_relative_paths() {
        let directory = TestDirectory::new();
        directory.write(
            "embeddings.jsonl",
            concat!(
                "{\"token\":\"x\",\"embedding\":[1.0,0.0]}\n",
                "{\"token\":\"a\",\"embedding\":[0.8,0.6]}\n"
            ),
        );
        let config = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "embedding",
                "embeddings": {"path": "embeddings.jsonl", "format": "jsonl"},
                "missing_substitution_cost": 0.7,
                "deletion_cost": 0.3,
                "insertion_cost": 0.4
            }"#,
        );

        let costs = load(&config, &CharacterTokenizer).unwrap();

        assert!((costs.substitution(&"x".to_owned(), &"a".to_owned()).get() - 0.2).abs() < 1e-6);
        assert_eq!(costs.substitution(&"x".to_owned(), &"z".to_owned()), 0.7);
        assert_eq!(costs.deletion(&"x".to_owned()), 0.3);
        assert_eq!(costs.insertion(&"a".to_owned()), 0.4);
    }

    #[test]
    fn loads_custom_defaults_and_directed_rules() {
        let directory = TestDirectory::new();
        directory.write(
            "rules.jsonl",
            concat!(
                "{\"operation\":\"substitution\",\"from\":\"colour\",\"to\":\"color\",\"cost\":0.25}\n",
                "{\"operation\":\"deletion\",\"token\":\"the\",\"cost\":0.1}\n",
                "{\"operation\":\"insertion\",\"token\":\"a\",\"cost\":0.2}\n"
            ),
        );
        let config = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "custom",
                "defaults": {"substitution": 0.8, "deletion": 0.9},
                "rules": {"path": "rules.jsonl", "format": "jsonl"}
            }"#,
        );

        let costs = load(&config, &WhitespaceTokenizer).unwrap();

        assert_eq!(
            costs.substitution(&"colour".to_owned(), &"color".to_owned()),
            0.25
        );
        assert_eq!(
            costs.substitution(&"color".to_owned(), &"colour".to_owned()),
            0.8
        );
        assert_eq!(costs.deletion(&"the".to_owned()), 0.1);
        assert_eq!(costs.deletion(&"other".to_owned()), 0.9);
        assert_eq!(costs.insertion(&"a".to_owned()), 0.2);
        assert_eq!(costs.insertion(&"other".to_owned()), 1.0);
    }

    #[test]
    fn accepts_an_empty_custom_rule_file() {
        let directory = TestDirectory::new();
        directory.write("rules.jsonl", "");
        let config = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "custom",
                "rules": {"path": "rules.jsonl", "format": "jsonl"}
            }"#,
        );

        let costs = load(&config, &CharacterTokenizer).unwrap();

        assert_eq!(costs.substitution(&"a".to_owned(), &"b".to_owned()), 1.0);
        assert_eq!(costs.deletion(&"a".to_owned()), 1.0);
        assert_eq!(costs.insertion(&"a".to_owned()), 1.0);
    }

    #[test]
    fn rejects_unknown_configuration_fields_and_versions() {
        let directory = TestDirectory::new();
        let unknown = directory.write(
            "unknown.json",
            r#"{
                "version": 1,
                "type": "custom",
                "rules": {"path": "rules.jsonl", "format": "jsonl"},
                "extra": true
            }"#,
        );
        let version = directory.write(
            "version.json",
            r#"{
                "version": 2,
                "type": "custom",
                "rules": {"path": "rules.jsonl", "format": "jsonl"}
            }"#,
        );

        let unknown_error = load(&unknown, &CharacterTokenizer).unwrap_err();
        let version_error = load(&version, &CharacterTokenizer).unwrap_err();

        assert!(unknown_error.to_string().contains("unknown.json"));
        assert!(format!("{unknown_error:#}").contains("unknown field"));
        assert!(version_error.to_string().contains("version.json"));
        assert!(format!("{version_error:#}").contains("expected 1"));
    }

    #[test]
    fn reports_embedding_validation_errors_with_line_numbers() {
        let directory = TestDirectory::new();
        directory.write(
            "embeddings.jsonl",
            concat!(
                "{\"token\":\"a\",\"embedding\":[1.0,0.0]}\n",
                "{\"token\":\"b\",\"embedding\":[1.0]}\n"
            ),
        );
        let config = directory.write(
            "costs.json",
            r#"{
                "version": 1,
                "type": "embedding",
                "embeddings": {"path": "embeddings.jsonl", "format": "jsonl"}
            }"#,
        );

        let error = load(&config, &CharacterTokenizer).unwrap_err();
        let message = format!("{error:#}");

        assert!(message.contains("embeddings.jsonl"));
        assert!(message.contains("line 2"));
        assert!(message.contains("dimension must be 2"));
    }

    #[test]
    fn rejects_malformed_jsonl_and_duplicate_embedding_tokens() {
        let directory = TestDirectory::new();
        directory.write(
            "malformed.jsonl",
            concat!("{\"token\":\"a\",\"embedding\":[1.0,0.0]}\n", "not json\n"),
        );
        let malformed_config = directory.write(
            "malformed.json",
            r#"{
                "version": 1,
                "type": "embedding",
                "embeddings": {"path": "malformed.jsonl", "format": "jsonl"}
            }"#,
        );
        directory.write(
            "duplicate.jsonl",
            concat!(
                "{\"token\":\"a\",\"embedding\":[1.0,0.0]}\n",
                "{\"token\":\"a\",\"embedding\":[0.0,1.0]}\n"
            ),
        );
        let duplicate_config = directory.write(
            "duplicate.json",
            r#"{
                "version": 1,
                "type": "embedding",
                "embeddings": {"path": "duplicate.jsonl", "format": "jsonl"}
            }"#,
        );

        let malformed = load(&malformed_config, &CharacterTokenizer).unwrap_err();
        let duplicate = load(&duplicate_config, &CharacterTokenizer).unwrap_err();

        assert!(format!("{malformed:#}").contains("malformed.jsonl"));
        assert!(format!("{malformed:#}").contains("line 2"));
        assert!(format!("{duplicate:#}").contains("duplicate token"));
        assert!(format!("{duplicate:#}").contains("line 2"));
    }

    #[test]
    fn rejects_invalid_tokens_and_duplicate_custom_rules() {
        let directory = TestDirectory::new();
        directory.write(
            "invalid-token.jsonl",
            "{\"operation\":\"deletion\",\"token\":\"two words\",\"cost\":0.5}\n",
        );
        let invalid_token_config = directory.write(
            "invalid-token.json",
            r#"{
                "version": 1,
                "type": "custom",
                "rules": {"path": "invalid-token.jsonl", "format": "jsonl"}
            }"#,
        );
        directory.write(
            "duplicate.jsonl",
            concat!(
                "{\"operation\":\"insertion\",\"token\":\"a\",\"cost\":0.5}\n",
                "{\"operation\":\"insertion\",\"token\":\"a\",\"cost\":0.2}\n"
            ),
        );
        let duplicate_config = directory.write(
            "duplicate.json",
            r#"{
                "version": 1,
                "type": "custom",
                "rules": {"path": "duplicate.jsonl", "format": "jsonl"}
            }"#,
        );

        let invalid_token = load(&invalid_token_config, &WhitespaceTokenizer).unwrap_err();
        let duplicate = load(&duplicate_config, &WhitespaceTokenizer).unwrap_err();

        assert!(format!("{invalid_token:#}").contains("line 1"));
        assert!(format!("{invalid_token:#}").contains("exactly one complete token"));
        assert!(format!("{duplicate:#}").contains("duplicate insertion"));
        assert!(format!("{duplicate:#}").contains("line 2"));
    }

    #[test]
    fn rejects_negative_costs_and_operation_specific_extra_fields() {
        let directory = TestDirectory::new();
        directory.write("rules.jsonl", "");
        let negative_config = directory.write(
            "negative.json",
            r#"{
                "version": 1,
                "type": "custom",
                "defaults": {"deletion": -0.1},
                "rules": {"path": "rules.jsonl", "format": "jsonl"}
            }"#,
        );
        directory.write(
            "extra-field.jsonl",
            "{\"operation\":\"deletion\",\"token\":\"a\",\"from\":\"b\",\"cost\":0.5}\n",
        );
        let extra_field_config = directory.write(
            "extra-field.json",
            r#"{
                "version": 1,
                "type": "custom",
                "rules": {"path": "extra-field.jsonl", "format": "jsonl"}
            }"#,
        );

        let negative = load(&negative_config, &CharacterTokenizer).unwrap_err();
        let extra_field = load(&extra_field_config, &CharacterTokenizer).unwrap_err();

        assert!(format!("{negative:#}").contains("defaults.deletion"));
        assert!(format!("{extra_field:#}").contains("unknown field"));
        assert!(format!("{extra_field:#}").contains("line 1"));
    }
}
