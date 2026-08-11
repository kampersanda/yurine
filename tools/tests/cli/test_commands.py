import pytest

from yurine_tools.cli.convert_embeddings import ConvertEmbeddingsArgs
from yurine_tools.cli.preprocess_corpus import PreprocessCorpusArgs


def test_chive_compatible_corpus_defaults() -> None:
    args = PreprocessCorpusArgs().parse_args(["input.txt", "output.txt"])

    assert args.tokenizer == "sudachi"
    assert args.mode == "B"
    assert args.form == "normalized"
    assert args.sudachi_config is None


def test_embedding_conversion_defaults_are_generic() -> None:
    args = ConvertEmbeddingsArgs().parse_args(["input.vec", "output.jsonl"])

    assert args.header == "auto"
    assert args.normalization == "none"


def test_embedding_cost_options_are_typed_and_validated() -> None:
    args = ConvertEmbeddingsArgs().parse_args(
        ["input.vec", "output.jsonl", "--missing-substitution-cost", "0.25"]
    )

    assert args.missing_substitution_cost == 0.25
    assert isinstance(args.missing_substitution_cost, float)

    with pytest.raises(ValueError, match="non-negative finite"):
        ConvertEmbeddingsArgs().parse_args(["input.vec", "output.jsonl", "--deletion-cost", "-1"])
