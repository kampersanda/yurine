import gzip
import json
from pathlib import Path

import pytest

from yurine_tools.cli import fetch_jawiki
from yurine_tools.cli.convert_embeddings import ConvertEmbeddingsArgs
from yurine_tools.cli.fetch_jawiki import FetchJawikiArgs
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

    with pytest.raises(ValueError, match="non-negative finite"):
        ConvertEmbeddingsArgs().parse_args(
            ["input.vec", "output.jsonl", "--deletion-cost", "1e100"]
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["input.vec", "input.vec"],
        ["input.vec", "output.jsonl", "--cost-config", "input.vec"],
        ["input.vec", "output.jsonl", "--cost-config", "output.jsonl"],
    ],
)
def test_embedding_conversion_rejects_colliding_paths(arguments: list[str]) -> None:
    with pytest.raises(ValueError, match="path must differ"):
        ConvertEmbeddingsArgs().parse_args(arguments)


def test_corpus_preprocessing_rejects_colliding_paths() -> None:
    with pytest.raises(ValueError, match="path must differ"):
        PreprocessCorpusArgs().parse_args(["corpus.txt", "corpus.txt"])


def test_standard_input_and_output_are_distinct_streams() -> None:
    assert ConvertEmbeddingsArgs().parse_args(["-", "-"]).input == "-"
    assert PreprocessCorpusArgs().parse_args(["-", "-"]).output == "-"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--mode", "A"),
        ("--form", "surface"),
        ("--sudachi-config", "sudachi.json"),
    ],
)
def test_whitespace_tokenizer_rejects_sudachi_options(option: str, value: str) -> None:
    with pytest.raises(ValueError, match="only applies to the Sudachi tokenizer"):
        PreprocessCorpusArgs().parse_args(
            ["input.txt", "output.txt", "--tokenizer", "whitespace", option, value]
        )


def test_jawiki_defaults_to_the_latest_c400_passages() -> None:
    args = FetchJawikiArgs().parse_args(["corpus.txt"])

    assert args.dataset == "passages-c400"
    assert args.dump == "20240401"
    assert args.limit is None
    assert args.cache is None


def test_jawiki_rejects_a_nonpositive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        FetchJawikiArgs().parse_args(["corpus.txt", "--limit", "0"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["corpus.txt", "--metadata", "corpus.txt"],
        ["corpus.txt", "--cache", "corpus.txt"],
        ["corpus.txt", "--metadata", "shared.jsonl", "--cache", "shared.jsonl"],
    ],
)
def test_jawiki_rejects_colliding_paths(arguments: list[str]) -> None:
    with pytest.raises(ValueError, match="path must differ"):
        FetchJawikiArgs().parse_args(arguments)


@pytest.mark.parametrize("option", ["--metadata", "--cache"])
def test_jawiki_keeps_standard_output_for_the_corpus(option: str) -> None:
    with pytest.raises(ValueError, match="not standard output"):
        FetchJawikiArgs().parse_args(["-", option, "-"])


def test_jawiki_writes_a_corpus_and_aligned_metadata(tmp_path: Path, monkeypatch) -> None:
    records = [
        {
            "id": index,
            "pageid": 5,
            "revid": 99347164,
            "title": "アンパサンド",
            "section": "__LEAD__",
            "text": text,
        }
        for index, text in enumerate(["前半。\n後半。", "二番目", "三番目"], start=1)
    ]
    source = tmp_path / "source.json.gz"
    with gzip.open(source, "wt", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    monkeypatch.setattr(fetch_jawiki, "release_url", lambda *_: source.resolve().as_uri())
    corpus = tmp_path / "corpus.txt"
    metadata = tmp_path / "metadata.jsonl"

    fetch_jawiki.run_command(
        FetchJawikiArgs().parse_args([str(corpus), "--limit", "2", "--metadata", str(metadata)])
    )

    assert corpus.read_text(encoding="utf-8") == "前半。 後半。\n二番目\n"
    lines = metadata.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["line"] for line in lines] == [1, 2]
    assert json.loads(lines[0])["title"] == "アンパサンド"


def test_path_aliases_are_rejected(tmp_path) -> None:
    source = tmp_path / "source.vec"
    alias = tmp_path / "alias.vec"
    source.touch()
    alias.symlink_to(source)

    with pytest.raises(ValueError, match="path must differ"):
        ConvertEmbeddingsArgs().parse_args([str(source), str(alias)])
