import gzip
import json
from io import StringIO

import pytest
from pydantic import BaseModel

from yurine_tools.embeddings import (
    convert_word2vec_text,
    write_cost_config,
)
from yurine_tools.schemas import ConversionStats, EmbeddingCostConfig, EmbeddingRecord
from yurine_tools.text_io import open_text_input


def test_generated_data_uses_pydantic_schemas() -> None:
    assert issubclass(ConversionStats, BaseModel)
    assert issubclass(EmbeddingRecord, BaseModel)
    assert issubclass(EmbeddingCostConfig, BaseModel)


def test_converts_word2vec_text_with_header() -> None:
    source = StringIO("2 2\n東京 1.0 0.0\n京都 0.5 0.5\n")
    destination = StringIO()

    stats = convert_word2vec_text(source, destination)

    assert stats.records == 2
    assert stats.dimension == 2
    assert [json.loads(line) for line in destination.getvalue().splitlines()] == [
        {"token": "東京", "embedding": [1.0, 0.0]},
        {"token": "京都", "embedding": [0.5, 0.5]},
    ]


def test_infers_dimension_without_header_and_normalizes_tokens() -> None:
    source = StringIO("ＡＢＣ 1 2\n東京 3 4\n")
    destination = StringIO()

    stats = convert_word2vec_text(source, destination, normalization="nfkc-casefold")

    assert stats.dimension == 2
    assert json.loads(destination.getvalue().splitlines()[0])["token"] == "abc"


def test_can_force_headerless_input_when_first_token_is_numeric() -> None:
    destination = StringIO()

    stats = convert_word2vec_text(StringIO("123 300\n456 200\n"), destination, header="absent")

    assert stats.records == 2
    assert stats.dimension == 1
    assert json.loads(destination.getvalue().splitlines()[0]) == {
        "token": "123",
        "embedding": [300.0],
    }


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("2 2\n東京 1 0\n", "header declares 2 records"),
        ("東京 1 0\n京都 1\n", "expected 2 dimensions"),
        ("東京 0 0\n", "zero norm"),
        ("東京 nan 0\n", "non-finite"),
    ],
)
def test_rejects_invalid_embeddings(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        convert_word2vec_text(StringIO(source), StringIO())


def test_reads_gzip_input(tmp_path) -> None:
    path = tmp_path / "vectors.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as output:
        output.write("1 2\n東京 1 0\n")

    with open_text_input(str(path)) as source:
        assert source.read() == "1 2\n東京 1 0\n"


def test_writes_relative_embedding_path_in_cost_config(tmp_path) -> None:
    embeddings = tmp_path / "data" / "embeddings.jsonl"
    config = tmp_path / "config" / "costs.json"

    write_cost_config(
        str(config),
        str(embeddings),
        missing_substitution_cost=0.9,
        deletion_cost=0.8,
        insertion_cost=0.7,
    )

    contents = json.loads(config.read_text())
    assert contents == {
        "version": 1,
        "type": "embedding",
        "embeddings": {"path": "../data/embeddings.jsonl", "format": "jsonl"},
        "missing_substitution_cost": 0.9,
        "deletion_cost": 0.8,
        "insertion_cost": 0.7,
    }
