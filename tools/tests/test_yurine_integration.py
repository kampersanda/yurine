import csv
import subprocess
from io import StringIO
from pathlib import Path

import pytest

from yurine_tools.embeddings import convert_word2vec_text, write_cost_config


def test_generated_files_work_with_yurine_cli(tmp_path: Path) -> None:
    embeddings = tmp_path / "embeddings.jsonl"
    with embeddings.open("w", encoding="utf-8") as destination:
        convert_word2vec_text(StringIO("3 2\nx 1 0\nあ 0.8 0.6\nb 0 1\n"), destination)
    config = tmp_path / "costs.json"
    write_cost_config(
        str(config),
        str(embeddings),
        missing_substitution_cost=1.0,
        deletion_cost=1.0,
        insertion_cost=1.0,
    )
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("あ b\n", encoding="utf-8")
    index = tmp_path / "index"
    repository = Path(__file__).parents[2]

    def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--manifest-path",
                str(repository / "Cargo.toml"),
                "-p",
                "yurine-cli",
                "--",
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    run_cli("index", "--tokenizer", "whitespace", str(index), str(corpus))
    result = run_cli(
        "search",
        "--costs",
        str(config),
        "--threshold",
        "0.25",
        str(index),
        "x",
    )

    row = next(csv.reader(StringIO(result.stdout), delimiter="\t"))
    assert row[0] == "0"
    assert float(row[1]) == pytest.approx(0.2)
    assert row[2:] == ["0", "3", "あ"]
