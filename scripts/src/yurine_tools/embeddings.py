"""Streaming conversion utilities for static word embedding files."""

from __future__ import annotations

import bz2
import gzip
import json
import lzma
import math
import os
import sys
import unicodedata
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TextIO

from yurine_tools.arguments import Header, Normalization


@dataclass(frozen=True)
class ConversionStats:
    """Metadata reported after a successful embedding conversion."""

    records: int
    dimension: int


def open_text_input(path: str):
    """Open UTF-8 input, detecting supported compression from the suffix."""
    if path == "-":
        return nullcontext(sys.stdin)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8")
    if path.endswith((".xz", ".lzma")):
        return lzma.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def open_text_output(path: str):
    """Open UTF-8 output and create its parent directory when necessary."""
    if path == "-":
        return nullcontext(sys.stdout)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8", newline="\n")
    if path.endswith(".bz2"):
        return bz2.open(path, "wt", encoding="utf-8", newline="\n")
    if path.endswith((".xz", ".lzma")):
        return lzma.open(path, "wt", encoding="utf-8", newline="\n")
    return open(path, "w", encoding="utf-8", newline="\n")


def normalize_token(token: str, normalization: Normalization) -> str:
    """Normalize a model token without changing it by default."""
    if normalization == "none":
        return token
    if normalization == "nfc":
        return unicodedata.normalize("NFC", token)
    if normalization == "nfkc":
        return unicodedata.normalize("NFKC", token)
    if normalization == "nfkc-casefold":
        return unicodedata.normalize("NFKC", token.casefold())
    raise AssertionError(f"unknown normalization: {normalization}")


def convert_word2vec_text(
    source: TextIO,
    destination: TextIO,
    *,
    header: Header = "auto",
    normalization: Normalization = "none",
) -> ConversionStats:
    """Stream word2vec text records as Yurine-compatible JSON Lines.

    Header presence can be forced for ambiguous headerless one-dimensional
    models. All records are checked against the declared or inferred vector
    dimension before they are written.
    """
    lines = enumerate(source, start=1)
    try:
        first_line_number, first_line = next(lines)
    except StopIteration as error:
        raise ValueError("embedding file is empty") from error

    expected_count: int | None = None
    dimension: int | None = None
    first_parts = first_line.split()
    has_header = header == "present" or (
        header == "auto" and len(first_parts) == 2 and all(part.isdigit() for part in first_parts)
    )
    if has_header:
        if len(first_parts) != 2 or not all(part.isdigit() for part in first_parts):
            raise ValueError("line 1: expected a word2vec header with record and dimension counts")
        expected_count, dimension = map(int, first_parts)
        if expected_count == 0 or dimension == 0:
            raise ValueError("word2vec header values must be positive")
        pending = []
    else:
        # The first line is a vector record when no header was detected, so it
        # must be replayed before consuming the remaining iterator.
        pending = [(first_line_number, first_parts)]

    records = 0
    for line_number, parts in chain(pending, lines):
        if not isinstance(parts, list):
            parts = parts.split()
        if len(parts) < 2:
            raise ValueError(f"line {line_number}: expected a token and vector")

        token = normalize_token(parts[0], normalization)
        if not token or any(character.isspace() for character in token):
            raise ValueError(f"line {line_number}: token must be non-empty without whitespace")

        try:
            embedding = [float(value) for value in parts[1:]]
        except ValueError as error:
            raise ValueError(f"line {line_number}: vector contains a non-number") from error

        if dimension is None:
            dimension = len(embedding)
        if len(embedding) != dimension:
            raise ValueError(
                f"line {line_number}: expected {dimension} dimensions, got {len(embedding)}"
            )
        if not all(math.isfinite(value) for value in embedding):
            raise ValueError(f"line {line_number}: vector contains a non-finite value")
        if not any(value != 0.0 for value in embedding):
            raise ValueError(f"line {line_number}: vector has zero norm")

        json.dump(
            {"token": token, "embedding": embedding},
            destination,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        destination.write("\n")
        records += 1

    if dimension is None:
        raise ValueError("embedding file has a header but no records")
    if expected_count is not None and records != expected_count:
        raise ValueError(f"header declares {expected_count} records, but found {records}")
    return ConversionStats(records=records, dimension=dimension)


def write_cost_config(
    path: str,
    embeddings_path: str,
    *,
    missing_substitution_cost: float,
    deletion_cost: float,
    insertion_cost: float,
) -> None:
    """Write a Yurine cost configuration referring to converted embeddings."""
    if embeddings_path == "-":
        raise ValueError("cannot create a cost config for embeddings written to standard output")
    if embeddings_path.endswith((".gz", ".bz2", ".xz", ".lzma")):
        raise ValueError("Yurine requires an uncompressed embedding file in a cost config")

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # Yurine resolves embedding paths from the configuration directory, not
    # from the process working directory.
    relative_embeddings = os.path.relpath(
        Path(embeddings_path).resolve(), start=config_path.parent.resolve()
    )
    config = {
        "version": 1,
        "type": "embedding",
        "embeddings": {"path": relative_embeddings, "format": "jsonl"},
        "missing_substitution_cost": missing_substitution_cost,
        "deletion_cost": deletion_cost,
        "insertion_cost": insertion_cost,
    }
    with config_path.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(config, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
