"""Normalization and tokenization primitives for line-oriented corpora."""

from __future__ import annotations

import multiprocessing
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Protocol

from yurine_tools.options import Normalization

# Lines per unit of work. Large enough to amortize inter-process transfer,
# small enough to keep the in-flight lines of a huge corpus bounded.
CHUNK_SIZE = 1000

_MORPHEME_FORMS = {
    "surface": "surface",
    "dictionary": "dictionary_form",
    "normalized": "normalized_form",
}


class Tokenizer(Protocol):
    """Minimal tokenizer interface used by the preprocessing pipeline."""

    def tokenize(self, text: str) -> Iterable[str]: ...


class WhitespaceTokenizer:
    """Reuse tokens from an already whitespace-delimited corpus."""

    def tokenize(self, text: str) -> Iterable[str]:
        """Split on Unicode whitespace and discard empty fields."""
        return text.split()


class SudachiTokenizer:
    """Adapt Sudachi split modes and token forms to the common interface."""

    def __init__(self, *, mode: str, form: str, config_path: str | None = None) -> None:
        from sudachipy import dictionary, tokenizer

        self._tokenizer = dictionary.Dictionary(config_path=config_path).create()
        self._mode = getattr(tokenizer.Tokenizer.SplitMode, mode)
        self._form = _MORPHEME_FORMS[form]

    def tokenize(self, text: str) -> Iterable[str]:
        """Yield the configured representation of each Sudachi morpheme."""
        form = self._form
        for morpheme in self._tokenizer.tokenize(text, self._mode):
            token = getattr(morpheme, form)()
            # Sudachi can emit a morpheme for source whitespace; Yurine uses
            # whitespace exclusively as a token separator.
            if token and not token.isspace():
                yield token


@dataclass(frozen=True)
class TokenizerConfig:
    """Picklable description of a tokenizer, so workers can build their own."""

    kind: str
    mode: str = "B"
    form: str = "normalized"
    config_path: str | None = None

    def build(self) -> Tokenizer:
        """Instantiate the described tokenizer."""
        if self.kind == "whitespace":
            return WhitespaceTokenizer()
        return SudachiTokenizer(mode=self.mode, form=self.form, config_path=self.config_path)


def normalize_text(text: str, normalization: Normalization) -> str:
    """Apply the explicitly selected Unicode normalization policy."""
    if normalization == "none":
        return text
    if normalization == "nfc":
        return unicodedata.normalize("NFC", text)
    if normalization == "nfkc":
        return unicodedata.normalize("NFKC", text)
    if normalization == "nfkc-casefold":
        # Normalize compatibility characters before case folding, then apply
        # NFKC once more for the canonical NFKC_Casefold operation.
        normalized = unicodedata.normalize("NFKC", text)
        return unicodedata.normalize("NFKC", normalized.casefold())
    raise AssertionError(f"unknown normalization: {normalization}")


def preprocess_lines(
    lines: Iterable[str],
    tokenizer: Tokenizer,
    *,
    normalization: Normalization = "none",
) -> Iterator[str]:
    """Produce one whitespace-tokenized output line for every input line."""
    for line in lines:
        # Removing only line endings preserves other source whitespace until
        # the selected tokenizer decides how to handle it.
        text = normalize_text(line.rstrip("\r\n"), normalization)
        yield " ".join(tokenizer.tokenize(text))


def _chunked(lines: Iterable[str], size: int) -> Iterator[list[str]]:
    """Slice a line stream into fixed-size chunks without materializing it."""
    iterator = iter(lines)
    while chunk := list(islice(iterator, size)):
        yield chunk


def _preprocess_chunk(chunk: list[str], tokenizer: Tokenizer, normalization: Normalization) -> str:
    """Preprocess one chunk into a single newline-terminated text block."""
    return "".join(
        f"{line}\n" for line in preprocess_lines(chunk, tokenizer, normalization=normalization)
    )


_worker_state: tuple[Tokenizer, Normalization]


def _start_worker(config: TokenizerConfig, normalization: Normalization) -> None:
    """Build the per-worker tokenizer once instead of once per chunk."""
    global _worker_state
    _worker_state = (config.build(), normalization)


def _preprocess_chunk_in_worker(chunk: list[str]) -> str:
    """Preprocess a chunk with the tokenizer this worker already loaded."""
    return _preprocess_chunk(chunk, *_worker_state)


def preprocess_blocks(
    lines: Iterable[str],
    config: TokenizerConfig,
    *,
    normalization: Normalization = "none",
    workers: int = 1,
) -> Iterator[str]:
    """Stream preprocessed text blocks, keeping input order and line count.

    Blocks concatenate ``CHUNK_SIZE`` output lines so that a huge corpus costs
    one write per chunk rather than one per line.
    """
    chunks = _chunked(lines, CHUNK_SIZE)
    if workers == 1:
        tokenizer = config.build()
        for chunk in chunks:
            yield _preprocess_chunk(chunk, tokenizer, normalization)
        return
    with multiprocessing.Pool(
        workers, initializer=_start_worker, initargs=(config, normalization)
    ) as pool:
        # Submitting a bounded batch at a time keeps every worker busy without
        # reading the whole corpus into memory.
        while batch := list(islice(chunks, 2 * workers)):
            yield from pool.map(_preprocess_chunk_in_worker, batch)
