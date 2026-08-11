"""Normalization and tokenization primitives for line-oriented corpora."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Iterator
from typing import Protocol

from yurine_tools.options import Normalization


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
        self._form = form

    def tokenize(self, text: str) -> Iterable[str]:
        """Yield the configured representation of each Sudachi morpheme."""
        for morpheme in self._tokenizer.tokenize(text, self._mode):
            if self._form == "surface":
                token = morpheme.surface()
            elif self._form == "dictionary":
                token = morpheme.dictionary_form()
            else:
                token = morpheme.normalized_form()
            # Sudachi can emit a morpheme for source whitespace; Yurine uses
            # whitespace exclusively as a token separator.
            if token and not token.isspace():
                yield token


def normalize_text(text: str, normalization: Normalization) -> str:
    """Apply the explicitly selected Unicode normalization policy."""
    if normalization == "none":
        return text
    if normalization == "nfc":
        return unicodedata.normalize("NFC", text)
    if normalization == "nfkc":
        return unicodedata.normalize("NFKC", text)
    if normalization == "nfkc-casefold":
        return unicodedata.normalize("NFKC", text.casefold())
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
