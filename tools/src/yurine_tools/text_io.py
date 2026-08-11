"""Text input and output helpers shared by data preparation commands."""

import bz2
import gzip
import lzma
import sys
from contextlib import nullcontext
from pathlib import Path


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
