"""Validation and error handling shared by command-line entry points."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from tap import Tap

from yurine_tools.schemas import F32_MAX

Arguments = TypeVar("Arguments", bound=Tap)


def require_nonnegative_finite(value: float, name: str) -> None:
    """Reject values that Yurine cannot represent as edit costs."""
    if not math.isfinite(value) or value < 0 or value > F32_MAX:
        raise ValueError(f"{name} must be a non-negative finite f32 value")


def require_distinct_file_paths(*paths: tuple[str, str, bool]) -> None:
    """Reject aliases before an output file can truncate an input file.

    Each tuple contains a display name, a path, and whether ``-`` denotes a
    standard stream rather than a filesystem path.
    """
    seen: dict[Path, str] = {}
    for name, value, supports_standard_stream in paths:
        if supports_standard_stream and value == "-":
            continue
        resolved = Path(value).resolve()
        if previous := seen.get(resolved):
            raise ValueError(f"{name} path must differ from {previous} path")
        seen[resolved] = name


def run(parser: Arguments, command: Callable[[Arguments], None]) -> None:
    """Parse arguments and present expected data errors without a traceback."""
    try:
        command(parser.parse_args())
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
