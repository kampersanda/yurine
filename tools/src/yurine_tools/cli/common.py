"""Validation and error handling shared by command-line entry points."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypeVar

from tap import Tap

Arguments = TypeVar("Arguments", bound=Tap)


def require_nonnegative_finite(value: float, name: str) -> None:
    """Reject values that Yurine cannot represent as edit costs."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")


def run(parser: Arguments, command: Callable[[Arguments], None]) -> None:
    """Parse arguments and present expected data errors without a traceback."""
    try:
        command(parser.parse_args())
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
