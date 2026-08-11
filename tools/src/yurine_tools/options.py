"""Shared option types used by CLI and domain modules."""

from typing import Literal

Normalization = Literal["none", "nfc", "nfkc", "nfkc-casefold"]
Header = Literal["auto", "present", "absent"]
