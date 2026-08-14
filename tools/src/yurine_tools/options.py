"""Shared option types used by CLI and domain modules."""

from typing import Literal

Normalization = Literal["none", "nfc", "nfkc", "nfkc-casefold"]
Header = Literal["auto", "present", "absent"]
JawikiDataset = Literal["passages-c400", "passages-c300", "passages-para"]
JawikiDump = Literal["20240401", "20230403"]
