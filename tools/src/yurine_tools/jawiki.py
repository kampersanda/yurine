"""Streaming download and extraction of Wikipedia-Utils passage releases."""

from __future__ import annotations

import gzip
import json
import re
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO, get_args

from yurine_tools.options import JawikiDataset
from yurine_tools.schemas import JawikiPassage

# The Wikipedia-Utils releases publish one gzipped JSON Lines file per dataset
# and dump, so a passage corpus needs no Parquet or dataset-library support.
RELEASE_TAGS = {"20240401": "2024-04-01", "20230403": "2023-04-03"}
PASSAGE_DATASETS = get_args(JawikiDataset)

_RELEASE_URL = "https://github.com/singletongue/wikipedia-utils/releases/download/{tag}/{name}"
_USER_AGENT = "yurine-tools"
_CHUNK_BYTES = 1 << 20
_PROGRESS_BYTES = 100 * _CHUNK_BYTES
_WHITESPACE = re.compile(r"\s+")


def release_url(dataset: str, dump: str) -> str:
    """Locate the release asset for a passage dataset and Wikipedia dump."""
    if dataset not in PASSAGE_DATASETS:
        raise ValueError(f"unknown passage dataset: {dataset}")
    tag = RELEASE_TAGS.get(dump)
    if tag is None:
        raise ValueError(f"unknown jawiki dump: {dump}")
    return _RELEASE_URL.format(tag=tag, name=f"{dataset}-jawiki-{dump}.json.gz")


def _open_stream(url: str):
    """Request one release asset without following the default user agent."""
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": _USER_AGENT}))


def download_archive(url: str, path: Path, *, report: Callable[[str], None]) -> int:
    """Download an archive through a temporary file so caches stay complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    downloaded = 0
    next_report = _PROGRESS_BYTES
    try:
        with _open_stream(url) as response, partial.open("wb") as destination:
            while chunk := response.read(_CHUNK_BYTES):
                destination.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    report(f"downloaded {downloaded // _CHUNK_BYTES} MiB")
                    next_report += _PROGRESS_BYTES
    except BaseException:
        # An interrupted download of a multi-gigabyte archive would otherwise
        # leave its partial file behind for every retry.
        partial.unlink(missing_ok=True)
        raise
    partial.replace(path)
    return downloaded


@contextmanager
def open_archive(url: str, cache: str | None, *, report: Callable[[str], None]) -> Iterator[TextIO]:
    """Yield the JSON Lines of an archive, downloading it only when needed.

    Without a cache path the archive is decompressed as it arrives and never
    stored. With one, a missing archive is downloaded in full and then reused
    by later runs, including runs that ask for a different number of passages.
    """
    if cache is None:
        with _open_stream(url) as response, gzip.open(response, "rt", encoding="utf-8") as lines:
            yield lines
        return

    path = Path(cache)
    if not path.exists():
        size = download_archive(url, path, report=report)
        report(f"cached {size} bytes in {path}")
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        yield lines


def flatten_text(text: str) -> str:
    """Collapse whitespace so one passage always occupies one output line."""
    return _WHITESPACE.sub(" ", text).strip()


def read_passages(lines: Iterable[str], *, limit: int | None = None) -> Iterator[JawikiPassage]:
    """Validate passage records and flatten every text to a single line.

    Passages whose text is empty after flattening are skipped because Yurine
    indexes one token sequence per line, and they do not count towards the
    limit.
    """
    if limit is not None and limit <= 0:
        return

    produced = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {line_number}: archive is not JSON Lines") from error
        try:
            passage = JawikiPassage.model_validate(record)
        except ValueError as error:
            raise ValueError(f"line {line_number}: not a passage record") from error

        text = flatten_text(passage.text)
        if not text:
            continue

        yield passage.model_copy(
            update={
                "text": text,
                "title": flatten_text(passage.title),
                "section": flatten_text(passage.section),
            }
        )
        produced += 1
        if limit is not None and produced >= limit:
            return
