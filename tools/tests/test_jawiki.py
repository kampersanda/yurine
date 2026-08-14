import gzip
import json
from pathlib import Path
from typing import get_args

import pytest

from yurine_tools.jawiki import (
    RELEASE_TAGS,
    download_archive,
    flatten_text,
    open_archive,
    read_passages,
    release_url,
)
from yurine_tools.options import JawikiDump


def passage(identifier: int, text: str, **overrides) -> dict:
    record = {
        "id": identifier,
        "pageid": 5,
        "revid": 99347164,
        "title": "アンパサンド",
        "section": "__LEAD__",
        "text": text,
    }
    return record | overrides


def json_lines(records: list[dict]) -> list[str]:
    return [json.dumps(record, ensure_ascii=False) + "\n" for record in records]


def write_archive(path: Path, records: list[dict]) -> str:
    with gzip.open(path, "wt", encoding="utf-8") as destination:
        destination.writelines(json_lines(records))
    return path.resolve().as_uri()


def test_release_url_names_the_published_asset() -> None:
    assert release_url("passages-c400", "20240401") == (
        "https://github.com/singletongue/wikipedia-utils/releases/download"
        "/2024-04-01/passages-c400-jawiki-20240401.json.gz"
    )


def test_every_selectable_dump_has_a_release_tag() -> None:
    assert set(get_args(JawikiDump)) == set(RELEASE_TAGS)


@pytest.mark.parametrize(
    ("dataset", "dump"),
    [("paragraphs", "20240401"), ("passages-c400", "20250401")],
)
def test_release_url_rejects_unpublished_combinations(dataset: str, dump: str) -> None:
    with pytest.raises(ValueError, match="unknown"):
        release_url(dataset, dump)


def test_passages_become_one_line_each() -> None:
    records = json_lines([passage(1, "前半。\n後半。"), passage(2, "京都  市")])

    texts = [found.text for found in read_passages(records)]

    assert texts == ["前半。 後半。", "京都 市"]


def test_blank_and_empty_passages_do_not_produce_lines() -> None:
    records = json_lines([passage(1, "  "), passage(2, "本文")])
    records.insert(1, "\n")

    found = list(read_passages(records))

    assert [record.id for record in found] == [2]


def test_limit_counts_written_passages() -> None:
    records = json_lines([passage(1, " "), passage(2, "一"), passage(3, "二")])

    assert [record.id for record in read_passages(records, limit=1)] == [2]
    assert list(read_passages(records, limit=0)) == []


def test_malformed_archives_name_the_line() -> None:
    with pytest.raises(ValueError, match="line 2: archive is not JSON Lines"):
        list(read_passages([*json_lines([passage(1, "本文")]), "{\n"]))

    with pytest.raises(ValueError, match="line 1: not a passage record"):
        list(read_passages(json_lines([{"text": "本文"}])))


def test_titles_are_flattened_for_the_metadata_sidecar() -> None:
    records = json_lines([passage(1, "本文", title="日本\t語", section=" 概要 ")])

    found = next(iter(read_passages(records)))

    assert (found.title, found.section) == ("日本 語", "概要")


def test_flatten_text_preserves_interior_words() -> None:
    assert flatten_text(" a \r\n b\tc ") == "a b c"


def test_archives_stream_without_a_cache(tmp_path: Path) -> None:
    url = write_archive(tmp_path / "source.json.gz", [passage(1, "本文")])

    with open_archive(url, None, report=lambda _: None) as archive:
        assert [record.id for record in read_passages(archive)] == [1]


def test_cached_archives_are_downloaded_once(tmp_path: Path) -> None:
    url = write_archive(tmp_path / "source.json.gz", [passage(1, "本文")])
    cache = tmp_path / "cache" / "passages.json.gz"

    with open_archive(url, str(cache), report=lambda _: None) as archive:
        assert list(read_passages(archive))
    assert cache.exists()

    cache.write_bytes(gzip.compress(json_lines([passage(7, "別の本文")])[0].encode()))
    with open_archive(url, str(cache), report=lambda _: None) as archive:
        assert [record.id for record in read_passages(archive)] == [7]


def test_downloads_leave_no_partial_file(tmp_path: Path) -> None:
    url = write_archive(tmp_path / "source.json.gz", [passage(1, "本文")])
    cache = tmp_path / "passages.json.gz"

    size = download_archive(url, cache, report=lambda _: None)

    assert size == cache.stat().st_size
    assert not cache.with_name(f"{cache.name}.part").exists()
