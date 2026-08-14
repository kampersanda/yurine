"""CLI for turning a Wikipedia-Utils passage release into a Yurine corpus."""

from __future__ import annotations

import sys
from contextlib import nullcontext

from tap import Positional, Tap

from yurine_tools.cli.common import require_distinct_file_paths, run
from yurine_tools.jawiki import open_archive, read_passages, release_url
from yurine_tools.options import JawikiDataset, JawikiDump
from yurine_tools.schemas import JawikiPassageMetadata
from yurine_tools.text_io import open_text_output


class FetchJawikiArgs(Tap):
    """Typed command-line arguments for jawiki passage retrieval."""

    output: Positional[str]  # text file, compressed file, or '-' for stdout
    dataset: JawikiDataset = "passages-c400"  # passage chunking of the release
    dump: JawikiDump = "20240401"  # Wikipedia dump date
    limit: int | None = None  # stop after this many passages
    metadata: str | None = None  # also write JSON Lines naming every output line
    cache: str | None = None  # keep the downloaded archive here and reuse it

    def process_args(self) -> None:
        """Validate the limit and the paths before any download starts."""
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")
        paths = [("output", self.output, True)]
        if self.metadata is not None:
            paths.append(("metadata", self.metadata, False))
        if self.cache is not None:
            paths.append(("cache", self.cache, False))
        require_distinct_file_paths(*paths)


def _report(message: str) -> None:
    """Send download progress to standard error, never to the corpus."""
    print(message, file=sys.stderr)


def run_command(args: FetchJawikiArgs) -> None:
    """Write one passage per line, ready for yurine-preprocess-corpus."""
    url = release_url(args.dataset, args.dump)
    metadata_output = (
        open_text_output(args.metadata) if args.metadata is not None else nullcontext(None)
    )

    count = 0
    with (
        open_archive(url, args.cache, report=_report) as archive,
        open_text_output(args.output) as destination,
        metadata_output as metadata,
    ):
        for passage in read_passages(archive, limit=args.limit):
            count += 1
            destination.write(passage.text)
            destination.write("\n")
            if metadata is not None:
                record = JawikiPassageMetadata(
                    line=count,
                    id=passage.id,
                    pageid=passage.pageid,
                    revid=passage.revid,
                    title=passage.title,
                    section=passage.section,
                )
                metadata.write(record.model_dump_json())
                metadata.write("\n")
    print(f"wrote {count} passages", file=sys.stderr)


def main() -> None:
    """Run the jawiki passage retrieval command."""
    run(
        FetchJawikiArgs(
            prog="yurine-fetch-jawiki",
            description="Download Japanese Wikipedia passages as a line-oriented corpus.",
        ),
        run_command,
    )


if __name__ == "__main__":
    main()
