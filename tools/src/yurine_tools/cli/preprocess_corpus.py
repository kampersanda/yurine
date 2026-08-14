"""CLI for normalizing and tokenizing corpora and search queries."""

from __future__ import annotations

import sys
from typing import Literal

from tap import Positional, Tap

from yurine_tools.cli.common import require_distinct_file_paths, run
from yurine_tools.corpus import TokenizerConfig, preprocess_blocks
from yurine_tools.options import Normalization
from yurine_tools.text_io import open_text_input, open_text_output

_PROGRESS_LINES = 100_000


class PreprocessCorpusArgs(Tap):
    """Typed command-line arguments for corpus preprocessing."""

    input: Positional[str]  # text file, compressed file, or '-' for stdin
    output: Positional[str]  # text file, compressed file, or '-' for stdout
    tokenizer: Literal["sudachi", "whitespace"] = "sudachi"  # tokenization strategy
    mode: Literal["A", "B", "C"] = "B"  # Sudachi split mode
    form: Literal["surface", "normalized", "dictionary"] = "normalized"  # token form
    sudachi_config: str | None = None  # custom Sudachi configuration
    normalization: Normalization = "none"  # input normalization
    workers: int = 1  # tokenization processes, or 0 for one per CPU

    def configure(self) -> None:
        """Preserve the public hyphenated name of the Sudachi config option."""
        self.add_argument(
            "--sudachi-config",
            type=str,
            default=None,
            help="path to a custom Sudachi configuration",
        )

    def process_args(self) -> None:
        """Reject Sudachi-only options when using generic tokenization."""
        require_distinct_file_paths(
            ("input", self.input, True),
            ("output", self.output, True),
        )
        if self.workers < 0:
            raise ValueError("--workers must not be negative")
        if self.tokenizer != "whitespace":
            return
        if self.mode != "B":
            raise ValueError("--mode only applies to the Sudachi tokenizer")
        if self.form != "normalized":
            raise ValueError("--form only applies to the Sudachi tokenizer")
        if self.sudachi_config is not None:
            raise ValueError("--sudachi-config only applies to the Sudachi tokenizer")


def run_command(args: PreprocessCorpusArgs) -> None:
    """Normalize and tokenize every input line while preserving line count."""
    config = TokenizerConfig(
        kind=args.tokenizer,
        mode=args.mode,
        form=args.form,
        config_path=args.sudachi_config,
    )
    count = 0
    next_report = _PROGRESS_LINES
    with open_text_input(args.input) as source, open_text_output(args.output) as destination:
        blocks = preprocess_blocks(
            source, config, normalization=args.normalization, workers=args.workers
        )
        for block in blocks:
            destination.write(block)
            # Every output line is newline-terminated and tokens never contain
            # a newline, so this counts the lines the block carries.
            count += block.count("\n")
            if count >= next_report:
                print(f"processed {count} lines", file=sys.stderr)
                next_report += _PROGRESS_LINES
    print(f"processed {count} lines", file=sys.stderr)


def main() -> None:
    """Run the corpus preprocessing command."""
    run(
        PreprocessCorpusArgs(
            prog="yurine-preprocess-corpus",
            description="Normalize and tokenize a line-oriented corpus or query.",
        ),
        run_command,
    )


if __name__ == "__main__":
    main()
