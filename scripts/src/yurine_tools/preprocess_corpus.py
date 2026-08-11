"""CLI for normalizing and tokenizing corpora and search queries."""

from __future__ import annotations

import sys
from typing import Literal

from tap import Positional, Tap

from yurine_tools.arguments import Normalization, run
from yurine_tools.corpus import SudachiTokenizer, WhitespaceTokenizer, preprocess_lines
from yurine_tools.embeddings import open_text_input, open_text_output


class PreprocessCorpusArgs(Tap):
    """Typed command-line arguments for corpus preprocessing."""

    input: Positional[str]  # text file, compressed file, or '-' for stdin
    output: Positional[str]  # text file, compressed file, or '-' for stdout
    tokenizer: Literal["sudachi", "whitespace"] = "sudachi"  # tokenization strategy
    mode: Literal["A", "B", "C"] = "B"  # Sudachi split mode
    form: Literal["surface", "normalized", "dictionary"] = "normalized"  # token form
    sudachi_config: str | None = None  # custom Sudachi configuration
    normalization: Normalization = "none"  # input normalization

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
        if self.tokenizer == "whitespace" and self.form != "normalized":
            raise ValueError("--form only applies to the Sudachi tokenizer")


def run_command(args: PreprocessCorpusArgs) -> None:
    """Normalize and tokenize every input line while preserving line count."""
    if args.tokenizer == "whitespace":
        tokenizer = WhitespaceTokenizer()
    else:
        tokenizer = SudachiTokenizer(
            mode=args.mode, form=args.form, config_path=args.sudachi_config
        )

    count = 0
    with open_text_input(args.input) as source, open_text_output(args.output) as destination:
        for line in preprocess_lines(source, tokenizer, normalization=args.normalization):
            destination.write(line)
            destination.write("\n")
            count += 1
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
