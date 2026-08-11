"""CLI for converting word2vec text embeddings to Yurine JSON Lines."""

from __future__ import annotations

import sys
from typing import Literal

from tap import Positional, Tap

from yurine_tools.arguments import Normalization, require_nonnegative_finite, run
from yurine_tools.embeddings import (
    convert_word2vec_text,
    open_text_input,
    open_text_output,
    write_cost_config,
)


class ConvertEmbeddingsArgs(Tap):
    """Typed command-line arguments for embedding conversion."""

    input: Positional[str]  # word2vec text file, compressed file, or '-' for stdin
    output: Positional[str]  # JSON Lines file, compressed file, or '-' for stdout
    header: Literal["auto", "present", "absent"] = "auto"  # word2vec header handling
    normalization: Normalization = "none"  # token normalization
    cost_config: str | None = None  # also write a Yurine embedding cost config
    missing_substitution_cost: float = 1.0  # cost when either token has no embedding
    deletion_cost: float = 1.0  # deletion cost
    insertion_cost: float = 1.0  # insertion cost

    def configure(self) -> None:
        """Preserve the public hyphenated names for multiword options."""
        self.add_argument(
            "--cost-config",
            type=str,
            default=None,
            help="also write a Yurine embedding cost config",
        )
        self.add_argument(
            "--missing-substitution-cost",
            type=float,
            default=1.0,
            help="cost when either token has no embedding",
        )
        self.add_argument("--deletion-cost", type=float, default=1.0, help="deletion cost")
        self.add_argument("--insertion-cost", type=float, default=1.0, help="insertion cost")

    def process_args(self) -> None:
        """Validate costs before opening a potentially large embedding file."""
        require_nonnegative_finite(self.missing_substitution_cost, "missing substitution cost")
        require_nonnegative_finite(self.deletion_cost, "deletion cost")
        require_nonnegative_finite(self.insertion_cost, "insertion cost")


def run_command(args: ConvertEmbeddingsArgs) -> None:
    """Convert embeddings and optionally emit the matching cost configuration."""
    with open_text_input(args.input) as source, open_text_output(args.output) as destination:
        stats = convert_word2vec_text(
            source, destination, header=args.header, normalization=args.normalization
        )
    if args.cost_config:
        write_cost_config(
            args.cost_config,
            args.output,
            missing_substitution_cost=args.missing_substitution_cost,
            deletion_cost=args.deletion_cost,
            insertion_cost=args.insertion_cost,
        )
    print(
        f"converted {stats.records} embeddings with {stats.dimension} dimensions",
        file=sys.stderr,
    )


def main() -> None:
    """Run the embedding conversion command."""
    run(
        ConvertEmbeddingsArgs(
            prog="yurine-convert-embeddings",
            description="Convert word2vec text embeddings to Yurine JSON Lines.",
        ),
        run_command,
    )


if __name__ == "__main__":
    main()
