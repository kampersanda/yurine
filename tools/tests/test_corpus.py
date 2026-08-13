from yurine_tools.corpus import (
    CHUNK_SIZE,
    SudachiTokenizer,
    TokenizerConfig,
    WhitespaceTokenizer,
    normalize_text,
    preprocess_blocks,
    preprocess_lines,
)


def test_preserves_line_structure_and_normalizes_whitespace() -> None:
    lines = ["new   york\n", "\n", "京都  市\r\n"]

    result = list(preprocess_lines(lines, WhitespaceTokenizer()))

    assert result == ["new york", "", "京都 市"]


def test_supports_unicode_normalization() -> None:
    assert normalize_text("ＡＢＣ", "nfkc-casefold") == "abc"
    assert normalize_text("𝐀", "nfkc-casefold") == "a"


def test_worker_processes_reproduce_the_sequential_output() -> None:
    # Spanning several chunks exercises the batching that feeds the workers.
    lines = [f"line   {index} ＡＢ\n" for index in range(2 * CHUNK_SIZE + 1)]
    config = TokenizerConfig(kind="whitespace")

    sequential = "".join(preprocess_blocks(lines, config, normalization="nfkc-casefold"))
    parallel = "".join(preprocess_blocks(lines, config, normalization="nfkc-casefold", workers=2))

    assert parallel == sequential
    assert sequential.splitlines() == [f"line {index} ab" for index in range(2 * CHUNK_SIZE + 1)]


def test_zero_workers_runs_one_worker_per_cpu() -> None:
    lines = ["ＡＢ  c\n", "d\n"]
    config = TokenizerConfig(kind="whitespace")

    blocks = list(preprocess_blocks(lines, config, normalization="nfkc", workers=0))

    assert blocks == ["AB c\nd\n"]


def test_blocks_terminate_every_line_including_the_last() -> None:
    blocks = list(preprocess_blocks(["a b", ""], TokenizerConfig(kind="whitespace")))

    assert blocks == ["a b\n\n"]


def test_sudachi_normalized_form_matches_chive_preprocessing() -> None:
    tokenizer = SudachiTokenizer(mode="B", form="normalized")

    result = list(preprocess_lines(["附属の空罐\n"], tokenizer))

    assert result == ["付属 の 空き缶"]
