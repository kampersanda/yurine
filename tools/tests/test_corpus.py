from yurine_tools.corpus import (
    SudachiTokenizer,
    WhitespaceTokenizer,
    normalize_text,
    preprocess_lines,
)


def test_preserves_line_structure_and_normalizes_whitespace() -> None:
    lines = ["new   york\n", "\n", "京都  市\r\n"]

    result = list(preprocess_lines(lines, WhitespaceTokenizer()))

    assert result == ["new york", "", "京都 市"]


def test_supports_unicode_normalization() -> None:
    assert normalize_text("ＡＢＣ", "nfkc-casefold") == "abc"
    assert normalize_text("𝐀", "nfkc-casefold") == "a"


def test_sudachi_normalized_form_matches_chive_preprocessing() -> None:
    tokenizer = SudachiTokenizer(mode="B", form="normalized")

    result = list(preprocess_lines(["附属の空罐\n"], tokenizer))

    assert result == ["付属 の 空き缶"]
