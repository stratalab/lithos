"""Tests for lithos.data.filters — drop reasons and accounting (PRD §8.7)."""

from lithos.data.filters import DocumentFilter, FilterConfig, check_document


def _cfg(**over):
    base = dict(
        min_chars=5,
        max_chars=100,
        max_repeated_char_run=5,
        max_duplicate_line_fraction=0.5,
        max_symbol_fraction=0.5,
    )
    base.update(over)
    return FilterConfig(**base)


def test_empty_and_whitespace():
    assert check_document("", _cfg()) == "empty"
    assert check_document("   \n\t ", _cfg()) == "empty"


def test_length_bounds():
    assert check_document("abc", _cfg()) == "too_short"
    assert check_document("a" * 200, _cfg()) == "too_long"


def test_repeated_chars():
    assert check_document("aaaaaaaa hello world", _cfg()) == "repeated_chars"


def test_symbol_density():
    assert check_document("!!!!@@@@####$$$$", _cfg()) == "symbol_density"


def test_duplicate_lines():
    text = "same line\nsame line\nsame line\nunique line"  # unique 2 / 4 -> 0.5 dup
    assert check_document(text, _cfg(max_duplicate_line_fraction=0.4)) == "duplicate_lines"


def test_clean_document_passes():
    assert check_document("hello world, this is a fine sentence.", _cfg()) is None


def test_language_filter():
    cfg = FilterConfig(allowed_languages=["en"])
    f = DocumentFilter(cfg)
    assert f.keep({"text": "hello world", "language": "en"}) is True
    assert f.keep({"text": "bonjour le monde", "language": "fr"}) is False
    assert f.stats()["dropped"]["language"] == 1


def test_filter_tallies_stats():
    f = DocumentFilter(_cfg())
    assert f.keep({"text": "hello world"}) is True
    assert f.keep({"text": "ab"}) is False
    stats = f.stats()
    assert stats["kept"] == 1
    assert stats["dropped"]["too_short"] == 1
