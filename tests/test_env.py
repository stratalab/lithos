"""Tests for lithos.utils.env — .env loading (no-override, export prefix, quotes)."""

import lithos.utils.env as env_mod
from lithos.utils.env import load_env


def test_load_env_parses_and_respects_existing(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        'export AWS_ACCESS_KEY_ID="abc123"\n'
        "AWS_SECRET_ACCESS_KEY=secret\n"
        "# a comment\n"
        "\n"
        "PRESET=fromfile\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PRESET", "fromenv")  # existing env must win
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(env_mod, "_LOADED", False)

    load_env()

    import os

    assert os.environ["AWS_ACCESS_KEY_ID"] == "abc123"  # quotes + export stripped
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert os.environ["PRESET"] == "fromenv"  # not overridden


def test_load_env_missing_file_is_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_mod, "_LOADED", False)
    load_env("definitely-not-here.env")  # should not raise
