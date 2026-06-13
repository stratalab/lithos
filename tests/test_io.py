"""Tests for lithos.utils.io — hashing, atomic writes, JSON/YAML, no-clobber."""

import pytest
from lithos.utils import io

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_bytes_known_value():
    assert io.sha256_bytes(b"") == EMPTY_SHA256


def test_atomic_write_creates_parents_and_roundtrips(tmp_path):
    p = tmp_path / "nested" / "file.txt"
    io.atomic_write_text(p, "hello")
    assert p.read_text() == "hello"


def test_sha256_file_matches_bytes(tmp_path):
    p = tmp_path / "f.bin"
    data = b"lithos"
    io.atomic_write_bytes(p, data)
    assert io.sha256_file(p) == io.sha256_bytes(data)


def test_json_roundtrip(tmp_path):
    p = tmp_path / "x.json"
    obj = {"a": 1, "b": [1, 2, 3], "c": "text"}
    io.write_json(p, obj)
    assert io.read_json(p) == obj


def test_yaml_roundtrip_preserves_key_order(tmp_path):
    p = tmp_path / "x.yaml"
    obj = {"model": {"n_layers": 4}, "lr": 0.001}
    io.write_yaml(p, obj)
    assert io.read_yaml(p) == obj


def test_ensure_new_dir_refuses_clobber(tmp_path):
    d = tmp_path / "run"
    io.ensure_new_dir(d)
    with pytest.raises(FileExistsError):
        io.ensure_new_dir(d)
    # Opt-in reuse succeeds.
    assert io.ensure_new_dir(d, allow_existing=True) == d
