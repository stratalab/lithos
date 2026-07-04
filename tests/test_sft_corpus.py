"""Tests for the offline SFT-corpus build (lithos/posttrain/sft_corpus.py, E2).

Uses the real fineweb-edu-32k tokenizer artifact (present in the repo) so rendering
+ dtype selection match production; the build is tiny so this stays fast.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from lithos.posttrain.sft_corpus import (
    SFTCorpusBuildConfig,
    SFTShardWriter,
    SFTSourceSpec,
    build_sft_corpus,
)

TOKENIZER = "artifacts/tokenizer/fineweb-edu-32k/tokenizer.json"
pytestmark = pytest.mark.skipif(not Path(TOKENIZER).exists(), reason="tokenizer artifact absent")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _convo(u: str, a: str) -> dict:
    return {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}


# ---- SFTShardWriter ----


def test_shard_writer_dual_stream_roundtrip(tmp_path):
    w = SFTShardWriter(tmp_path, tokens_per_shard=10, dtype="uint16", tokenizer_name="t")
    w.add([1, 2, 3, 4], [False, False, True, True])
    w.add([5, 6, 7], [True, True, False])
    shards = w.close()
    assert len(shards) == 1
    s = shards[0]
    assert s["num_tokens"] == 7
    assert s["tokens_path"].endswith(".tokens.bin") and s["mask_path"].endswith(".mask.bin")
    toks = np.fromfile(tmp_path / Path(s["tokens_path"]).name, dtype="uint16")
    mask = np.fromfile(tmp_path / Path(s["mask_path"]).name, dtype="uint8")
    assert toks.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert mask.tolist() == [0, 0, 1, 1, 1, 1, 0]
    assert w.total_loss_tokens == 4


def test_shard_writer_flushes_full_shards(tmp_path):
    w = SFTShardWriter(tmp_path, tokens_per_shard=4, dtype="uint16", tokenizer_name="t")
    w.add(list(range(10)), [True] * 10)
    shards = w.close()
    assert [s["num_tokens"] for s in shards] == [4, 4, 2]  # two full + remainder


def test_shard_writer_length_mismatch_raises(tmp_path):
    w = SFTShardWriter(tmp_path, tokens_per_shard=10, dtype="uint16", tokenizer_name="t")
    with pytest.raises(ValueError, match="length mismatch"):
        w.add([1, 2, 3], [True, False])


# ---- build_sft_corpus: mixer, manifest, drops ----


def _build(tmp_path, sources, **kw):
    cfg = SFTCorpusBuildConfig(
        tokenizer_path=TOKENIZER,
        output_dir=str(tmp_path / "out"),
        sources=sources,
        seq_len=kw.pop("seq_len", 128),
        tokens_per_shard=kw.pop("tokens_per_shard", 10_000),
        **kw,
    )
    manifest = build_sft_corpus(cfg)
    return cfg, manifest


def test_build_writes_manifest_and_shards(tmp_path):
    src = tmp_path / "a.jsonl"
    _write_jsonl(src, [_convo(f"question {i}", f"answer {i}") for i in range(20)])
    _, manifest = _build(tmp_path, [SFTSourceSpec(path=str(src), name="a")])

    assert manifest["kind"] == "sft_packed"
    assert manifest["num_examples"] == 20
    assert manifest["num_tokens"] > 0
    assert 0.0 < manifest["loss_token_fraction"] <= 1.0
    assert manifest["mixture"]["a"]["examples"] == 20
    out = tmp_path / "out"
    assert (out / "sft_manifest.json").exists()
    for s in manifest["shards"]:
        assert (out / s["tokens_path"]).exists()
        assert (out / s["mask_path"]).exists()


def test_mixer_cap_and_repeats(tmp_path):
    big = tmp_path / "big.jsonl"
    gems = tmp_path / "gems.jsonl"
    _write_jsonl(big, [_convo(f"q{i}", f"a{i}") for i in range(100)])
    _write_jsonl(gems, [_convo("special", "gold")] * 5)
    _, manifest = _build(
        tmp_path,
        [
            SFTSourceSpec(path=str(big), name="big", max_examples=10),
            SFTSourceSpec(path=str(gems), name="gems", max_examples=5, repeats=4),
        ],
    )
    mix = manifest["mixture"]
    assert mix["big"]["kept_unique"] == 10 and mix["big"]["examples"] == 10  # capped
    assert mix["gems"]["kept_unique"] == 5 and mix["gems"]["examples"] == 20  # 5 * repeats(4)
    assert manifest["num_examples"] == 30


def test_build_drops_overlong(tmp_path):
    src = tmp_path / "a.jsonl"
    _write_jsonl(src, [_convo("short", "ok"), _convo("x " * 500, "y " * 500)])
    _, manifest = _build(tmp_path, [SFTSourceSpec(path=str(src), name="a")], seq_len=64)
    assert manifest["mixture"]["a"]["dropped_overlong"] == 1
    assert manifest["num_examples"] == 1


def test_build_val_split_is_disjoint(tmp_path):
    src = tmp_path / "a.jsonl"
    _write_jsonl(src, [_convo(f"q{i}", f"a{i}") for i in range(100)])
    _, manifest = _build(tmp_path, [SFTSourceSpec(path=str(src), name="a")], val_fraction=0.2)
    out = tmp_path / "out"
    val_manifest = json.loads((out / "val" / "sft_manifest.json").read_text())
    assert manifest["num_examples"] == 80
    assert val_manifest["num_examples"] == 20
    for s in val_manifest["shards"]:  # val shard paths resolve under out/val/
        assert (out / "val" / s["tokens_path"]).exists()


def test_build_screens_decontam_leak(tmp_path):
    from lithos.data.decontam import write_probes

    probe = (
        "A train leaves Chicago traveling west at sixty miles per hour while a second "
        "train departs from Denver heading east at forty miles per hour on the same track"
    )
    probes = tmp_path / "probes.jsonl"
    write_probes(probes, [probe])

    src = tmp_path / "a.jsonl"
    _write_jsonl(
        src,
        [
            _convo("what is 2 plus 2", "four"),  # clean
            _convo(probe + " when do they meet", "in two hours"),  # leaked prompt
            _convo("define entropy", "a measure of disorder"),  # clean
        ],
    )
    _, manifest = _build(
        tmp_path, [SFTSourceSpec(path=str(src), name="a")], decontam_probes=str(probes)
    )
    assert manifest["mixture"]["a"]["decontam_dropped"] == 1
    assert manifest["num_examples"] == 2
    assert manifest["decontam"]["dropped"] == 1


def test_config_rejects_bad_values():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        SFTSourceSpec(path="p", name="n", repeats=0)  # would silently drop the source
    with pytest.raises(pydantic.ValidationError):
        SFTSourceSpec(path="p", name="n", max_examples=0)
    with pytest.raises(pydantic.ValidationError):
        SFTCorpusBuildConfig(tokenizer_path=TOKENIZER, output_dir="o", sources=[], seq_len=8)
    with pytest.raises(pydantic.ValidationError):
        SFTCorpusBuildConfig(
            tokenizer_path=TOKENIZER, output_dir="o",
            sources=[SFTSourceSpec(path="p", name="n")], val_fraction=1.0,  # empties train
        )


def test_build_empty_raises(tmp_path):
    src = tmp_path / "a.jsonl"
    _write_jsonl(src, [{"messages": [{"role": "user", "content": "only a question"}]}])
    with pytest.raises(ValueError, match="no usable SFT examples"):
        _build(tmp_path, [SFTSourceSpec(path=str(src), name="a")])
