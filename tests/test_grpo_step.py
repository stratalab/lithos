"""Tests for the GRPO step: action-mask loss exclusion + arith/TIR steps (E4).

The exclusion test is the E4 "Done" claim — tool-result tokens contribute nothing
to the policy gradient or the KL. The step tests lock the (refactored) arithmetic
path and run one end-to-end TIR step on a toy model with the TIR vocab.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
from lithos.model.config import ModelConfig
from lithos.posttrain.chat_template import TIR_TOKENS
from lithos.posttrain.dpo import token_logprobs
from lithos.posttrain.grpo_trainer import _labels_from_action_mask, train_grpo
from lithos.posttrain.sft_dataset import IGNORE_INDEX
from lithos.train.config import DataConfig, OptimConfig, ScheduleConfig, TrainConfig

TOKENIZER = "artifacts/tokenizer/fineweb-edu-32k/tokenizer.json"
_has_tok = Path(TOKENIZER).exists()
_POSIX = sys.platform.startswith(("linux", "darwin"))


def test_labels_from_action_mask():
    # positions 4,5 are an injected tool-result span (action_mask False)
    token_ids = [5, 6, 7, 8, 9, 10, 11, 12]
    action_mask = [False, False, True, True, False, False, True, True]
    labels = _labels_from_action_mask(token_ids, action_mask)
    assert len(labels) == len(token_ids) - 1
    assert labels == [
        token_ids[i + 1] if action_mask[i + 1] else IGNORE_INDEX
        for i in range(len(token_ids) - 1)
    ]


def test_tool_result_tokens_excluded_from_pg_and_kl():
    # The E4 invariant, proven numerically without a full model.
    token_ids = [5, 6, 7, 8, 9, 10, 11, 12]
    action_mask = [False, False, True, True, False, False, True, True]  # 4,5 = tool result
    labels = _labels_from_action_mask(token_ids, action_mask)
    labels_t = torch.tensor([labels])

    torch.manual_seed(0)
    vocab = 40
    p_logits = torch.randn(1, len(labels), vocab)
    r_logits = torch.randn(1, len(labels), vocab)
    p_lp = token_logprobs(p_logits, labels_t)  # (1, T)
    r_lp = token_logprobs(r_logits, labels_t)
    diff = r_lp - p_lp
    kl_term = torch.exp(diff) - diff - 1.0  # per-position k3 KL contribution

    masked = [i for i, lab in enumerate(labels) if lab == IGNORE_INDEX]
    assert masked  # sanity: the tool-result span produced masked labels
    for i in masked:
        assert p_lp[0, i].item() == 0.0  # PG contribution (adv * logprob) is 0
        assert abs(kl_term[0, i].item()) < 1e-6  # KL contribution is 0


def _toy_cfg(tmp_path, tokenizer_path, vocab, **over):
    base = dict(
        run_name="grpo-test",
        runs_dir=str(tmp_path / "runs"),
        device="cpu",
        precision="fp32",
        micro_batch_size=1,  # P: prompts/step
        grpo_group_size=2,  # G: rollouts/prompt
        grpo_max_new=6,
        log_interval=1,
        model=ModelConfig(vocab_size=vocab, n_layers=2, hidden=32, n_heads=2, seq_len=64),
        data=DataConfig(kind="grpo", corpus_manifest="unused", seq_len=64, tokenizer_path=tokenizer_path),
        optim=OptimConfig(lr=1e-4),
        schedule=ScheduleConfig(warmup_steps=1, max_steps=1),
    )
    base.update(over)
    return TrainConfig(**base)


@pytest.mark.skipif(not _has_tok, reason="tokenizer artifact absent")
def test_grpo_arith_step_runs(tmp_path):
    # locks the refactored arithmetic path: one step runs, loss logged
    run = train_grpo(_toy_cfg(tmp_path, TOKENIZER, 32000))
    assert run is not None
    recs = [json.loads(line) for line in run.metrics.read_text().splitlines()]
    step_recs = [r for r in recs if "loss" in r]
    assert step_recs and "reward" in step_recs[0] and "accuracy" in step_recs[0]


@pytest.mark.skipif(not (_has_tok and _POSIX), reason="tokenizer artifact / POSIX sandbox")
def test_grpo_tir_step_runs(tmp_path):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER)
    tok.add_special_tokens(list(TIR_TOKENS))  # give the vocab the TIR tokens
    tok_path = tmp_path / "tir_tok.json"
    tok.save(str(tok_path))

    bank = tmp_path / "tasks.jsonl"
    bank.write_text(json.dumps({"id": "1", "prompt": "What is 2 + 2?", "kind": "numeric", "answer": "4"}) + "\n")

    cfg = _toy_cfg(
        tmp_path, str(tok_path), tok.get_vocab_size(),
        grpo_tir=True, grpo_task_bank=str(bank), grpo_max_new=8, grpo_max_tool_calls=1,
    )
    run = train_grpo(cfg)  # random toy model rarely emits a real tool call — just must not crash
    assert run is not None
    recs = [json.loads(line) for line in run.metrics.read_text().splitlines()]
    assert any("tool_calls_per_rollout" in r for r in recs)


def test_grpo_tir_requires_task_bank(tmp_path):
    if not _has_tok:
        pytest.skip("tokenizer artifact absent")
    with pytest.raises(ValueError, match="grpo_task_bank"):
        train_grpo(_toy_cfg(tmp_path, TOKENIZER, 32000, grpo_tir=True))


@pytest.mark.skipif(not _has_tok, reason="tokenizer artifact absent")
def test_grpo_tir_empty_bank_errors(tmp_path):
    # an empty-but-present bank must fail loudly, not ZeroDivisionError mid-step
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="no tasks"):
        train_grpo(_toy_cfg(tmp_path, TOKENIZER, 32000, grpo_tir=True, grpo_task_bank=str(empty)))
