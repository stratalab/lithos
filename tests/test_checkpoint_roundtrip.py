"""Tests for lithos.train.checkpoint — save/load preserves outputs (PRD §6.3.6)."""

import torch
from lithos.train.checkpoint import load_checkpoint, save_checkpoint
from lithos.train.config import OptimConfig
from lithos.train.optim import build_optimizer

from tests.helpers import make_model


def test_checkpoint_preserves_outputs_and_state(tmp_path):
    model = make_model(seed=1)
    model.eval()
    opt = build_optimizer(model, OptimConfig())
    ids = torch.randint(0, model.cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits_ref, _ = model(ids)

    dl_state = {"epoch": 0, "position": 4, "seed": 0}
    ckpt = save_checkpoint(
        tmp_path / "ck",
        model=model,
        optimizer=opt,
        step=5,
        tokens_seen=1000,
        dataloader_state=dl_state,
        meta={"tokenizer": "t", "corpus": "c"},
    )

    # A freshly, differently-initialized model produces different outputs...
    fresh = make_model(seed=999)
    fresh.eval()
    fresh_opt = build_optimizer(fresh, OptimConfig())
    with torch.no_grad():
        before, _ = fresh(ids)
    assert not torch.allclose(before, logits_ref)

    # ...until we load the checkpoint, after which outputs match exactly.
    state = load_checkpoint(ckpt, fresh, fresh_opt)
    assert state["step"] == 5
    assert state["tokens_seen"] == 1000
    assert state["dataloader"] == dl_state
    with torch.no_grad():
        after, _ = fresh(ids)
    torch.testing.assert_close(after, logits_ref, atol=1e-6, rtol=1e-6)
