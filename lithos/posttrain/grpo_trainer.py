"""GRPO trainer for RLVR (Phase 11).

DeepSeek-style GRPO (no value model): for each prompt, sample a *group* of G rollouts
from the policy, verify each (shaped reward), and use the group mean/std as the
baseline — advantage = (reward - group_mean) / group_std. The loss is a policy
gradient (raise log-prob of above-average rollouts) plus a per-token KL leash to the
frozen reference. Rollouts are generated inside the step (the new mechanic vs SFT/DPO).

Single-process for now (post-training fits one GPU through ~3B). Logs BOTH the shaped
`reward` (what GRPO optimizes) and `accuracy` (the true objective) — divergence = the
shaping being farmed (the DPO-v1 Goodhart lesson).
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from lithos.model import LithosForCausalLM
from lithos.model.generation import generate
from lithos.posttrain.chat_template import render_prompt, special_ids
from lithos.posttrain.dpo import token_logprobs
from lithos.posttrain.sft_dataset import IGNORE_INDEX
from lithos.posttrain.verifier import MathVerifier, gen_arithmetic
from lithos.train.checkpoint import load_model_weights, save_checkpoint
from lithos.train.config import TrainConfig
from lithos.train.distributed import cleanup_distributed, setup_distributed
from lithos.train.logging import JsonlWriter, RunDir, create_run_dir
from lithos.train.optim import build_optimizer
from lithos.train.scheduler import cosine_lr, set_lr
from lithos.train.tracking import Reporter, init_reporter
from lithos.utils.config import save_resolved_config
from lithos.utils.seed import seed_everything


def _autocast(device: str, precision: str) -> Any:
    if device.startswith("cuda") and precision in ("bf16", "fp16"):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def _pad(seqs: list[list[int]], value: int, length: int) -> list[list[int]]:
    return [s + [value] * (length - len(s)) for s in seqs]


def train_grpo(cfg: TrainConfig, *, resume_from: str | None = None) -> RunDir | None:
    dist = setup_distributed(cfg.device)
    device = dist.device
    seed_everything(cfg.seed)

    policy = LithosForCausalLM(cfg.model).to(device)
    reference = LithosForCausalLM(cfg.model).to(device)
    if cfg.init_from:
        load_model_weights(cfg.init_from, policy)
        load_model_weights(cfg.init_from, reference)
    reference.eval()
    reference.requires_grad_(False)
    optimizer = build_optimizer(policy, cfg.optim)

    tok = Tokenizer.from_file(cfg.data.tokenizer_path)
    sids = special_ids(tok)
    end_id, pad_id = sids["<|end|>"], sids["<pad>"]
    verifier = MathVerifier()
    G, P = cfg.grpo_group_size, cfg.micro_batch_size  # rollouts/prompt, prompts/step

    run: RunDir | None = None
    metrics: JsonlWriter | None = None
    reporter = Reporter(None)
    if dist.is_main:
        run = create_run_dir(cfg.run_name, base=cfg.runs_dir)
        save_resolved_config(cfg, run.resolved_config)
        metrics = JsonlWriter(run.metrics)
        gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else None
        reporter = init_reporter(
            cfg, run_id=run.root.name, run_dir=str(run.root), is_main=True,
            runtime={"resolved_device": device, "gpu": gpu, "stage": "grpo", "group_size": G},
        )

    gen_g = torch.Generator(device=device).manual_seed(cfg.seed)
    policy.train()
    for step in range(cfg.schedule.max_steps):
        lr = cosine_lr(step, cfg.schedule, cfg.optim.lr)
        set_lr(optimizer, lr)
        tasks = gen_arithmetic(P, seed=cfg.seed + step + 1, max_val=10, ops="+")

        inputs: list[list[int]] = []
        labels: list[list[int]] = []
        advantages: list[float] = []
        sum_reward = sum_acc = 0.0
        n_roll = 0
        for task in tasks:
            pids = render_prompt([{"role": "user", "content": task["prompt"]}], tok)
            with torch.no_grad():
                out = generate(
                    policy, torch.tensor([pids], device=device).repeat(G, 1), cfg.grpo_max_new,
                    temperature=cfg.grpo_temperature, top_p=0.95, eos_token_id=end_id, generator=gen_g,
                )
            rewards, responses = [], []
            for gi in range(G):
                resp = out[gi].tolist()[len(pids):]
                if end_id in resp:
                    resp = resp[: resp.index(end_id) + 1]  # keep the terminator
                text = tok.decode([i for i in resp if i != end_id], skip_special_tokens=True)
                rewards.append(verifier.reward(text, task["answer"]))
                sum_acc += verifier.correctness(text, task["answer"])
                responses.append(resp)
            n_roll += G
            sum_reward += sum(rewards)

            mean_r = float(np.mean(rewards))
            std_r = float(np.std(rewards))
            for gi in range(G):
                resp = responses[gi]
                if not resp:
                    continue
                full = pids + resp
                y = [full[i + 1] if (i + 1) >= len(pids) else IGNORE_INDEX for i in range(len(full) - 1)]
                if all(v == IGNORE_INDEX for v in y):
                    continue
                inputs.append(full[:-1])
                labels.append(y)
                advantages.append((rewards[gi] - mean_r) / (std_r + 1e-4))

        if not inputs:  # every rollout degenerate this step
            continue
        length = max(len(s) for s in inputs)
        x = torch.tensor(_pad(inputs, pad_id, length), device=device)
        y = torch.tensor(_pad(labels, IGNORE_INDEX, length), device=device)
        adv = torch.tensor(advantages, device=device, dtype=torch.float32)

        with _autocast(device, cfg.precision):
            p_logits, _ = policy(x)
            p_tok = token_logprobs(p_logits, y)
        with torch.no_grad(), _autocast(device, cfg.precision):
            r_logits, _ = reference(x)
            r_tok = token_logprobs(r_logits, y)

        pg_loss = -(adv * p_tok.sum(-1)).mean()
        diff = r_tok - p_tok                              # 0 at masked/pad positions
        kl = (torch.exp(diff) - diff - 1.0).sum(-1).mean()  # k3 estimator, >= 0
        loss = pg_loss + cfg.grpo_kl_coef * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optim.grad_clip)
        optimizer.step()

        if dist.is_main and (step % cfg.log_interval == 0 or step == cfg.schedule.max_steps - 1):
            rec = {
                "step": step, "learning_rate": lr, "loss": round(loss.item(), 4),
                "pg_loss": round(pg_loss.item(), 4), "kl": round(kl.item(), 4),
                "reward": round(sum_reward / max(n_roll, 1), 4),
                "accuracy": round(sum_acc / max(n_roll, 1), 4),  # the TRUE objective — watch this
            }
            metrics.write(rec)  # type: ignore[union-attr]
            reporter.log(rec, step)

    if dist.is_main and run is not None:
        save_checkpoint(
            run.checkpoints / f"step_{cfg.schedule.max_steps:06d}", model=policy, optimizer=optimizer,
            step=cfg.schedule.max_steps, tokens_seen=0, dataloader_state={},
            meta={"stage": "grpo", "init_from": cfg.init_from, "tokenizer": cfg.data.tokenizer_path},
        )
    cleanup_distributed()
    return run
