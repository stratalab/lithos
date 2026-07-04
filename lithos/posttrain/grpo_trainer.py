"""GRPO trainer for RLVR (Phase 11 arithmetic test bench + E4 TIR mode).

DeepSeek-style GRPO (no value model): for each prompt, sample a *group* of G rollouts
from the policy, verify each (shaped reward), and use the group mean/std as the
baseline — advantage = (reward - group_mean) / group_std. The loss is a policy
gradient (raise log-prob of above-average rollouts) plus a per-token KL leash to the
frozen reference. Rollouts are generated inside the step (the new mechanic vs SFT/DPO).

Two collection modes share one loss (``_grpo_loss``): the arithmetic test bench
(single ``generate`` + ``MathVerifier``) and **TIR** (``cfg.grpo_tir``): multi-segment
``tir_rollout`` that executes tool calls in the E1 sandbox and scores with the E1
verifier. Both build labels from a per-token **action mask** (``_labels_from_action_mask``);
injected tool-result tokens are non-actions, so the existing ``IGNORE_INDEX`` machinery
excludes them from the policy gradient AND the KL for free.

Single-process for now (post-training fits one GPU through ~3B; TIR rollouts are
sequential — batching is E5). Logs BOTH the shaped `reward` (what GRPO optimizes) and
`accuracy` (the true objective) — divergence = the shaping being farmed (Goodhart).
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from lithos.model import LithosForCausalLM
from lithos.model.generation import generate
from lithos.posttrain.chat_template import render_prompt, special_ids, tir_token_ids
from lithos.posttrain.dpo import token_logprobs
from lithos.posttrain.sft_dataset import IGNORE_INDEX
from lithos.posttrain.taskbank import load_tasks, verify
from lithos.posttrain.tir_rollout import tir_rollout
from lithos.posttrain.verifier import (
    MathVerifier,
    gen_arithmetic,
    heuristic_gaming_check,
    shaped_reward,
)
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


def _labels_from_action_mask(token_ids: list[int], action_mask: list[bool]) -> list[int]:
    """Next-token labels, ``IGNORE_INDEX`` wherever the (shifted) token is not a policy
    action — i.e. prompt and injected tool-result positions drop out of PG + KL."""
    return [
        token_ids[i + 1] if action_mask[i + 1] else IGNORE_INDEX
        for i in range(len(token_ids) - 1)
    ]


def _grpo_loss(
    policy: LithosForCausalLM,
    reference: LithosForCausalLM,
    inputs: list[list[int]],
    labels: list[list[int]],
    advantages: list[float],
    *,
    pad_id: int,
    kl_coef: float,
    device: str,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Policy-gradient loss + per-token k3 KL to the frozen reference. ``token_logprobs``
    returns 0 at ``IGNORE_INDEX`` positions, so masked (prompt/tool-result) tokens
    contribute nothing to either term. Returns (total_loss, pg_loss, kl)."""
    length = max(len(s) for s in inputs)
    x = torch.tensor(_pad(inputs, pad_id, length), device=device)
    y = torch.tensor(_pad(labels, IGNORE_INDEX, length), device=device)
    adv = torch.tensor(advantages, device=device, dtype=torch.float32)

    with _autocast(device, precision):
        p_logits, _ = policy(x)
        p_tok = token_logprobs(p_logits, y)
    with torch.no_grad(), _autocast(device, precision):
        r_logits, _ = reference(x)
        r_tok = token_logprobs(r_logits, y)

    pg_loss = -(adv * p_tok.sum(-1)).mean()
    diff = r_tok - p_tok  # 0 at masked/pad positions
    kl = (torch.exp(diff) - diff - 1.0).sum(-1).mean()  # k3 estimator, >= 0
    return pg_loss + kl_coef * kl, pg_loss, kl


def _collect_arith(
    policy: LithosForCausalLM,
    tok: Tokenizer,
    sids: dict[str, int],
    verifier: MathVerifier,
    cfg: TrainConfig,
    step: int,
    device: str,
    generator: torch.Generator,
) -> tuple[list[list[int]], list[list[int]], list[float], dict[str, Any]]:
    """Arithmetic test-bench rollouts: one batched ``generate`` of G per prompt,
    scored by ``MathVerifier``. The action mask is prompt-False + completion-True."""
    G, P = cfg.grpo_group_size, cfg.micro_batch_size
    end_id = sids["<|end|>"]
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
                temperature=cfg.grpo_temperature, top_p=0.95, eos_token_id=end_id, generator=generator,
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

        mean_r, std_r = float(np.mean(rewards)), float(np.std(rewards))
        for gi in range(G):
            resp = responses[gi]
            if not resp:
                continue
            full = pids + resp
            action_mask = [False] * len(pids) + [True] * len(resp)
            y = _labels_from_action_mask(full, action_mask)
            if all(v == IGNORE_INDEX for v in y):
                continue
            inputs.append(full[:-1])
            labels.append(y)
            advantages.append((rewards[gi] - mean_r) / (std_r + 1e-4))

    metrics = {
        "reward": round(sum_reward / max(n_roll, 1), 4),
        "accuracy": round(sum_acc / max(n_roll, 1), 4),  # the TRUE objective — watch this
    }
    return inputs, labels, advantages, metrics


def _collect_tir(
    policy: LithosForCausalLM,
    tok: Tokenizer,
    tir_ids: dict[str, int],
    sids: dict[str, int],
    tasks: list,
    cfg: TrainConfig,
    step: int,
    device: str,
    generator: torch.Generator,
) -> tuple[list[list[int]], list[list[int]], list[float], dict[str, Any]]:
    """TIR rollouts: G multi-segment ``tir_rollout``s per prompt, scored by the E1
    verifier; tool-result tokens are masked out of the loss via the action mask."""
    G, P = cfg.grpo_group_size, cfg.micro_batch_size
    step_tasks = [tasks[(step * P + i) % len(tasks)] for i in range(P)]

    inputs: list[list[int]] = []
    labels: list[list[int]] = []
    advantages: list[float] = []
    sum_reward = sum_acc = 0.0
    n_roll = n_tool = n_gamed = 0
    for task in step_tasks:
        pids = render_prompt([{"role": "user", "content": task.prompt}], tok)
        rollouts, rewards = [], []
        for _ in range(G):
            roll = tir_rollout(
                policy, pids, tok, tir_ids, sids, device=device,
                max_new=cfg.grpo_max_new, max_tool_calls=cfg.grpo_max_tool_calls,
                temperature=cfg.grpo_temperature, top_p=0.95, generator=generator,
                timeout_s=cfg.grpo_tool_timeout_s, result_token_cap=cfg.grpo_result_token_cap,
            )
            # Reward from the completion text — correct for answer-checked tasks
            # (numeric/symbolic/units), where the tool computes the answer the model
            # states. TODO: code-kind TIR should instead verify the executed tool
            # code against the task's tests (roll.tool_calls), not the prose.
            result = verify(roll.completion_text, task, timeout_s=cfg.grpo_tool_timeout_s)
            reward = shaped_reward(roll.completion_text, result)
            if any(heuristic_gaming_check(code, task.answer) for _, code in roll.tool_calls):
                reward = 0.0  # anti-gaming pre-screen (E1e); the LLM judge is deferred
                n_gamed += 1
            rewards.append(reward)
            rollouts.append(roll)
            sum_acc += float(result.correct)
            n_tool += roll.num_tool_calls
        n_roll += G
        sum_reward += sum(rewards)

        mean_r, std_r = float(np.mean(rewards)), float(np.std(rewards))
        for gi, roll in enumerate(rollouts):
            if len(roll.token_ids) < 2:
                continue
            y = _labels_from_action_mask(roll.token_ids, roll.action_mask)
            if all(v == IGNORE_INDEX for v in y):
                continue
            inputs.append(roll.token_ids[:-1])
            labels.append(y)
            advantages.append((rewards[gi] - mean_r) / (std_r + 1e-4))

    metrics = {
        "reward": round(sum_reward / max(n_roll, 1), 4),
        "accuracy": round(sum_acc / max(n_roll, 1), 4),
        "tool_calls_per_rollout": round(n_tool / max(n_roll, 1), 3),
        "gamed": n_gamed,
    }
    return inputs, labels, advantages, metrics


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
    pad_id = sids["<pad>"]
    verifier = MathVerifier()
    G = cfg.grpo_group_size

    tir_ids = tir_tasks = None
    if cfg.grpo_tir:
        if not cfg.grpo_task_bank:
            raise ValueError("grpo_tir=true requires grpo_task_bank (a problem-bank JSONL)")
        tir_tasks = load_tasks(cfg.grpo_task_bank)
        if not tir_tasks:
            raise ValueError(f"grpo_task_bank {cfg.grpo_task_bank!r} contains no tasks")
        tir_ids = tir_token_ids(tok)  # requires the STEM tokenizer's TIR vocab

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
            runtime={"resolved_device": device, "gpu": gpu, "stage": "grpo",
                     "group_size": G, "tir": cfg.grpo_tir},
        )

    gen_g = torch.Generator(device=device).manual_seed(cfg.seed)
    policy.train()
    for step in range(cfg.schedule.max_steps):
        lr = cosine_lr(step, cfg.schedule, cfg.optim.lr)
        set_lr(optimizer, lr)

        if cfg.grpo_tir:
            assert tir_ids is not None and tir_tasks is not None
            inputs, labels, advantages, extra = _collect_tir(
                policy, tok, tir_ids, sids, tir_tasks, cfg, step, device, gen_g
            )
        else:
            inputs, labels, advantages, extra = _collect_arith(
                policy, tok, sids, verifier, cfg, step, device, gen_g
            )
        if not inputs:  # every rollout degenerate this step
            continue

        loss, pg_loss, kl = _grpo_loss(
            policy, reference, inputs, labels, advantages,
            pad_id=pad_id, kl_coef=cfg.grpo_kl_coef, device=device, precision=cfg.precision,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optim.grad_clip)
        optimizer.step()

        if dist.is_main and (step % cfg.log_interval == 0 or step == cfg.schedule.max_steps - 1):
            rec = {
                "step": step, "learning_rate": lr, "loss": round(loss.item(), 4),
                "pg_loss": round(pg_loss.item(), 4), "kl": round(kl.item(), 4), **extra,
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
