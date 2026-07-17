# Tinker (Thinking Machines) — post-training API review & what we take from it

> **Investigated 2026-07-17.** Status: **T1+T2 LANDED same day** (code green:
> ruff + mypy + 486 tests — see the LANDED notes in §4); T3–T6 remain **PROPOSED**,
> each naming where it would land once accepted.
> Companion to the settled post-training stack (`post-training-implementation-plan.md`)
> and the SFT/TIR data machinery (`lithos/posttrain/`).

## 1 · What Tinker is, and why it's worth studying

[Tinker](https://thinkingmachines.ai/tinker/) is Thinking Machines' managed training API:
the training loop is Python on your laptop; their cluster runs it. The design bet is
**primitives, not pipelines** — four core calls (`forward_backward`, `optim_step`,
`sample`, `save_state`) from which every post-training method (SFT, GRPO/PPO, DPO,
RLHF, distillation) is expressed as *client-side code* in the open-source
[Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook). Since the
Oct-2025 launch it has grown from LoRA-only to full fine-tuning, 28+ open-weight models
(1B–1T+, dense + MoE), VLMs, and audio.

Why we care: John Schulman (PPO, RLHF) is behind the API design. This is his answer to
"what should the training-loop abstraction be" — the same question our
`lithos/posttrain/` stack answers. We do **not** care about the service itself (§6);
we care about the *seams* they chose, because we chose most of the same ones
independently and the deltas are cheap to adopt now and expensive later.

## 2 · The data format: there isn't one — that's the finding

There is no dataset schema in the JSONL sense. The atomic unit is the
[`Datum`](https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-sft/):

```
Datum:
  model_input:     token sequence, shape (N,)
  loss_fn_inputs:  dict of per-token tensors — everything the loss needs
```

Per method, only `loss_fn_inputs` changes ([losses reference](https://tinker-docs.thinkingmachines.ai/tinker/losses/)):

| method | loss name | loss_fn_inputs fields (all shape (N,)) |
|---|---|---|
| SFT | `cross_entropy` | `target_tokens`, `weights` (0 = no gradient, 1 = train) |
| RL (all variants) | `importance_sampling` / `ppo` / `cispo` / `dro` | `target_tokens`, `weights`, `logprobs` (sampler's), `advantages` |
| DPO | via chosen/rejected Datum pairs + frozen π_ref | same fields, textbook DPO loss |
| anything else | `forward_backward_custom` | arbitrary loss over logprobs |

Their motto: **"everything the loss needs is in the Datum."** Consequences:

- **One trainer loop serves the entire post-training stack.** The *method* lives
  entirely in (a) how Datums are constructed and (b) which loss name is passed.
  SFT→RLVR→DPO→distillation is a data-construction change, not a trainer change.
- **`weights` is per-token and float**, not a boolean mask — 0/1 masking is just the
  degenerate case. Curriculum weighting, boilerplate down-weighting, and
  advantage-weighted SL are all "put a different float in the array."
- **`logprobs` (the sampler's) is a first-class field** because the sampler and
  trainer are different processes with different numerics — rollouts are always
  slightly off-policy, and the `importance_sampling` loss corrects with the stored
  p/q ratio. The correction is possible *only because the rollout record carries it*.

Above the Datum sits a [`Renderer`](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/renderers/)
registry — messages → tokens + weights, with a `TrainOnWhat` enum (which turns get
loss: last assistant message vs all) and **bidirectional** token↔message conversion
(sampled tokens parse back into structured messages). For RL, an
[`Env` stack](https://tinker-docs.thinkingmachines.ai/cookbook/rl/):
`Env.initial_observation()` / `Env.step(action) → StepResult(reward, episode_done)`;
`ProblemEnv` for single-turn verifiable tasks (`get_question()` /
`check_answer(response) → float`); `EnvGroupBuilder.compute_group_rewards()` as the
GRPO group-normalization seam; `RLDataset.get_batch()` above that. Generic assembly
(`compute_advantages` → `assemble_training_data`) turns trajectories into Datums —
the env never sees the training format.

## 3 · Where we already match (validation, not work)

- `chat_template.render_conversation` → `(input_ids, loss_mask)` **is** their
  Renderer → Datum in miniature, including render-by-name specials resolution.
- `taskbank` + `verifier` ≈ `ProblemEnv` (`get_question`/`check_answer → float`).
  The verifier-shared-with-evals doctrine matches their verifiable-reward recipes.
- GRPO group normalization, sandboxed code-RL, DPO-with-frozen-reference: all
  textbook in the cookbook, all already in `lithos/posttrain/`. Their DPO teaches
  us nothing new.

Two independent teams choosing the same seams is evidence the seams are right.

## 4 · Deltas to adopt (PROPOSED, ranked)

**T1 · Float `weights` + one canonical record across all four trainers.**
*(→ `post-training-implementation-plan.md`, format epic; touches `sft_corpus.py`,
`sft_dataset.py`, `preference_dataset.py`, `grpo_trainer.py`.)*
Generalize `loss_mask: list[bool]` to per-token float `weights` (bool = 0/1 case),
and converge the per-trainer data shapes on one canonical record:
`tokens + weights (+ logprobs + advantages when present)`. The trainer loop becomes
method-agnostic; methods differ only in record construction + loss choice.
**Doctrinal bonus that makes this more than plumbing:** the tier gate's rule is
*"only tokens that receive a gradient are gated"* — with this format **the weights
vector literally is the gradient gate**. One array becomes the single source of truth
for training *and* the attestation manifest (`weight > 0 ⇒ tier must pass`), replacing
the parallel bookkeeping of `SFTSourceSpec.tier` (targets) vs `prompt_tier` (masked).
Cheap now; a migration after the SFT corpora are built at scale.
**LANDED 2026-07-17:** `lithos/posttrain/record.py` (`TrainingRecord`: tokens +
float `weights` + optional `logprobs`/`advantages`; `labels()` is the one shift
implementation). The renderer emits float weights (`Rendered.weights`; `loss_mask`
is the derived view); SFT shards store a float32 `.weights.bin` stream
(`weights_path` in the manifest; legacy uint8 `.mask.bin` still loads); GRPO
collectors emit records with per-token advantages; DPO's `build_xy` renders
through the record, so all four trainers now share the shape. **Remaining seam:**
weighted cross-entropy in the train loop — until it lands, the packed SFT loader
*fail-closes* on fractional weights (they would otherwise silently train at 1.0).

**T2 · Rollout records carry per-token sampler `logprobs`, starting now.**
*(→ same epic; touches `tir_rollout.py`, `grpo_trainer.py` record shape.)*
We deferred fast vLLM rollouts to the flagship. The moment rollouts move off the
trainer's own forward pass, sampler≠trainer numerics make the data off-policy — the
exact failure Tinker's `importance_sampling` loss exists for. If the record format
already carries sampler logprobs, the fix on that day is a loss-function swap; if not,
it's a data-format migration mid-flagship. Cost today: one array per rollout.
**LANDED 2026-07-17:** `generate(..., return_logprobs=True)` returns the sampler's
per-token log-probability under the distribution actually sampled from (after
decode policy, temperature, top-k/top-p; 0.0 on forced eos padding — verified
against a manual log-softmax recompute in `test_generation.py`);
`RolloutResult.logprobs` carries it through TIR episodes (0.0 at prompt +
injected tool-result positions) and `to_record()` lifts the episode into the
canonical record. The on-policy GRPO loss deliberately does not read it (p/q = 1
today); the importance-sampling variant slots into `_grpo_loss` when E5 lands.

**T3 · Bidirectional rendering (tokens → messages) for TIR.**
*(→ check against `chat_template.py`; small add if missing.)*
Their renderers parse sampled tokens back into structured messages — precisely
`tir_rollout`'s loop (sample → extract tool call → sandbox → re-render). Verify our
chat template round-trips; add the inverse if it doesn't. Multi-turn TIR RL uses this
every step.

**T4 · On-policy distillation experiment for the 4B hero.**
*(→ bank beside the recipe: SFT → **[OPD?]** → RLVR-TIR → DPO.)*
Phase-11 banked "distillation transfers style, not substance, on a tiny student" —
but that was *off-policy* (training on teacher text).
[On-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation)
is a different mechanism: the **student** samples, the teacher grades every token
(reverse KL) — dense per-token supervision on the student's own states, reportedly
around an order of magnitude cheaper than RL for comparable gains. The 4B hero shares
Qwen's tokenizer with larger Qwen3 teachers, so logits align (open teachers only —
doctrine-clean). May work exactly where the off-policy attempt failed; slot it as an
experiment between SFT and RLVR. Their cookbook has single- and multi-teacher
reference implementations.

**T5 · Chisel ships a Tinker-compatible `Env` adapter (strategic hook).**
*(→ `chisel.md`, env-seam requirement.)*
Tinker's cookbook has math RL, code RL, and Search-R1-style tool use — **nothing in
physical engineering** (thermo/controls/fluids/EM), which is Chisel's claimed wedge.
Their `Env`/`EnvGroupBuilder` interface is on track to become the de-facto standard
for RL-environment interop. If Chisel's verified environments expose a
Tinker-compatible adapter, every Tinker customer is a potential Chisel customer, at
the cost of one thin wrapper. Design Chisel's env seam with this in mind; build the
adapter when Chisel GTM unparks.

**T6 · LoRA as the default for 4B RLVR (not a deferred optimization).**
*(→ compute-routing note in the plans.)*
[LoRA Without Regret](https://thinkingmachines.ai/blog/) (Schulman et al., Sept 2025):
LoRA on **all** layers (MLP included, ~10× the full-FT learning rate) matches full
fine-tuning when capacity isn't binding — and for RL it essentially always suffices,
because policy gradients carry ~bits per episode and capacity never binds. We filed
LoRA under "flagship-only deferrals"; this reframes it as the *default* for 4B RLVR,
pulling those runs down the price curve (rented H100 → cheaper cards, longer local
iteration). Verify on our own 100M/500M RLVR before trusting at 4B.

## 5 · Priority & sequencing

T1+T2 are **format decisions — cheap now, expensive later**; they belong in whatever
epic next touches the record shapes, before the SFT corpora are built at scale.
T3 is an afternoon check. T4/T6 are banked experiments (flagship-tier, GPU-gated).
T5 is a design constraint on Chisel, zero code today.

## 6 · What we do NOT take

- **The service.** The models-only catalog can't train a from-scratch Lithos; renting
  their loop builds none of the craft (the whole own-the-foundation thesis); and the
  stack is already built. The cookbook's value is as a **reference implementation to
  diff against**, not infrastructure. (It would be *doctrine-clean* to use — open
  models, our data — just strategically empty for us.)
- **Their DPO** — textbook, same as ours.
- **Managed-infra envy.** Our spot-first R2-checkpointed setup already covers the
  "no cluster babysitting" property at our scale.

## Sources

[Tinker docs](https://tinker-docs.thinkingmachines.ai/tinker/) ·
[First SFT / Datum](https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-sft/) ·
[Losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/) ·
[Renderers](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/renderers/) ·
[RL abstractions](https://tinker-docs.thinkingmachines.ai/cookbook/rl/) ·
[DPO guide](https://tinker-docs.thinkingmachines.ai/cookbook/preferences/dpo-guide/) ·
[Cookbook (GitHub)](https://github.com/thinking-machines-lab/tinker-cookbook) ·
[Announcing Tinker](https://thinkingmachines.ai/news/announcing-tinker/) ·
[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation) ·
[Connectionism blog](https://thinkingmachines.ai/blog/)
