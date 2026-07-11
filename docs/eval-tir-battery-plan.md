# E8 — the TIR tool-uplift battery + the public benchmark

Realizes the two eval instruments that don't exist yet and matter most: the
**tool-uplift metric** (`eval-plan.md` §1 — the product thesis, quantified) and the
**public benchmark** (§6 — the category has no scoreboard). The 2026 eval survey found
*no community-accepted standard for reason-then-execute STEM eval and no engineering
board that separates sub-4B models*, so this is the credible path, not a flourish.

## Context

The whole F7 seam we hardened with Chisel — `check_code`, the pinned
`CHECKER_IMPORT_SET`, the code-harness goldens, the sandbox — was built to be **one
verifier with three consumers**: RLVR reward, data verification, and **eval**. Eval is
the consumer we never wired. And the generation-with-tools loop the battery needs
already exists: `posttrain/tir_rollout.py::tir_rollout` generates a TIR episode,
executes tool calls in the sandbox, injects results, and returns the completion +
`tool_calls`/`tool_outputs` + health flags. **The tool-uplift metric is that same
rollout run twice per problem — once with tools off, once on — and the verified
solve-rate diffed.** So E8 is mostly wiring + a statistics layer + packaging, not new
capability.

**Finding (from reading the RLVR path):** `_collect_tir` (grpo_trainer.py:203) grades
a rollout with `verify(roll.completion_text, task)` — the *prose* final answer. That is
correct for answer-checked kinds (numeric/symbolic/units, where the tool computes the
value the model then states) but **wrong for `kind=code`**, which should verify the
*executed tool code* against the task's tests (the TODO at grpo_trainer.py:201). The
battery must get this right, so E8 introduces one `verify_tir(roll, task)` helper that
both the battery and (later) GRPO can share.

## Approach

### Part A — the tool-uplift battery (the internal instrument, unblocked now)

**1. Eval pool = post-cutoff, disjoint (reuse, don't rebuild).** `load_tasks` →
`filter_by_level` → `split_by_year(cutoff)`; score **only** the post-cutoff hold-out;
`assert_disjoint(rlvr_pool, eval_pool)` as a hard gate (family-aware — already built).
The battery never touches a task the RLVR pool trained on. This is `eval-plan.md`
principle 5, and the code exists in `taskbank.py`; E8 just calls it.

**2. Matched two-arm rollout** (`lithos/evals/tir_battery.py`, new). Per task, run
`tir_rollout` **twice with identical decoding + seed**:
- **tools-off** (`max_tool_calls=0`): the loop is `range(max_tool_calls + 1)` = one
  segment, so the model answers from **chain-of-thought alone** — the honest control
  for "computation via code vs. parametric recall." (Off = *no sandbox*, not *no
  reasoning*.)
- **tools-on** (`max_tool_calls=N`): the full TIR episode.
- Grade both with `verify_tir(roll, task)` (step 3). Decoding: **greedy** (the honest
  pass@1 default); `maj@k`/`RM@k` only when explicitly labeled (`eval-plan.md` §4 pin —
  RM@k alone can swing a 7B 12→21 AIME solves, a decoding artifact not a capability).

**3. `verify_tir(roll, task)`** (shared helper — fold into `taskbank.py` beside
`verify`). For answer-checked kinds → `verify(roll.completion_text, task)` (unchanged).
For `kind=code` → run the model's **executed tool code** (`roll.tool_calls`) against
`task.tests` via `check_code` — closing the grpo_trainer.py:201 TODO. One helper, two
consumers (battery now, GRPO reward later).

**4. The metric + error bars** (`eval-plan.md` §0.9). Each problem is run in *both*
arms → a **paired** design, so:
- **Tool-uplift = solve_rate(on) − solve_rate(off), reported per difficulty tier**
  (`task.level`) *and* overall — a single average understates it (uplift ≈0 on
  easy/saturated items, largest on computation-heavy ones).
- **Paired CI:** McNemar on the discordant pairs (off✗→on✓ vs on✗→off✓) + a
  **clustered** standard error (cluster by `family_id`, since near-duplicates aren't
  independent — clustered SEs run >3× naive). State whether the uplift is significant,
  don't just report the point.
- **Rollout health** (from `RolloutResult`): tool-call rate, `num_malformed_calls`
  rate, `truncated` rate, tool-calls-per-solve.

**5. Transcript capture** (`eval-plan.md` principle 8 — watch rollouts, not the curve).
Persist a sample of `RolloutResult`s (`tool_calls`, `tool_outputs`, `completion_text`)
to JSONL so every reported delta is spot-checkable against real transcripts — and the
same records feed Petra's "what lit up" gaming screen (`tool` vs parametric channel).

**6. Scorecard + config wiring.** Add `TIRBatteryConfig` to `evals/config.py`; extend
the scorecard entry with a `tir` block (per-tier uplift + CIs + health); reuse
`scorecard.append_entry`/`diff` and `report._render_markdown` for the uplift table.

### Part B — the public benchmark (packaging Part A; post-MVP, design-as-if now)

**7. Freeze + publish the post-cutoff slice.** The public set *is* the battery's
hold-out, packaged: frozen JSONL (problem + answer + `level` + `year` + `family_id` +
**canary GUID**), the `tir_battery` runner as the one-command harness, **executable
grading only** (no judge anywhere), and a **baseline scorecard** from running the §3
anchors — size-matched, weight-class-above, same-size specialists — through the
*identical* loop, **losses included** (that's what earns trust; `eval-plan.md` §6's
four suspicions each get a structural answer here).

**8. Contamination hardening = the credibility bar.** Canary strings embedded; rolling
**year-partition renewed annually** (LiveCodeBench pattern); the decontam probe list
shipped with it. The novel, category-defining content is the **engineering + physics
exam-derived sets** (value+units checked) — the white space no public board covers.

## Files

**New:** `lithos/evals/tir_battery.py` (two-arm runner + metric); `lithos/evals/tir_stats.py`
(McNemar + clustered SE — or fold into tir_battery); `tests/test_tir_battery.py`;
`configs/eval/tir.yaml`; `scripts/run_tir_battery.py`.
**Modify:** `lithos/posttrain/taskbank.py` (`verify_tir` helper); `lithos/evals/config.py`
(`TIRBatteryConfig`); `lithos/evals/scorecard.py` (`tir` entry block);
`lithos/evals/report.py` (uplift table); `lithos/evals/__init__.py` (exports).

## Reuse (don't rebuild)

- **`tir_rollout`** (posttrain/tir_rollout.py:65) — the entire tools loop. Off-arm =
  `max_tool_calls=0`; on-arm = `max_tool_calls=N`. This is the win: the engine exists.
- **`verify` / `check_code`** (taskbank.py:96 / verifier.py) — the grader. Eval is the
  third consumer of the one verifier; `verify_tir` extends it to code-kind.
- **`split_by_year` / `filter_by_level` / `assert_disjoint`** (taskbank.py) — the
  post-cutoff, disjoint, family-aware pool discipline, already built + tested.
- **`render_prompt`** (chat_template) + **`load_model_from_checkpoint`** (evals/run.py:31)
  + the `tir_ids`/`sids` the GRPO path already constructs — model/prompt plumbing.
- **`scorecard.append_entry`/`diff`, `report._render_markdown`** — results + rendering.
- **The §3 anchor set** (`reference_scorecard.jsonl` pattern) — baselines, re-run on the
  TIR loop (scores only compare within a battery_version, so anchors don't transfer from
  v1 — re-run them here).

## Scope notes (honest boundaries)

- **Code-kind grading is the correctness crux** — `verify_tir` must verify executed tool
  code vs. tests, not prose; a battery that graded code by prose would silently mis-score
  the exact tasks the tool thesis is about.
- **Both arms share seed + decoding**, or the uplift is a decoding artifact. Greedy is the
  headline; `maj@k` labeled and separate.
- **The eval reports *verified* solves.** The `heuristic_gaming_check` (grpo_trainer.py:205)
  + Petra transcript read is the reward-hacking *audit* layer, not the headline metric —
  but the captured transcripts (step 5) are what make the audit possible.
- **Publishing (Part B) is post-MVP**; the internal battery (Part A) is unblocked today —
  the sandbox, rollout, verifier, and pool-split all exist.
- **Not in E8:** the judged-comparison harness, the regurgitation eval, and the
  quantized edge/tool-call-integrity probe (separate `eval-plan.md` §7 items).

## Verification

1. **Unit:** `max_tool_calls=0` yields `num_tool_calls == 0`; `verify_tir` grades a
   `kind=code` rollout by its tool code (passing tests → correct, wrong code → not),
   distinct from prose; McNemar + clustered-SE funcs correct on synthetic contingency
   tables (incl. the zero-discordant and all-clustered degenerate cases).
2. **The "Done" (positive uplift on a fixture):** a computation-heavy task
   (e.g. eigenvalues / numeric integration) **solves with tools and fails without**,
   and the battery reports a positive tool-uplift with a CI on a tiny TIR-SFT'd
   checkpoint — the affirmative, quantified answer to the product thesis.
3. **Anchor baseline:** a same-size instruct anchor runs through the identical loop and
   lands in the scorecard as the public-set baseline (Part B smoke).
4. **Reproducibility:** `scripts/run_tir_battery.py --config configs/eval/tir.yaml
   --checkpoint <ckpt>` → per-tier uplift table + CIs + transcript sample, appended to
   the scorecard; re-run is deterministic under the pinned seed.
5. **Gates:** `uv run ruff check` + `uv run mypy lithos` clean; full `uv run pytest`
   green.
