# The Flywheel (Tectonics) — the autonomous training loop

**Status: parked / design-only.** Premature to build (decided 2026-07) — the loop is only
worth automating once each leg runs by hand end-to-end. This doc captures the shape so it
isn't re-derived later, and pins the one part that decides whether the loop converges or
spins: **the eval loop is the flywheel's fitness function.**

Tectonics (provisional codename) is the 7th, capstone leg of the Strata ecosystem — the
control plane that turns the other six into a closed loop that improves the model without a
human in the inner cycle. It is not a new model or a new tool; it is the *scheduler +
selection pressure* wired across the legs we already have.

## The legs it ties together

| Leg | Role in the loop |
|---|---|
| **Chisel** | produces + verifies data (problem banks, TIR traces, prefs) |
| **Lithos** | trains the next checkpoint (SFT → GRPO-TIR) |
| **the eval loop** | *scores* the checkpoint — the fitness function (below) |
| **Petra** | attributes capability → training source; surfaces what's missing/failing |
| **StrataDB** | holds the branched lineage (weights + data + eval results) — fork/merge |
| **Verity** | the *action* fence — deterministically verifies each tool call a deployed agent would execute (external, model-independent) |

## The loop

```
Chisel data  ──►  Lithos train  ──►  EVAL LOOP (fitness)  ──►  keep / branch / abandon
     ▲                                      │
     │                                      ▼
 new data targets  ◄──  Petra: which sources drove which gains, and what's still missing
```

Each turn: Chisel emits a data cut → Lithos trains a checkpoint → **the eval loop scores it
and decides its fate** → Petra attributes the result back to training sources and names the
gaps → those gaps become Chisel's next targets. StrataDB records the whole turn as one
branch so a direction can be forked, compared, and merged or abandoned. The loop iterates
faster than a human lab because the *decision* — is this checkpoint better, and why — is made
by the eval loop + Petra, not a person reading curves.

## The eval loop is the fitness function

This is the load-bearing reference (see `docs/eval-plan.md` and `docs/eval-tir-battery-plan.md`).
Everything else in the loop is machinery; **the eval loop is the selection pressure that
decides which direction survives.** Four things it must be for the flywheel to converge
rather than drift:

1. **Cheap enough to run every iteration.** The keeper suite (parity matrix, judged
   comparisons, quantized edge run) is too expensive to gate an inner loop. The flywheel
   needs a *fast proxy battery* — a subset that produces signal in minutes — run every turn,
   with the full keeper suite reserved for promotion gates. This proxy is a distinct,
   named artifact the loop owns; without it the loop optimizes cost, not capability.
   *(This is the open design item — the eval-plan flags it but the proxy isn't specced yet.)*
2. **On-thesis.** The headline fitness signal is **tool-uplift** — the verified solve-rate
   difference with the sandbox vs without (`eval-tir-battery-plan.md`), reported per
   difficulty tier. It measures the exact thing the product bets on, so optimizing it
   optimizes the product, not a proxy for it. The parity matrix is the longer-horizon
   scoreboard the loop climbs.
3. **Contamination-resistant, or the loop Goodharts its own scoreboard.** An autonomous loop
   that both *chooses* the training data and *grades itself* will overfit its eval set unless
   the eval set is structurally immune. The year-split (train pre-cutoff, eval post-cutoff,
   renewed annually — `eval-plan.md` §5) + the disjoint-pool guard (`assert_disjoint`) are
   what make the fitness signal trustworthy *when the optimizer is a machine*, not just a
   nicety. The public benchmark's canary + content-hash (`eval-tir-battery-plan.md` Part B)
   extend this: the loop cannot silently train on its own yardstick.
4. **Watched, not believed.** `eval-plan.md` principle 8 — score deltas get spot-checked
   against real rollout transcripts before a checkpoint is kept. The battery already
   captures a transcript sample for exactly this; in the loop, that check is Petra's job
   (reward-hacking audit + the "what lit up" channel view), and a failed audit vetoes a
   keep even when the number went up.

## Guardrails (why it doesn't run away)

- **Three fences the optimizer cannot widen** — the loop may only *train on*, *emit*, and
  *act on* what policy permits, and those are three distinct enforcement points, not one:
  the **train-on** fence is **Chisel** (the teacher doctrine — never closed-model targets,
  verifier-gated open-weight teachers); the **emit** fence is Lithos's **decode-policy** (a
  forbidden token is unreachable at decode); the **action** fence is **Verity** (the external,
  model-independent monitor that verifies each resolved tool call before execution). The
  optimizer cannot widen any of them.
- **Fitness is verified, not judged** — executable grading everywhere possible, so the loop
  can't farm a compliant judge. Where a judge is unavoidable it's advisory, never a gate.
- **Every turn is a branch, not a mutation** — StrataDB lineage means a bad direction is
  abandoned, not baked in; nothing is irreversible.
- **Promotion needs the keeper suite + a human** — the inner loop proposes; shipping a
  checkpoint still passes the full battery and a person. Autonomy is in the *search*, not
  the *release*.

## What has to exist before this is worth building

1. The hand-cranked loop running once end-to-end (Chisel → Lithos → eval → Petra → Chisel).
2. **The fast proxy battery** (item 1 above) — the fitness signal the loop can afford.
3. Petra's source-attribution + reward-hacking audit as a callable step, not a manual read.
4. StrataDB branch/merge over (weights + data + eval) as one lineage.

Until then this is a map, not a machine — and the eval loop is the part already being built
(`eval-tir-battery-plan.md`), which is why it's the right place to start.
