# Per-Domain Quality Classifiers — Stage-4 Design

Model-based quality filtering is the single biggest documented pretraining lever
(DCLM; FineWeb-Edu). This doc designs ours: what "quality" means per domain,
where labels come from, the classifier architecture, and how thresholds get
chosen. Companion to `docs/data-construction.md` (pipeline stage 4, §1.4
recipes) and `docs/eval-plan.md` (the ablation harness renders all verdicts).

## 0. Principles

1. **Downstream decides, not label agreement.** The classifier's eval is
   whether thresholding improves **per-domain bpb on the 100M rig** (the
   FineWeb methodology: many small threshold ablations). AUC vs labels is a
   development signal only.
2. **Free labels first.** A third of the corpus arrives pre-scored (see §1);
   the labeling budget goes only to unscored sources.
3. **Executable signals beat opinions** where they exist. Code that parses,
   LaTeX that compiles, units that balance — objective, free, and no
   classifier drift. Rubric scores fill in where execution can't reach.
4. **Form-aware thresholds.** A prose rubric scores reference tables (CRC,
   datasheets) at ~0; they're indexed deliberately (`form=reference`, 4% of
   the budget). Thresholds are set per (domain, form); some forms are exempt.
5. **Open labeler.** Qwen3-32B (or current open equivalent) writes the labels.
   Closed models as labelers are doctrine-grey-OK, but there's no reason to
   carry the asterisk — and it's the path NVIDIA proved for its own classifiers.
6. **Goodhart rule (banked, restated):** never gate *synthetic* data with the
   same classifier family a generation loop could optimize against. Synthetic's
   gate is the **verifier** (correctness), not the quality scorer.

## 1. Free-label inventory (spend nothing here)

| Source | Carried score | Covers |
|---|---|---|
| FineWeb-Edu | edu classifier score (already thresholded in `data/quality.py`) | general glue |
| FineMath | `score` / `int_score` | web-math |
| Nemotron-CC-Math | `finemath_scores` + `nemocurator_scores` per record | web-math |
| MegaMath-web | `math_score` | web-math |

**Math and general are pre-scored.** Our labeling budget concentrates on:
physics/eng sources (arXiv, Stack Exchange, patents, datasheets, books), code
beyond license/heuristic filters, and xdomain intersections.

## 2. What "quality" means, per domain

**Math — reuse FineMath's rubric** (reasoning density: worked derivations vs
formula name-dropping). Solved problem; carried scores + the same rubric for
unscored math sources (arXiv math, ProofWiki).

**Physics / engineering — the rubric we own** (0–5, "technical training
value"). The axis generic edu-classifiers miss: *quantitative* substance vs
prose about a technical topic.
- **0** — no technical content; spam/boilerplate.
- **1** — mentions technical topics without substance (news blurbs, product
  marketing, course listings).
- **2** — descriptive/popular explanation; correct but no quantitative
  content (no equations, units, or derivations).
- **3** — substantive exposition with *some* quantitative content (equations,
  numbers with units, code), but fragmentary or shallow.
- **4** — solid technical material: derivations or worked examples or design
  reasoning with quantities and units; coherent and self-contained.
- **5** — textbook/reference grade: rigorous derivation, worked problems,
  precise units, pedagogically complete.

**Code — executability first, rubric second.**
- *Executable tier (objective, free):* parses (tree-sitter per language), not
  truncated/minified/generated/vendored, imports plausible; run-ability where
  cheap. This is the primary gate (§1.3's unique lever).
- *Rubric tier (0–5)* for educational value of code+prose documents
  (tutorials, notebooks, answered questions): does the code teach — named
  variables, comments that explain why, prose interleaved with working code?

**xdomain** (notebooks, physics-via-code, math-as-program): score with the
physics/eng rubric **plus** the code executable tier; both must clear.

## 3. Labeling protocol

- **Prompt shape:** rubric + first ~2k tokens of the doc → output = one
  integer + one-line justification (the justification measurably improves
  label quality; parse the integer). Temperature 0.
- **Pilot:** 5k docs per domain from the local overlap/dev samples.
  Double-label 500 (self-agreement); the user hand-checks ~100 — rubric bugs
  surface there, not in aggregate stats. Iterate the rubric once, then freeze.
- **Scale:** 100–500k labels/domain. Budget: ~1.5k tokens each → ~10⁹ tokens
  through a 32B-q4 — days on the 5090 (prefill-bound) or a ~$10–20 rented-H100
  burst. Labels + rubric version recorded in a manifest (provenance §0.4).

## 4. Classifier architecture: fastText v0 → embedder only on proven ceiling

- **v0 — fastText** (DCLM's choice): trains in minutes on CPU, scores the
  whole corpus in hours, instant iteration. One model per domain rubric.
- **v1 — small embedder + regression head** (FineWeb-Edu's choice), adopted
  *per domain* only where fastText demonstrably ceilings on held-out labels.
- **Open empirical question (cheap to test, don't assume):** heavy
  LaTeX/notation may embed poorly on natural-text embedders while fastText's
  character n-grams cope — decide per domain from the pilot, not from taste.

## 5. Threshold selection = ablation

Thresholds are **outputs of the 100M mix-sweep machinery**, not tuning
intuitions: sweep score cutoffs per (domain, form) → build variant corpora →
proxy runs → per-domain bpb (`evals/ablation.py` already orchestrates this
loop). Expect the FineWeb-Edu shape: aggressive cutoffs win on quality-dense
slices; reference/QA forms need laxer or exempt gates. Every kept/dropped
ratio lands in the corpus manifest.

## 6. Pipeline integration

`data/quality.py` already thresholds carried scores via
`DocumentSource.quality_field`. Extension: a `QualityScorer` seam
(`score(doc) -> float`) with three implementations — carried-score passthrough,
fastText, embedder — selected per source in the corpus config. Scores and
thresholds recorded per-shard in manifests; the executable code tier runs as a
stage-3.5 filter before scoring (cheap rejections first).

## 7. Risks

- **Rare-form deletion** — the classifier silently killing reference/problems
  forms the budget requires → form-aware thresholds (§0.4) + budget report
  watches supply per form after filtering.
- **Label noise from rubric ambiguity** → pilot + one iteration + freeze;
  double-label agreement reported alongside every classifier version.
- **Notation-blind embedders** → §4's per-domain empirical check.
- **Goodhart via synthetic** → §0.6; verifier gates synthetic, full stop.

## 8. Sequencing (all pre-GPU except the last)

1. ✅ Carried-score thresholding (FineWeb-Edu path live in `data/quality.py`).
2. Wire carried scores for FineMath / Nemotron-CC-Math / MegaMath at ingestion.
3. Write rubric prompts (§2) + code executable tier (tree-sitter parse gate).
4. Pilot: 5k/domain labels on local samples → agreement stats → freeze rubrics.
5. fastText v0 per domain; held-out label agreement reported.
6. Threshold ablations on the 100M rig (first GPU consumer) → ship cutoffs.
