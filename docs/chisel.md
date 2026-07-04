# Chisel — Requirements (2026-07-04)

**Chisel cuts rocks into tools: raw data in, verified high-quality data out.**
The Strata stack's data factory — the agentic harness that converts curated raw
corpus into verified problems, reasoning traces, RL environments, and evals.
**Status: candidate product leg of Strata, BUILD PRODUCT-SHAPED NOW, GTM
DEFERRED** — the go/no-go on selling anything waits for the 500M flagship's
parity-frontier benchmarks (the dogfooding proof). Until then Chisel has
exactly one customer: Lithos. Companions: `docs/strata-gpu-hot-tier.md`,
`docs/moho.md` (other legs, recorded in the same pass); doctrine in
`docs/data-construction.md`; internal epics in
`docs/post-training-implementation-plan.md` Wave 4.

## 1. The two framing decisions

**The scarce asset is verified problems, not traces.** A verified problem
(statement + machine-checkable key + provenance) is permanent capital serving
three customers at once — SFT traces, RL tasks, eval items. Traces are a
regenerable derivative: they improve every time teachers improve, and later
when Lithos self-distills. Chisel is therefore a **problem factory with trace
generation as its back half**, not a text-to-traces converter.

**The answer key never comes from the model that writes the trace.** For niche
physics/engineering, teacher hallucination rises exactly where we need data
most; an unverified "convert text to reasoning" harness would train our
differentiation on fluent, subtly-wrong derivations. Key sources, ranked:

1. **Native keys** — mined from the corpus (textbook worked examples,
   end-of-chapter answers, FE/PE prep, olympiad archives).
2. **Tool-computed keys** — the goldmine: *sample parameters first, compute the
   answer deterministically (CoolProp, python-control, pint, sympy/numpy), then
   write the problem statement around it.* Ground truth by construction.
3. **Cross-model majority vote** — last resort only, flagged in provenance.

## 2. Scope

**In scope:** mining, synthesis, teacher trace generation with rejection
sampling, template amplification, the four output contracts (§4), provenance,
and per-domain plug-ins.
**Out of scope:** verification logic (lives in the shared Lithos verifier —
CH-2), model training itself, human labeling (post-gate option), and any
customer-facing surface until the GTM gate opens.

## 3. Functional requirements

**CH-1 · Pipeline stages.** `mine → synthesize → solve → filter → convert`:
extract native problems+keys from curated corpus; synthesize tool-keyed
problems; teachers solve **in our sandbox via TIR**; keep only what verifies
(rejection sampling); convert kept traces to training format (the
rollout→segments path, epic E13).

**CH-2 · One verifier.** Chisel calls the shared Lithos verifier stack (E1
sandbox + verify/verify_batch, E15 dimensional checking). It must never grow
its own checking logic — a second source of truth would let the data filter
drift from the RL reward, silently optimizing two definitions of "correct."

**CH-3 · Domain plug-ins.** A domain (thermo, controls, fluids, circuits, EM,
statics/dynamics, …) is a plug-in: `{key generators, verifier bindings,
template families, taxonomy tags}`. Adding a domain adds no core code. The
taxonomy spine is the FE-exam topic list (engineering breadth) + the standard
physics curriculum ladder. For closed-form domains the plug-in's core asset is
a **formula registry**: each entry = equation, variables/units, *applicability
conditions*, and *common mistakes*. The registry triple-serves — synthesis
substrate (sample params → formula → tool-computed key), applicability
checking (rules contributed to the shared verifier, not a second checker), and
the bait inventory for adversarial variants (CH-11).

**CH-4 · Parameterized templates.** One verified seed problem → a template →
many parameter-instantiated variants, each with a freshly tool-computed key and
independent verification. This is the amplifier that makes niche coverage and
the ~10B-token problems-class gap tractable, and parameter ranges map to
difficulty levels (curriculum for free). Statement paraphrase varies surface
form so templates don't imprint.

**CH-5 · Provenance manifest on every artifact.** Machine-readable, stamped at
creation: source (what corpus material seeded it; license tier), key source
(native / tool@version / vote), teacher (model + license), verifier verdict +
version, template lineage, taxonomy tags, and a per-record **`allowed_use`
gate** (`train | eval | reference_only | exclude`) that **defaults to
`reference_only` when provenance is unclear** — records earn their way into
training, they don't drift in. (Tier semantics come from the sourcing doctrine
in `docs/data-construction.md` — the grey/copyrighted-published tier is valid
for internal training; `allowed_use` gates the record, the tier gates export.) Two enforced consequences: **only
green-tier provenance can ever leave the building** (grey/books-seeded material
is internal-only — the export filter is the `tier` field), and **closed-model
text never becomes a training target** (teachers are open-licensed only;
GPT/Claude may appear in provenance only in build-tool roles). This field-level
enforcement is what makes the output "unimpeachable" — the sellable property.

**CH-6 · Four output contracts.** From one verified bank: **(a) task bank** —
Task JSONL (kind/prompt/answer/level/year) for RLVR + curriculum; **(b) SFT
corpus** — verified traces as segments JSONL → E2 packed shards; **(c) RL
environment** — bank + verifier binding, packaged (problem in, reward out);
**(d) eval set** — held-out splits (`validation`, `held_out_eval`,
`adversarial_eval`) with year-based decontam (`assert_disjoint`).
All four from the same artifacts; no per-output re-verification.
**Family-level split integrity:** all descendants of one parent problem —
template instantiations (CH-4) and adversarial variants (CH-11) — land on the
same side of the train/eval split; near-duplicate and source-family leakage is
tracked. Year-based decontam alone does not cover this.

**CH-7 · Teacher abstraction.** Teachers are pluggable inference backends:
self-hosted open models (the `serve_labeler.sh` H100 + Qwen3-32B vLLM stack is
the proven reference), Qwen3-family models driven *natively in-sandbox* via
`load_qwen3` + `tir_rollout` (the E7 synthesis), and serverless open-model
APIs. Rejection-sampling multiplicity (K per problem) is a per-domain knob —
low pass rates on hard niches are expected and are the point.

**CH-8 · Coverage ledger.** Per-taxonomy-node counts of verified problems,
templates, kept traces, and teacher pass rate. Low-pass-rate nodes are
simultaneously where Lithos can differentiate and where synthesis needs
tool-key help — the ledger is the factory's steering instrument, and later the
product's coverage map.

**CH-9 · Reasoning-quality filters (beyond correctness).** Right answer, wrong
reasoning is real: require dedup across kept traces per problem, cap traces per
problem, and support an optional second-pass process check (weighted lightly —
the verifier remains the gate). Anti-shortcut screens reuse E4's gaming
pre-screen patterns.

**CH-10 · Batch/offline architecture.** Chisel is offline tooling (scripts +
a factory package), resumable and idempotent per item (the `acquire.py`
manifest-gated pattern), never in the training path or the serving path. Every
record carries a lifecycle state (`extracted → normalized → solved → verified →
approved_for_{train,eval} | needs_review | rejected`) so batches resume and
audits reconstruct.

**Storage substrate (DECIDED 2026-07-04): JSONL + R2 manifests now, StrataDB
later, behind a seam.** The eventual goal is Lithos' toolchain on StrataDB (at
least for metadata), but StrataDB is too green today and a solo builder cannot
debug both sides of the fence at once — Chisel's substrate must be *above
suspicion* when a batch fails. Two requirements keep the migration cheap:
(a) all storage access goes through one thin metadata-store interface (no raw
file paths in stage logic); (b) the data model stays **graph-shaped on disk** —
every record has a stable id and typed references (`source_id`, `parent_id`,
`verification_of`, `member_of_dataset`), so lineage edges are explicit fields.
Migration to StrataDB then = a loader script, not a redesign — and Chisel's
lineage graph at scale becomes StrataDB's first real stress-test workload,
adopted deliberately rather than debugged under fire.

**CH-11 · Adversarial variants.** Beyond CH-4's parameter variation (valid
problems at scale), generate variants that test **judgment**: unit-conversion
traps, missing-required-variable (correct behavior = *ask for the missing
information*, not solve), boundary-condition swaps, invalid-assumption setups,
wrong-formula bait (seeded from the registry's common-mistakes inventory).
Each variant records `expected_behavior` (`solve | ask_for_missing_info |
flag_invalid_assumption`) and expected failure mode. These feed the
`adversarial_eval` split and judgment-training data — the engineering-judgment
surface no calculation dataset covers. Honest caveat: `expected_behavior ≠
solve` is not tool-verifiable; these are checked by pattern/judge (build-tool
role) and are **excluded from RL reward banks** unless a deterministic check
exists. Related: registry `common_mistakes` instantiated as realistic wrong
solutions are natural **DPO negatives** (E8 preference pairs).

## 4. Non-functional requirements

- **Unit economics visibility:** cost per verified problem and per kept trace,
  tracked per domain/teacher (the frontier-API vs. self-hosted decision needs
  data, not vibes).
- **Reproducibility:** any artifact regenerable from its manifest (seed, prompt,
  teacher version, sandbox version).
- **Tenancy-readiness (design-only for now):** nothing in the architecture may
  assume factory data and Lithos training data are the same pool — the future
  enterprise mode ("bring your corpus") demands customer material never touches
  our training. Structural, not policy, when that day comes.

## 5. Sequencing and the gate

Chisel v0 **is** Wave 4 built product-shaped: E12 (task banks = the mine stage),
E13 (rollout→segments = the convert stage), E15 (dimensional verification —
pulled early: tool keys are generators, not just checkers), plus the harness
epic (E16, to be added to the plan when factory work starts) for mining agents,
tool-key synthesis, and templates. Order: **mine first** (cheapest, exercises
P0 data as it lands) → tool-key synthesis → the full agentic loop.

**v0 vertical slice = Mechanics of Materials** (axial stress/strain, torsion,
beam bending/deflection, Euler buckling, thin-wall pressure vessels, factor of
safety): closed-form, pint-verifiable with pure algebra (no property-lookup
tools), textbook-dense, diagram-light — the ideal first plug-in.
Diagram-requiring problems are flagged (`diagram_required`) and deferred, v0
ingests markdown/plain text (PDF/HTML extraction later). Thermo is plug-in #2
(exercises CoolProp-class key generators). **v0 Done (falsifiable):** ~100
approved training traces + 25 held-out eval + 25 adversarial-eval items from
the MoM slice, 100% provenance coverage, **100% executable verification for
anything entering an RL/eval bank** (partial coverage is tolerable only for
SFT-only traces), family-split integrity enforced, one JSONL export consumed
end-to-end by the Lithos eval runner (E14).
**GTM gate:** 500M flagship benchmarks ship → publish a bank slice + model card
as the demo → let inbound demand choose the product form (environments hub vs.
factory-as-a-service vs. licensing). No sales motion before proof.

## 6. StrataDB migration workload spec (banked 2026-07-04)

What Chisel demands of StrataDB when the CH-10 seam flips — recorded now so the
billion-scale roadmap can target a real customer instead of a synthetic one.

**Scale, phased:**

| Phase | Records | Edges | Metadata bytes | Verdict |
|---|---|---|---|---|
| v0 (MoM 100/25/25) | 10³–10⁴ | ~10⁴ | MBs | anything works |
| 500M-flagship era | 10⁶–10⁷ | ~10⁷ | 1–10 GB | first migration candidate |
| 3B / product era | 10⁸–10⁹ | 10⁹–5×10⁹ | 0.5–2 TB | the 1B-keys × 1KB roadmap point |

Population is dominated by **trace attempts + verification runs**, not problems
(problems ≈ 10⁷–10⁸ even template-amplified; ~10 attempts kept per problem —
failed attempts are provenance + DPO negatives — × ~1.5 verification runs each,
plus 3–5 lineage edges per record). Consequence: **edges reach 10⁹ before
nodes do** — price edge capacity at ~5× node count.

**The hard rule that makes 1KB values true: metadata in Strata, payloads in
R2.** Trace bodies (2–8 KB) and problem statements live as content-addressed
blobs; the DB holds metadata + typed edges + hashes. This keeps records at
~0.5–1 KB, total ≤ ~2 TB — inside a single local NVMe. **Chisel never forces
StrataDB to go distributed; the embedded single-node thesis survives the
product era.**

**Workload character — forgiving:** batch, append-mostly, single-writer, no
hot rows, relaxed consistency (crash = re-run an idempotent batch),
latency-tolerant (10 ms reads fine). The three dimensions that actually stress
the roadmap, in priority order:

1. **Bulk ingest throughput** — generation runs append millions in a burst;
   target ≥ 50–100k inserts/sec sustained (full 10⁹ rebuild in hours-to-a-day).
2. **Secondary access paths at scale** — the real queries are scans/rollups
   (records by lifecycle state, coverage-ledger rollups by taxonomy node,
   dataset-version assembly by split), not point lookups. Secondary indexing or
   prefix-scan key design is where embedded stores usually hurt first.
3. **Bounded graph walks over ~10⁹ edges** — family-split integrity =
   "all descendants of parent P" (1–3 hops), run over millions of parents
   during an export build; provenance audits walk the full lineage chain.

Framing for the Strata roadmap: Chisel is a **throughput** problem on a
forgiving consistency model; the later R2/Moho tier is a **latency** problem
inside a decode loop. Migrating Chisel first climbs the difficulty curve in the
right order — it is StrataDB's planned first production workload
(CH-10), with the JSONL fallback one config flip away.

## 7. Open questions (for the design doc / post-gate)

Environment packaging standard (what format labs actually consume); how much
process-checking is worth its false-negative rate; per-domain expert review for
the unverifiable tail; pricing shape (per-problem vs. per-domain vs. access);
whether the coverage ledger itself becomes a public leaderboard surface;
a
**Lithos-failure feedback loop** post-v0 (eval failures steer the next mining/
synthesis round via the coverage ledger).
