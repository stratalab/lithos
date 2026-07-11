# Lithos documentation

## Product & plan
- [PRD](prd.md) — requirements
- [Implementation plan](implementation-plan.md) — the phased build plan
- [v1 on a Qwen base](v1-on-qwen.md) — **DECIDED**: v1 drops from-scratch pretraining, post-trains Qwen3 0.6–8B; attestation is *scoped to post-training*; from-scratch returns as the v2 attestation demonstrator

## Architecture & design
- [Composite model layer](composite-model-layer.md) — the absorption test; which of R1/R2/TIR/decode-policy are moats and which are scaffolding
- [Composite instrumentation](composite-instrumentation.md) — the measurement apparatus: three capture points, three tables, C0–C5 as queries
- [Composite plan](composite-plan.md) — **DECIDED**: mechanisms above the token stream have positive evidence, below it negative; R1 → in-context RAG, R2 first, the six acceptance gates
- [Composite plate](composite-plate.html) — the architecture drawn as a stratigraphic cross-section (rev B, post-sweep). Self-contained HTML; open in a browser
- [C0 spec](c0-spec.md) — the R1 gate experiment, post-literature-sweep (absorb first, measure the residual; superseded on architecture by composite-plan.md)
- [Architecture audit](architecture-audit.md) — Lithos vs Qwen3, component-by-component + the applied default fixes
- [Tokenizer](tokenizer.md) — the 32k STEM BPE design
- [TIR format](tir-format.md) — tool-integrated-reasoning tokens + loss masking (read before writing TIR data)
- [Evaluation plan](eval-plan.md) — the frozen battery + parity-frontier anchors
- [Quality classifiers](quality-classifiers.md) — the owned quality model + labeling
- [Data construction](data-construction.md) — corpus doctrine

## Post-training
- [Implementation plan](post-training-implementation-plan.md)
- [Review](post-training-review.md) · [Review 2](post-training-review-2.md)
- [Pressure test](posttrain-pressure-test.md) — the E1–E8 end-to-end shakedown + findings

## Corpus & ingestion
- [Physics/eng ingestion](physics-eng-ingestion.md) — the physics/eng wave + extractor suite
- [STEM pretraining corpus research](stem-pretraining-corpus-research.md)
- [Remote training](remote-training.md) — the rented-cluster workflow

## Strata ecosystem (other legs + handoffs)
- [Chisel](chisel.md) — the data factory (Lithos's one customer)
- [Chisel → Lithos: producer migration](chisel-producer-migration.md) — what data code moves to Chisel
- [Chisel → Lithos: R2 output contract](chisel-lithos-r2-contract.md) — the storage tiers + what Chisel writes to R2
- [Chisel F7 response](chisel-f7-response.md) — Lithos-side decisions on the generation back-half (teacher doctrine, verifier seam, code harness, TIR schema, splits)
- [Chisel → Lithos: the tier gate](chisel-tier-gate.md) — acquisition (not license) decides what enters the weights; the gate follows the **gradient**, so restricted text may be a prompt but never a target
- [Lithos → Petra: provenance](petra-provenance-lithos.md) — docindex schema, frozen dedup, TracIn reconstructibility
- [Lithos → Petra: composite attribution](petra-composite-attribution.md) — the counterfactual `scope` defect, `provenance_channel`, and the attribution-calibration offer
- [Strata GPU hot tier](strata-gpu-hot-tier.md)
- [Moho](moho.md) — the kernel/boundary leg
