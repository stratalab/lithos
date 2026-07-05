# Lithos documentation

## Product & plan
- [PRD](prd.md) — requirements
- [Implementation plan](implementation-plan.md) — the phased build plan

## Architecture & design
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
- [Strata GPU hot tier](strata-gpu-hot-tier.md)
- [Moho](moho.md) — the kernel/boundary leg
