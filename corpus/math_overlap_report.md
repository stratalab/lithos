# Math-corpus overlap matrix (sample-based estimates)

`overlap(A→B)` = est. fraction of A's documents with a duplicate in B.
Estimator inverts counterpart-inclusion probability (see lithos/data/overlap.py);
assumes shuffled samples and ~1 counterpart per dup — read as upper-ish bounds.

| A → B | n_A | N_B | url | exact-text | near-dup | raw near matches | notes |
|---|---|---|---|---|---|---|---|
| openwebmath → finemath | 200000 | 21,405,610 | 50.2% | 21.0% | 45.4% | 848 |  |
| finemath → openwebmath | 200000 | 6,315,233 | 14.8% | 6.2% | 13.4% | 847 |  |
| openwebmath → nemotron-cc-math | 200000 | 52,000,000 | — | 0.0% | 0.0% | 0 | no shared url field — url overlap skipped; N_nemotron-cc-math is approximate — estimates toward it scale with it |
| nemotron-cc-math → openwebmath | 200000 | 6,315,233 | — | 0.0% | 0.0% | 0 | no shared url field — url overlap skipped; N_nemotron-cc-math is approximate — estimates toward it scale with it |
| openwebmath → megamath | 200000 | 121,000,000 | 100.0% | 1.8% | 42.7% | 141 | N_megamath is approximate — estimates toward it scale with it |
| megamath → openwebmath | 200000 | 6,315,233 | 16.0% | 0.1% | 2.3% | 143 | N_megamath is approximate — estimates toward it scale with it |
| finemath → nemotron-cc-math | 200000 | 52,000,000 | — | 0.0% | 0.0% | 0 | no shared url field — url overlap skipped; N_nemotron-cc-math is approximate — estimates toward it scale with it |
| nemotron-cc-math → finemath | 200000 | 21,405,610 | — | 0.0% | 0.0% | 0 | no shared url field — url overlap skipped; N_nemotron-cc-math is approximate — estimates toward it scale with it |
| finemath → megamath | 200000 | 121,000,000 | 100.0% | 0.6% | 64.1% | 212 | N_megamath is approximate — estimates toward it scale with it |
| megamath → finemath | 200000 | 21,405,610 | 46.3% | 0.1% | 11.6% | 216 | N_megamath is approximate — estimates toward it scale with it |
| nemotron-cc-math → megamath | 200000 | 121,000,000 | — | 0.0% | 0.3% | 1 | no shared url field — url overlap skipped; only 1 matches — wide error bars, treat as noisy; N_nemotron-cc-math is approximate — estimates toward it scale with it; N_megamath is approximate — estimates toward it scale with it |
| megamath → nemotron-cc-math | 200000 | 52,000,000 | — | 0.0% | 0.1% | 1 | no shared url field — url overlap skipped; only 1 matches — wide error bars, treat as noisy; N_nemotron-cc-math is approximate — estimates toward it scale with it; N_megamath is approximate — estimates toward it scale with it |

## Addendum — Nemotron-CC-Math's zeros are extraction divergence, not absence

Nemotron-CC-Math carries no URL/WARC-offset join key (its `metadata` has
`warc_filename` + `warc_id` only — and, tellingly, **per-record `finemath_scores`**:
NVIDIA filtered with FineMath's own classifier, so the corpora are provably
siblings over the same crawl population). Its text is a Lynx + Phi-4 cleaned,
LaTeX-standardized re-extraction — divergent enough that Jaccard-0.8 near-dup
matching sees nothing. Relaxed-threshold sweep (cached 200k samples):

| Jaccard ≥ | nem↔owm raw | nem↔finemath raw | control owm↔finemath raw |
|---|---|---|---|
| 0.7 | 1 | 2 | 966 |
| 0.5 | 29 | 45 | 1,401 |
| 0.35 | 138 | 311 | 2,356 |

At J≥0.35 that is ~15–20% cross-presence, and true page-level overlap is higher
(a rewrite easily pushes same-page Jaccard below 0.35). **Read the main table's
nemotron zeros as "different text, overlapping pages," not "disjoint data."**

## Decisions this implies (validated by ablation per doc §0.2 before shipping)

1. **The web-math corpora are one heavily-shared page population**, not four
   independent sources. OWM is essentially inside MegaMath (URL 100%-capped)
   and half-inside FineMath; naive concatenation multi-epochs the intersection.
2. **Cross-source dedup must key on URL + MinHash** (exact-text is nearly
   blind: same pages, different extractions) — and for Nemotron pairs, MinHash
   at a *relaxed* threshold (~0.5) since URL is unavailable.
3. **Dedup is best-of-N selection, not deletion**: prefer Nemotron's
   equation-preserving extraction where variants collide (it also carries
   FineMath scores for free thresholding); MegaMath-web fills coverage after
   dedup-vs-Nemotron; FineMath/OWM contribute their unique remainder.
4. **Epoch accounting must treat the four as substitutes**, not additive
   supply — the unique web-math pool is far below the naive 460B-token sum.

Method + estimator: `lithos/data/overlap.py` (counterpart-inclusion inversion;
200k shuffled-stream samples per corpus; N_megamath/N_nemotron approximate).
