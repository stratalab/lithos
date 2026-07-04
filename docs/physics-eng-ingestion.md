# Physics + Engineering ingestion wave — scoping (2026-07-04)

**The next document-ingestion wave, prioritized ahead of the code wave (P1) per
the 2026-07-04 decision.** P0 proved the *easy* layer — point `acquire.py` at HF,
get math + web. The inventory (`/data/corpus-staging/inventory.json`) then
quantified what the dataset-layer thesis always assumed: the free HF datasets
over-deliver the two slices that were free (math 54%, glue 46%) and deliver
~nothing of the three that aren't — **physics, engineering, and code together are
≈0.2% of P0 tokens.** This wave attacks the physics/eng gap, which is also the
product's differentiator (the "applied physics that computes" identity, §1.8) and
the edge audience (EE/firmware/robotics). Companion docs: `docs/data-construction.md`
(the funnel + §1.8 discipline targeting), `docs/chisel.md` (verified synthetic —
the *other* answer to the physics/eng gap, arriving later).

## 1. The one reframe: this wave is extract-bound, not download-bound

P0 was download-bound (the bytes sat in clean HF Parquet; the work was moving
them). Physics/eng is **extract-bound**: the value is locked inside PDFs, LaTeX,
HTML, and patent XML that nobody has cleaned for us. Two consequences:

- **The bill of materials is already built.** `corpus/seed_index.csv` (223 rows)
  already carries the physics/eng canon (Feynman, Landau–Lifshitz, Griffiths,
  Jackson, Purcell; the EE/eng canon — Sedra–Smith, Razavi, Horowitz–Hill, Pozar,
  Sze, Nise/Ogata control, Beer–Johnston, Cengel, Incropera, Callister) plus the
  corpus-scale sources (`pes2o`, `arxiv-bulk-src`, `uspto-patents`, `openstax`,
  `libretexts`, `doab-otl`, `standards-slice`, `datasheets-appnotes`). Cataloging
  is **not** the work.
- **The work is routes + extractors.** Most physics/eng sources use acquisition
  routes `acquire.py` does not yet support (`scrape`, `arxiv-src`, `lawful-copy`,
  `pd`), and produce formats the pipeline cannot yet read (PDF, LaTeX, patent XML,
  textbook HTML). The critical path is the **extractor suite** (§4) and the
  **route additions** (§5) — not more downloading.

## 2. Scope

**In:** acquiring and extracting physics + engineering *raw documents* into the
canonical record schema, and the route/extractor tooling that requires.
**Out:** the code wave (P1 — Stack v2 etc.), verified synthetic (Chisel), and the
downstream funnel/mix/tokenizer (those consume this wave's output, unchanged).
**Targeting:** the settled §1.8 disciplines — **EE tier-1a** (electronics/electrical/
computer eng), the **mechanical engineering-science core** (statics/dynamics,
thermo, fluids, heat transfer, materials), **controls as the crown** (most
TIR-friendly), aerospace/chemical riding along, civil/industrial indexed-not-targeted.
Not "all of engineering."

## 3. Source landscape (already indexed → route → extractor → phase → tier)

| seed_index id(s) | what | route | needs extractor | phase | tier |
|---|---|---|---|---|---|
| `pes2o` | arXiv + PMC full-text papers (cleaned) | hf | none | **A** | green |
| `openstax` | ~60 CC-BY textbooks (physics/chem/math) | scrape | HTML/PDF | B | green |
| `libretexts`, `doab-otl` | CC STEM textbook libraries (incl. Engineering) | scrape | HTML/PDF | B | green |
| free-web notes (`david-tong-notes`, `feynman-lectures`, `astrom-murray`, `sethna-statmech`, MIT OCW) | author/official-free lecture notes & texts | free-web | HTML/PDF | B | mixed |
| `nasa-ntrs`, `doe-osti`, `nist-pubs` *(new)* | public-domain gov technical reports | pd | PDF | B | green |
| `uspto-patents` | USPTO full-text (STEM slice) | dump | patent-XML | C | green |
| `arxiv-bulk-src` | arXiv LaTeX source (physics/eess beyond pes2o) | arxiv-src | LaTeX | C | grey |
| `datasheets-appnotes` | TI/ADI/ST/NXP/Microchip datasheets + app-notes (EE crown) | scrape | PDF (table-heavy) | C | grey |
| physics/eng **canon** (~157 `book`/`series` rows) | the grey textbook canon | lawful-copy | PDF/EPUB | C | grey |
| `standards-slice` | IEEE/ISO public specs + abstracts | lawful-copy | PDF | C | grey |

## 4. The extractor suite (the critical path — build in this order)

Each is a reusable `lithos/data/` extractor (library + CLI + tests), mirroring the
StackExchange extractor (`.7z`→canonical JSONL) already shipped. Output is always
the canonical record `{id, text, source, subset, license, metadata}`.

- **EX-1 · HTML→text.** trafilatura/resiliparse-based, math-aware (keep `<code>`,
  MathJax/LaTeX spans). Unlocks OpenStax web, LibreTexts, DOAB, free-web notes,
  MIT OCW. *Highest leverage — the largest fraction of Phase B.*
- **EX-2 · PDF→text (math-aware).** Marker/Nougat/GROBID. Unlocks gov reports,
  the textbook canon, datasheets, standards. *The big one; the canon and the EE
  crown both wait on it.* Datasheets are table-heavy — a known-hard sub-case.
- **EX-3 · LaTeX→text.** arXiv source (`\section`/math preserved). Unlocks
  targeted arXiv physics/eess beyond pes2o. Source beats PDF OCR (doc §1.3).
- **EX-4 · patent-XML→text.** USPTO bulk / Google Patents schema → claims + spec.
- **EX-5 · datasheet parse.** EX-2 plus table/spec-block structure (the applied-EE
  content lives in the tables). The crown; hardest; last.

## 5. Acquisition-route additions to `acquire.py`

The driver supports `hf` and `dump`. This wave needs, roughly in Phase order:

- **`free-web` / `pd`** — fetch author/official-free and public-domain URLs
  (single files or small crawls). Simplest; unlocks lecture notes + gov reports.
- **`scrape`** — bounded, polite crawl/API harvest per source manifest (OpenStax
  CNXML, LibreTexts, DOAB, datasheet portals). Per-source adapters.
- **`arxiv-src`** — arXiv S3 requester-pays bulk source tarballs (doc §1.3).
- **`lawful-copy`** — per-work acquisition of the grey canon; the manifest records
  provenance, and §7's grey controls (epoch-cap + dedup + regurgitation-eval) are
  enforced downstream. Gated behind the canon being index-prioritized (it is).

## 6. Sequencing and gates

- **Phase A — now, zero new tooling:** `pes2o` (hf). Moves physics/eng from 0.1%
  to a real academic slice immediately. In the `physeng` wave in `acquisition.yaml`.
- **Phase B — build EX-1 + EX-2, then `free-web`/`pd`/`scrape`:** the green
  foundation — OpenStax, LibreTexts, DOAB, gov reports, free-web notes. CC/PD, no
  doctrine friction, high-quality expository text (prime annealing-set material).
- **Phase C — build EX-3/EX-4/EX-5 + `arxiv-src`/`lawful-copy`:** targeted arXiv
  physics, USPTO patents, the EE datasheets/app-notes crown, and the grey canon
  (with §7 controls). The longest pole and the deepest differentiation.

**Gate:** nothing here blocks or is blocked by the 100M mix-sweeps (P0 already
suffices to start those on the GPU). This is CPU-local extractor work, runnable in
parallel. Verified synthetic (Chisel) is the complementary fill for the same
disciplines and stays on its own gate (500M flagship).

## 7. Doctrine screening

- **Green (unrestricted, exportable):** `pes2o` (OA subset), OpenStax/LibreTexts/
  DOAB (CC), gov reports + patents (US-gov / public domain), most free-web notes.
- **Grey (ingest with caveats):** the textbook canon (`lawful-copy`), `standards-slice`,
  `arxiv-bulk-src` (per-paper author licenses). Grey handling per the settled
  doctrine — aggressive **dedup**, per-work **epoch caps** (already a column in
  `seed_index.csv`; the validator requires it on grey books), and the
  **regurgitation eval** in the frozen battery. Grey content is ingestible but
  **never exportable** (the `tier` column is the export filter).
- **Universal:** provenance manifests, PII/secret-scanning, decontamination —
  unchanged from the funnel.

## 8. §1.8 discipline mapping (so the wave targets, not boils the ocean)

- **EE tier-1a (crown supply):** `datasheets-appnotes`, `electronics-se`, canon
  (Sedra–Smith, Razavi, Horowitz–Hill, Pozar, Sze, Rabaey, Weste–Harris), Cortex-M.
- **Mechanical core:** canon (Beer–Johnston, Hibbeler, Shigley, Cengel, Incropera,
  White/Munson, Callister/Ashby), Marks'/Roark's handbooks, FE/PE prep.
- **Controls (the crown, most TIR-friendly):** Nise, Ogata, Astrom–Murray
  (free-web), Modern Robotics — `python-control`/`scipy.signal` make nearly every
  problem executable, so this slice also feeds Chisel nearly for free.
- **Physics spine:** the physics canon + `pes2o`/`arxiv-bulk-src` + gov reports.
- **Ride-along (Tier 2):** aerospace (Anderson, Curtis orbital), chemical (BSL
  transport, Fogler). **Index-only (Tier 3):** civil, industrial.

## 9. Open questions (for the build)

PDF extractor choice (Marker vs Nougat vs GROBID) and its math fidelity vs cost;
whether datasheet tables are worth EX-5 or better left to Chisel's structured
mining; how aggressively to crawl LibreTexts (thousands of pages) vs. take its PDF
exports; arXiv physics de-dup against pes2o (overlap is large); the canon
acquisition mechanics and how the regurgitation-eval probe list is seeded from the
grey `book`/`series` rows.
