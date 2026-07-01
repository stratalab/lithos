# Corpus Seed Index — the Catalog of Intent

The bill of materials for the Lithos STEM pretraining corpus (doctrine and rationale: `docs/data-construction.md` §1.5–1.7). **Nothing is acquired that isn't indexed first.** This table is where coverage, gaps, licensing, and cost are measured *before* acquisition spend — and where the sourcing doctrine is enforced mechanically (tier and epoch cap are columns, not vibes).

## Files

- `seed_index.csv` — one row per work / corpus / dump / series. The seed release covers the bulk corpora + the per-domain canon backbone (~190 rows); it grows toward the full canon over time.
- `scripts/validate_seed_index.py` — schema check + coverage report (tokens by domain × tier × priority).

## Schema

| Column | Meaning |
|---|---|
| `id` | stable slug, never reused |
| `kind` | `corpus` (ML-ready dataset) · `dump` (raw public dump) · `series` (multi-volume set) · `book` · `reference` (handbook) · `notes` (lecture-note collections) · `problems` (exam/olympiad/problem-set banks — post-training feedstock, see doc §2.2) |
| `domain` | `code` · `math` · `physics` · `eng` · `chem` · `general` · `xdomain` (the intersections we over-weight: physics-via-code, math-as-program) |
| `subfield` | free text (e.g. `analysis`, `qft`, `controls`) |
| `level` | `intro` · `ug` · `grad` · `research` · `mixed` |
| `title` / `creator` | human-readable identity |
| `canonical_id` | `hf:` dataset id, dump URL, `arxiv:` category — or `isbn:TBD`, filled at acquisition (**never guessed**) |
| `tier` | `green` (public domain / open-licensed / invited distillation) · `grey` (copyrighted-but-published; ingest with caveats) · `mixed` |
| `license_note` | short qualifier (e.g. `official-free-pdf` = copyrighted but the rightsholder distributes it free) |
| `est_tokens` | rough size, `K`/`M`/`B` suffix — planning numbers, not measurements |
| `priority` | `P0` backbone (first acquisition wave) · `P1` second wave · `P2` coverage/optional |
| `epoch_cap` | max training epochs for this work; `4` default for grey works (the §1.5 caveat), `-` = mix-determined |
| `route` | `hf` · `dump` · `arxiv-src` · `free-web` · `pd` · `scrape` · `lawful-copy` |
| `status` | `indexed` → `acquired` → `extracted` → `shipped` |

## Doctrine enforcement notes

- **Grey-tier books route through `lawful-copy`** (purchased/scanned/licensed copies). This is deliberate, not squeamishness: *Bartz v. Anthropic* (N.D. Cal. 2025) drew exactly this line — training on lawfully-acquired books was ruled transformative fair use, while sourcing from pirate libraries was not. The acquisition route, not the training use, is where the legal exposure concentrates. Same doctrine (§1.5), cheaper risk.
- **Grey works get `epoch_cap=4`** and are probed by the regurgitation eval (frozen battery). Green works are capped only by the mix sweep.
- **`official-free-pdf`** in `license_note` marks copyrighted works the rightsholder gives away (Feynman Lectures, PRML, ESL, Sutton–Barto): still grey for epoch-cap purposes, but acquisition is trivially clean via `free-web`.
- **Secret-scanning and PII redaction apply to every tier** — a leaked API key in a public repo is private data.

## How to extend

Add rows; run `python scripts/validate_seed_index.py` (also a pre-commit sanity check). Harvest curation, don't invent it: university syllabi, qualifying-exam reading lists, per-field "bibles", review-article bibliographies — and at scale, the **Wikipedia topic-graph job** (`docs/data-construction.md` §1.7): PPR-expanded topic families whose aggregated References sections yield citation-ranked canon candidates per subfield. When acquiring, fill `canonical_id` with the real ISBN/DOI and advance `status`.
