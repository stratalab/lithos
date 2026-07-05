# Public GitHub Repositories of Reusable STEM Learning Content: A Categorized Inventory for Open STEM Foundation-Model Training

## TL;DR
- **Math and CS are extremely well-served** by public GitHub/HF corpora spanning every training stage (base pretrain → RLVR); the highest-value single assets are proof-pile-2 (55B tokens), OpenWebMath (14.7B tokens), MathPile (~9.5B tokens), NuminaMath-1.5 (~896K verifiable problems), and the code-contest trio (APPS/CodeContests/TACO).
- **Physics and engineering are a real gap**: there is no pretraining-scale (billions-of-tokens) standalone physics or engineering text corpus, and no large (>10K), text-only, numeric-answer engineering RLVR set. These domains must be built by scraping university course repos, textbook mirrors (OpenStax/LibreTexts), and arXiv (physics + eess) yourself.
- **Recommended pipeline**: use math/CS mega-corpora for base pretrain; layer OpenStax/LibreTexts + MIT-OCW course repos + arXiv physics/eng subsets for continued pretrain; use awesome-list link-harvesting to expand textbook coverage; and rely on NuminaMath + code-contest datasets for RLVR, treating physics/engineering RLVR as an in-house build.

## Key Findings
1. **The open STEM data ecosystem is heavily math- and code-skewed.** Every pretraining-scale corpus that exists (proof-pile-2, OpenWebMath, MathPile, AutoMathText, FineMath) is math-centric; physics/CS content appears only as a minority "spillover" fraction inside them. OpenWebMath's own documentation notes it spans "mathematics, physics, statistics, computer science, and more," but the majority of documents are directly mathematical.
2. **Curated "awesome" lists are the best discovery layer, not content themselves.** They contain thousands of links to downloadable textbooks/notes, but the actual bytes live on external sites; they are indexes to harvest, not corpora to ingest directly.
3. **University course repos (MIT OCW especially) are the richest structured physics/engineering source**, pairing lecture notes + problem sets + solutions, but they are scattered across many small personal repos rather than one canonical dump.
4. **Formal-proof libraries (Lean mathlib, Isabelle AFP, Coq, Metamath) are a distinctive, high-value asset** for verifiable reasoning that most teams under-use.
5. **RLVR is turnkey for math and code, DIY for physics/engineering.** NuminaMath, DeepScaleR, GSM8K, MATH, APPS, CodeContests, and TACO give roughly two million verifiable items for math+code combined; physics has only small (≤1K) evaluation benchmarks and engineering essentially none at scale.

## Details

### Category 1 — Curated "Awesome" Lists (discovery/link-harvesting layer; not corpora themselves)

| Repo | Domain(s) | Content | Approx size | Format | Stars / freshness | Best stage |
|---|---|---|---|---|---|---|
| EbookFoundation/free-programming-books | CS (all langs) | The canonical free-books index; thousands of book links | ~390K stars; CC-BY-4.0 | Markdown lists | ~390K stars; actively maintained | Discovery → base pretrain (after harvest) |
| rossant/awesome-math | Math | Curated math books, lecture notes, papers | Hundreds of links | Markdown | Popular, maintained | Discovery |
| awesomelistsio/awesome-mathematics (brandonhimpfen) | Math | Theory, datasets, tools, learning materials | Hundreds of links | Markdown | Maintained | Discovery |
| Empia/awesome-math-1, FGDBTKD/awesome-math, cyc790/awesome-math, FrankLoud/awesome-math | Math | Forks/mirrors of rossant list | Hundreds of links each | Markdown | Static forks | Discovery |
| ngmsonn/Awesome_Mathematics | Math | "List of awesome lists" on math (incl. Cambridge/NYU course notes) | Medium | Markdown | Static | Discovery |
| wbierbower/awesome-physics | Physics | Physics software + concepts | Medium | Markdown | Older but useful | Discovery |
| imrehg/awesome-physics | Physics | Research-oriented physics resources | Medium | Markdown | Older | Discovery |
| mikeroyal/Physics-Guide | Physics | Broad physics topic guide w/ links | Large | Markdown | Maintained | Discovery |
| donovanglover/awesome-physics | Physics | Study resources | Small | Markdown | Static | Discovery |
| SwapneelM/awesome-particle-physics-for-non-physicists | Physics (particle/QFT) | Intro particle physics resources | Small | Markdown | Static | Discovery |
| iris-hep/awesome-hep | Physics (HEP) | HEP software | Medium | Markdown | Maintained | Discovery |
| A-make/awesome-control-theory | Eng (controls) | Control-theory learning resources | 731 stars, 93 forks | Markdown | Maintained | Discovery |
| UlisseMini/awesome-control, robotsorcerer/awesome-neurocontrol | Eng (controls) | Control-theory / ML-control resources | Small-medium | Markdown | Static | Discovery |
| shaoanlu/awesome-control-engineering-online-materials | Eng (controls) | Free control-engineering course materials | Medium | Markdown | Static | Discovery |
| kitspace/awesome-electronics | Eng (EE) | Electronics resources/tools/tutorials | Large | Markdown | Maintained | Discovery |
| samyk/awesome-electronics, gautamsharma0095/awesome-electronics | Eng (EE) | Electronics resources | Large | Markdown | Static | Discovery |
| m2n037/awesome-mecheng | Eng (mech) | Mechanical-engineering books, notes, exams, numerical-methods texts | Medium-large | Markdown | Static | Discovery |
| awesomelistsio/awesome-mechanical-engineering | Eng (mech) | ME tools/resources/courses (incl. MIT OCW, NPTEL) | Medium | Markdown | Maintained | Discovery |
| rg3l3dr/awesome-hardware | Eng (hardware) | Hardware-engineering design/tools index | Medium | Markdown | Static | Discovery |
| chennachaos/awesome-FEM4CFD | Eng (FEM/CFD) | FEM-for-CFD learning resources + books | Small-medium | Markdown | Maintained | Discovery |
| lento234/awesome-fluid-dynamics | Eng (fluids) | Fluid-dynamics code + learning | Medium | Markdown | Maintained | Discovery |
| thw1021/Code4CFD | Eng (CFD) | CFD code repositories index | Medium | Markdown | Maintained | Discovery |
| tkoyama010/awesome-finite-elements | Eng (FEA) | FEA software for structural eng | Medium | Markdown | Maintained | Discovery |
| ossu/computer-science | CS | Full self-taught CS curriculum (course links) | ~32K+ stars | Markdown | Actively maintained | Discovery |
| lnishan/awesome-competitive-programming | CS | Competitive-programming resources | Large | Markdown | Maintained | Discovery |
| amao0o0/awesome-AI-Math-Datasets | Math | Index of open math LLM datasets/benchmarks | Medium | Markdown | Actively updated | Discovery (dataset index) |
| lmmentel/awesome-python-chemistry | Chemistry | Python chemistry packages | Large | Markdown | Maintained | Discovery |
| kjappelbaum/awesome-chemistry-datasets | Chemistry | ML chemistry datasets index (incl. LibreText, OpenStax Chem 2e) | Medium | Markdown | Maintained | Discovery |
| blaiszik/awesome-matchem-datasets | Chem/Materials | Materials+chem datasets incl. TextbookReasoning (650K reasoning Qs from 12K textbooks), MegaScience (1.25M instances) | Medium | Markdown | Actively updated | Discovery |
| hsiaoyi0504/awesome-cheminformatics, mlederbauer/awesome-learning-digital-chemistry | Chemistry | Cheminformatics + digital-chem learning | Medium | Markdown | Maintained | Discovery |

### Category 2 — Direct Textbook / Open-Textbook Mirror Repos

| Repo | Domain(s) | Content | Approx size | Format | Freshness | Best stage |
|---|---|---|---|---|---|---|
| philschatz/textbooks (+ philschatz/book-updater) | All STEM | OpenStax textbooks ported to GitHub/GH-Pages (bio, chem, physics, stats, calc) | Dozens of books | Markdown/HTML | Static (mirror) | Continued pretrain |
| openstax org repos (osbooks-*: college-physics, university-physics-bundle, chemistry, organic-chemistry, introductory-statistics, etc.) | Physics, Chem, Math, Bio | Canonical CC-BY OpenStax book sources | ~383 org repos | CNXML/XML | Active | Continued pretrain |
| VinPu/Textbooks | Physics/Math/Eng | Undergrad textbook PDF collection (Griffiths E&M + solutions, Riley-Hobson-Bence, McIntyre QM) | Dozens of PDFs | PDF | Static | Continued pretrain |
| LibreTexts (libretexts.org, exportable per book) | All STEM | Open-access textbook network (chem, phys, eng, bio, math) | Very large | HTML/PDF export | Active | Continued pretrain |
| CK-12 FlexBooks, OpenStax CNX, Open Textbook Library (external, GitHub-linked) | All STEM | Open-textbook repositories | Large | HTML/PDF | Active | Continued pretrain |

### Category 3 — University Course-Material Repos

| Repo | Domain(s) | Content | Approx size | Format | Freshness | Best stage |
|---|---|---|---|---|---|---|
| ocw.mit.edu (8.01/8.02/8.04 physics; 2.xx mech eng; 6.xx EE; controls/thermo/fluids) | Physics, Eng, CS | Lecture notes + problem sets + solutions + exams; downloadable course packages | 2,500+ courses | PDF | Active (site) | Continued pretrain + SFT |
| Seangottarun/MIT-OCW-Notes | Physics/CS | LaTeX notes + pset solutions (incl. 8.04 Quantum Physics I) | Small | LaTeX/PDF | Static | SFT |
| ramanakshay/mitocw | CS/Math | Solutions + notes for OCW courses | Small | Mixed | Static | SFT |
| goepigen/8.01SC-Classical-Mechanics | Physics | Completed 8.01SC psets + weekly notes | 38 stars, small | Mixed/Maple | Static | SFT |
| knzhou.github.io/handouts (Kevin Zhou) | Physics (olympiad/undergrad) | ~1,000 tough physics problems w/ full solutions across all subfields | Large PDF set | PDF | Maintained | SFT + RLVR seed |

### Category 4 — Problem Sets / Solutions / Exercise Corpora

| Repo | Domain(s) | Content | Approx size | Format | Freshness | Best stage |
|---|---|---|---|---|---|---|
| AnupamShaw/QuantumMechanicsGriffiths | Physics (QM) | Mathematica solutions to Griffiths QM computer problems | Small | Mathematica | Static | SFT |
| stemjock.com (Griffiths QM/E&M, Callister, etc.) | Physics/Materials | Worked textbook solutions | Medium | HTML/PDF | Maintained | SFT |
| leduckhai/Awesome-Competitive-Programming | CS | Must-know CP problems w/ solutions + visualizations | Medium | Markdown/code | Maintained | SFT |
| sojolrana/Competitive-Programming-Solutions | CS | Codeforces/LeetCode/AtCoder solutions by algorithm | Medium | C++ | Maintained | SFT/RLVR |
| kunal-kushwaha/Competitive-Programming-Resources | CS | CP + system-design resources | Medium | Markdown | Maintained | Discovery/SFT |
| Winged-Coders (CodeForces-CodeChef-CodeJam-HackerRank-LeetCode) | CS | Aggregated multi-platform solutions | Medium | Multi-lang | Maintained | SFT/RLVR |
| ipho-unofficial.org (open source) + ipho.olimpicos.net | Physics (olympiad) | Past IPhO problems + solutions archive (1967–present) | 50+ years | HTML/PDF | Maintained | SFT + RLVR seed |

### Category 5 — Pretraining-Scale STEM Datasets & Formal-Proof Repos

| Repo | Domain(s) | Content | Approx size | Format | Freshness | Best stage |
|---|---|---|---|---|---|---|
| EleutherAI/proof-pile-2 (+ zhangir-azerbayev/proof-pile) | Math (+phys/CS spillover) | arxiv (29B) + open-web-math (15B) + algebraic-stack (11B); created to train Llemma 7B/34B | **55B tokens** | text/LaTeX | 2023, stable | Base pretrain |
| keirp/OpenWebMath | Math (+phys/CS) | Filtered mathematical web text; extracted from 200B+ Common Crawl HTML down to 6.3M docs across 130k+ domains | **14.7B tokens, 6.3M docs** | text/LaTeX | 2023 (arXiv:2310.06786) | Base pretrain |
| GAIR-NLP/MathPile | Math | Textbooks (~0.19B tokens), arXiv, Wikipedia, ProofWiki, StackExchange, web | **~9.5B tokens** | text/LaTeX | 2024 (NeurIPS D&B; arXiv:2312.17120) | Base pretrain |
| allenai/peS2o | All science | ~40M CC open-access academic papers cleaned/filtered for pretraining; derived from S2ORC (unfiltered 11.3M full-text papers / 46.9B tokens as of Jan 2023) | ~40M papers | text | v2, maintained (~183 stars) | Base pretrain |
| togethercomputer/RedPajama-Data (arXiv subset) | All science | LaTeX arXiv full-text (preamble/comments/bib removed) | 28B tokens (arXiv slice); 1.2T total | LaTeX text | Maintained | Base pretrain |
| leanprover-community/mathlib4 | Math (formal) | Lean 4 formalized math library (quadratic reciprocity, ZFC model, Lebesgue measure, etc.) | Very large | Lean | Very active | Continued pretrain / verifiable |
| InternLM/Lean-GitHub (dataset) + InternLM/InternLM-Math | Math (formal) | Compiled Lean proofs from 237 GitHub repos | 0.131B tokens | Lean/JSONL | 2024 (arXiv:2407.17227) | Verifiable / continued pretrain |
| Isabelle AFP (Archive of Formal Proofs) | Math (formal) | Extensive Isabelle proof library | Very large | Isabelle | Active | Continued pretrain / verifiable |
| Coq Mathematical Components / Metamath set.mm / HOL Light / Coquelicot | Math (formal) | Formal proof libraries (set.mm in proof-pile w/ 10% held out) | Large | Coq/Metamath/HOL | Active | Continued pretrain / verifiable |
| wellecks/naturalproofs (+ naturalproofs-gen) | Math | 32K theorem statements/proofs, 14K definitions, 2K other (ProofWiki/Stacks/Trench RA/Stein NT) | ~48K items | JSON/LaTeX | 2021 (arXiv:2104.01112) | SFT / continued pretrain |
| EleutherAI/hendrycks_math, FineMath, AutoMathText (per amao0o0 index) | Math | Additional math corpora (FineMath 34B/54B tokens; AutoMathText ~200GB) | Multi-B tokens | text/LaTeX | Recent | Base pretrain |

*Note (flagged gap): No physics-only or engineering-only equivalent to proof-pile-2/OpenWebMath/MathPile exists. Physics/engineering content is present only as a minority fraction inside math and general-science corpora (peS2o, RedPajama arXiv).*

### Category 6 — arXiv-adjacent Corpora & Tooling

| Repo | Domain(s) | Content | Approx size | Format | Freshness | Best stage |
|---|---|---|---|---|---|---|
| potamides/arxiv-latex-extract | All (phys/eng/math) | Bulk-extract LaTeX from arXiv archives (uses archive.org mirror; RedPajama-based cleanup) | Tool | Python | Maintained | Base pretrain (build tool) |
| mattbierbaum/arxiv-public-datasets | All | Scripts to pull arXiv PDFs + fulltext + citation graph | Tool; ~1.37M docs / ~64GB text / ~11B words | Python | Maintained | Base pretrain (build tool) |
| arXiv Bulk Data (S3 requester-pays / Kaggle) | All (hep-th, cond-mat, gr-qc, quant-ph, eess) | Full arXiv source/PDF; ~1.1TB PDFs | Whole arXiv (~$100 S3) | LaTeX/PDF | Monthly | Base/continued pretrain |

### Category 7 — RLVR-Relevant Verifiable-Answer Datasets

| Repo | Domain(s) | Content | Approx size | Format | Freshness | Best stage |
|---|---|---|---|---|---|---|
| AI-MO/NuminaMath-1.5 | Math | ~900K competition-level problems w/ verifiable answer + problem_type/question_type metadata (numeric or "proof"/"notfound") | **896K problems** | parquet | Jan 2025, Apache-2.0 | RLVR |
| AI-MO/NuminaMath-CoT | Math | CoT solution pairs (aggregates GSM8K/MATH/AMC/AIME/CN-K12/Orca-Math/Olympiads) | **859,494 train** | parquet | Nov 2024 | SFT/RLVR (contamination risk) |
| agentica-org/DeepScaleR-Preview-Dataset (+ agentica-org/rllm) | Math | RL math train set (AIME 1984–2023, AMC pre-2023, Omni-MATH, STILL) | ~40K problem-answer pairs | JSON | Feb 2025, MIT | RLVR (GRPO via Verl fork) |
| openai/grade-school-math (GSM8K) | Math | Grade-school word problems, numeric final answer after `####`, 2–8 steps | 8.5K (7,473/1,319) | JSONL | Stable, MIT | RLVR |
| hendrycks/math (MATH) | Math | Competition problems, boxed answers, full step solutions, 7 subjects × 5 levels | 12,500 (7,500/5,000) | JSON | Stable, MIT | RLVR + SFT |
| hendrycks/apps (codeparrot/apps) | CS (code) | Programming problems w/ unit tests + ~232K solutions | 10,000 (5,000/5,000) | JSON | Stable, ~450 stars | RLVR |
| google-deepmind/code_contests | CS (code) | Competitive problems w/ comprehensive hidden test suites; ~1.5M solutions | ~13,610 | Riegeli | Stable, ~2K stars | RLVR |
| FlagOpen/TACO (BAAI/TACO) | CS (code) | Largest algorithmic code set (aggregates APPS+CodeContests+CodeChef+CF+GfG+HR); ~1.55M solutions | 26,443 (25,443/1,000) | JSON | Apache-2.0 | RLVR |
| LiveCodeBench/LiveCodeBench | CS (code) | Contamination-controlled live code-gen w/ hidden tests (avg 59+/problem) + release-date tags | ~400–1,055 (windowed) | JSON | Continuously updated | RLVR/eval |
| google-research/mbpp | CS (code) | Entry-level Python problems + 3 tests each (427 sanitized) | 974 | JSONL | Stable | RLVR |
| TIGER-AI-Lab/TheoremQA | Math/Phys/EE&CS | Theorem-application QA, numeric answers, WolframAlpha-evaluated | 800 (350+ theorems) | JSON | 2023 (EMNLP) | RLVR/eval |
| OpenBMB/OlympiadBench (Hothan/OlympiadBench) | Math + Physics | Olympiad problems, bilingual, multimodal; OE numeric + TP proof subsets | 8,476 | JSON+images | 2024 (ACL) | RLVR/eval |
| Eureka-Lab/PHYBench (phybench-official/phybench) | Physics | 500 original problems HS→Olympiad difficulty, symbolic answers scored via Expression Edit Distance | 500 | LaTeX | 2025, MIT (arXiv:2504.16074); best model 36.9% vs 61.9% human | RLVR/eval |
| CMPhysBench/CMPhysBench | Physics (cond-mat) | Graduate condensed-matter calculation problems, SEED metric | 520 | JSON | 2025 | eval |
| mandyyyyii/scibench | Physics/Chem/Math | College textbook numeric problems w/ solutions | ~695 open | JSON | 2023–24 (ICML; arXiv:2307.10635) | RLVR/eval |
| Jun-Kai-Zhang/MatSciBench | Materials | College materials-science problems, numeric+units, 5% tolerance eval | 1,340 | JSON | 2025 (arXiv:2510.12171) | eval |
| SoM-1K (som-1k.github.io) | Eng (strength of materials) | Strength-of-materials problems, multimodal (text + schematics) | 1,065 | Multimodal | 2025 (arXiv:2509.21079); best 56.6% | eval |

### Category 8 — Simulation / Code-Adjacent Engineering Corpora

| Repo | Domain(s) | Content | Approx size | Format | Freshness | Best stage |
|---|---|---|---|---|---|---|
| OpenFOAM/OpenFOAM-dev | Eng (CFD) | Full CFD package + extensive tutorials | Very large | C++ | Active | Continued pretrain + code SFT |
| su2code/SU2 | Eng (multiphysics/CFD) | Open-source multiphysics simulation + design suite | Large | C++/Python | Active | Continued pretrain + code SFT |
| KratosMultiphysics/Kratos | Eng (multiphysics) | Parallel multi-disciplinary simulation framework | Large | C++/Python | Active | Continued pretrain + code SFT |
| FEniCS/dolfinx + firedrakeproject/firedrake | Eng (FEM) | Automated PDE solving via FEM, w/ tutorials | Large | Python/C++ | Active | Continued pretrain + code SFT |
| PyFR/PyFR, xcompact3d/Incompact3d, Nek5000/Nek5000, DedalusProject/dedalus | Eng (CFD/spectral) | High-order flow / spectral PDE solvers w/ docs | Large | Python/Fortran | Active | Continued pretrain + code SFT |
| FluidityProject/fluidity | Eng (CFD) | Adaptive-mesh multiphase CFD + Python interface | Large | Fortran/Python | Active | Continued pretrain + code SFT |
| djeada/Computational-Fluid-Dynamics-CFD-Resources | Eng (CFD) | CFD notes, tutorials, learning curricula | Medium | Markdown/Python | Maintained | Continued pretrain |
| CalculiX examples (Kraska), MFEM, Gridap.jl, scikit-fem | Eng (FEA) | FEA solvers + worked examples/tutorials | Medium-large | Multi-lang | Active | Continued pretrain + code SFT |

## Top 15–20 Highest-Value Repos Overall (ranked by size × quality × structure)

1. **EleutherAI/proof-pile-2** (55B tokens) — the single biggest ready-made math+science pretraining corpus; base pretrain backbone.
2. **keirp/OpenWebMath** (14.7B tokens, 6.3M docs) — highest-quality math web text; base pretrain.
3. **GAIR-NLP/MathPile** (~9.5B tokens) — most diverse math corpus (textbooks + arXiv + forums); base pretrain.
4. **allenai/peS2o** (~40M papers) — broadest all-science academic corpus, the best lever for physics/chem/eng breadth at pretrain scale.
5. **AI-MO/NuminaMath-1.5** (896K verifiable problems) — the premier math RLVR/SFT set.
6. **AI-MO/NuminaMath-CoT** (859K CoT pairs) — premier math SFT set (watch contamination).
7. **FlagOpen/TACO** (26K code problems, ~1.55M solutions) — largest execution-verifiable code RLVR set.
8. **google-deepmind/code_contests** (~13.6K) — cleanest hidden-test code reward signal.
9. **hendrycks/apps** (10K) — standard code RLVR set.
10. **hendrycks/math (MATH)** (12.5K) — canonical competition-math RLVR + SFT.
11. **openai/grade-school-math (GSM8K)** (8.5K) — canonical arithmetic-reasoning RLVR.
12. **leanprover-community/mathlib4** — flagship formal-proof corpus for verifiable reasoning.
13. **MIT OpenCourseWare course packages** (2,500+ courses) — richest structured physics/engineering lecture-notes+psets+solutions source.
14. **openstax osbooks-* + philschatz/textbooks** — cleanest CC-BY textbook sources across physics/chem/math/bio.
15. **togethercomputer/RedPajama-Data (arXiv subset)** (28B tokens) — turnkey arXiv LaTeX for physics/eng breadth.
16. **EbookFoundation/free-programming-books** (~390K stars) — the master discovery index for CS books.
17. **ossu/computer-science** — best-structured CS curriculum index for course harvesting.
18. **OpenBMB/OlympiadBench** (8,476, math+physics) — the largest verifiable set that actually includes physics.
19. **wellecks/naturalproofs** (~48K theorems/proofs) — best natural-language proof SFT/eval set.
20. **OpenFOAM / SU2 / FEniCS** (engineering simulation code + docs) — best paired theory-and-code engineering assets for continued pretrain and tool-use SFT.

## Recommendations

**Stage 1 — Base pretrain corpus (500M & 1B from scratch; 3B/7B continued on Qwen3).** Ingest the math/science mega-corpora immediately: proof-pile-2 (55B), OpenWebMath (14.7B), MathPile (9.5B), peS2o (~40M papers), and a fresh arXiv LaTeX pull (via potamides/arxiv-latex-extract or mattbierbaum/arxiv-public-datasets) restricted to physics (hep-th, cond-mat, gr-qc, quant-ph), math, CS, and eess. This is the fastest path to a strong quantitative-reasoning base. *Threshold to escalate:* if physics/eng eval (PHYBench, SciBench physics, OlympiadBench physics-OE) lags math eval by more than ~15 absolute points, add more physics/eng continued-pretrain text before spending further compute.

**Stage 2 — Continued pretrain (domain balancing, where you close the physics/eng gap).** Harvest the "awesome" lists programmatically to pull the actual textbook/lecture-note bytes, prioritizing OpenStax osbooks-* (college/university physics, chemistry) and LibreTexts exports, MIT OCW course packages (8.xx physics; 2.xx/6.xx engineering; controls/thermo/fluids/circuits), and the engineering simulation repos' docs/tutorials (OpenFOAM, SU2, FEniCS, Kratos). There is no shortcut corpus here, so budget real engineering effort for scraping and cleaning.

**Stage 3 — SFT (instruction-style QA).** Use NuminaMath-CoT, MATH solutions, naturalproofs, competitive-programming solution repos (leduckhai, sojolrana, Winged-Coders), and worked textbook-solution collections (Griffiths solutions, Kevin Zhou handouts, IPhO archive). Reformat OCW lecture-note + problem-set + solution triples into instruction pairs — this is the highest-leverage way to convert your Stage-2 physics/eng text into instruction data.

**Stage 4 — RLVR (verifiable answers).** Turnkey for math (NuminaMath-1.5 ~896K, DeepScaleR ~40K, GSM8K, MATH) and code (APPS, CodeContests, TACO, LiveCodeBench, MBPP — all execution-verifiable). For physics, use the small verifiable benchmarks (PHYBench, OlympiadBench physics-OE, SciBench, TheoremQA) as seed + eval, but plan to synthesize your own physics numeric-answer set (extract numeric endpoints from OCW/olympiad solutions and verify with a CAS). For engineering, plan to build a verifiable set in-house (numeric answers with unit checking on circuit-analysis, statics/dynamics, thermodynamics-cycle problems) — no public at-scale set exists.

**Stage 5 (optional) — GRPO.** Reuse the same verifiable reward sources; the code datasets with hidden test suites (CodeContests, TACO) provide the cleanest, lowest-noise reward signal, followed by CAS-checked math answers.

## Caveats
- **Physics/engineering pretraining-scale corpora do not exist as standalone assets** — a confirmed gap. Every billion-token corpus is math-centric with physics/CS as minority spillover. Physics/eng breadth must come from peS2o + arXiv (physics/eess) + textbook/course harvesting.
- **Engineering RLVR is essentially unserved**: the largest verifiable engineering benchmarks (MatSciBench 1,340; SoM-1K 1,065; SciBench ~695) are small, frequently multimodal/diagram-dependent, and evaluation-only rather than training-scale. Treat an in-house engineering verifiable set as a required build, not a download.
- **Contamination risk**: NuminaMath-CoT aggregates GSM8K/MATH/AMC/AIME; using it in training will contaminate those benchmarks. Deduplicate against eval sets (your pipeline handles dedup, but flag benchmark overlap explicitly).
- **"Awesome" lists are indexes, not content** — their value is the outbound links; the actual bytes must be fetched from external hosts, and link rot is common.
- **Formal-proof libraries need extraction tooling** (LeanDojo, mm-extract, IsarStep) to convert into human-readable/training-ready text — raw Lean/Coq source is not directly usable as natural-language reasoning data.
- Star counts and last-commit dates are approximate; verify at ingestion time. A few very recent (2025–2026) physics/engineering benchmarks surfaced with future-dated arXiv IDs; their existence should be treated as provisional until confirmed on live repos, though the engineering-gap conclusion holds regardless.