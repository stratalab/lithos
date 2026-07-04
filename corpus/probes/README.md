# Tokenizer probe sets

Small, checked-in fixtures for `scripts/eval_tokenizer.py` (design rationale in
the tokenizer-quality discussion; tiers 1–2 of the evaluation). One JSONL per
file, `{"text": ...}` records.

- `general/math/code/physics/engineering.jsonl` — per-domain compression probes.
  These give a fast baseline and a regression diff across tokenizer retrains;
  for *statistically meaningful* fertility and vocab-usage numbers, pass large
  held-out samples with `--sample name=path.jsonl` instead.
- `adversarial.jsonl` — roundtrip torture (unicode, whitespace, bidi, emoji).
  Byte-level BPE must decode all of these losslessly; any failure means a
  normalizer/decoder setting is corrupting input.
- `segmentation.jsonl` — short `{"text", "note"}` snippets (numbers, LaTeX,
  code idioms, units) whose token splits are printed for eyeballing and diffed
  across retrains.

Editing rules: keep files small (this is a fixture, not a corpus); never paste
text from eval benchmarks (decontamination applies here too); when adding a
domain, name the file after the domain — the filename becomes the report row.

The definitive tier-3 comparison is a per-domain bits-per-byte ablation on
trained models, not anything in this directory.
