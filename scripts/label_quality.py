#!/usr/bin/env python3
"""Label document quality with an open LLM (docs/quality-classifiers.md §3).

Talks to any OpenAI-compatible endpoint (vLLM / llama.cpp / ollama serving
Qwen3-32B or similar). Pilot flow: sample docs → label → write labels.jsonl +
score histogram; --second-pass relabels a subset at temperature 0.7 for
stability stats (temp-0 double labels are trivially identical).

Typical:
  # doc sources: local JSONL (text field) or HF streaming
  uv run python scripts/label_quality.py --domain physics-eng \\
      --hf allenai/peS2o --n 5000 \\
      --endpoint http://localhost:8000/v1 --model qwen3-32b \\
      --out data/labels/pes2o-physics-eng.jsonl --second-pass 500

  # inspect prompts without an endpoint
  uv run python scripts/label_quality.py --domain code --jsonl docs.jsonl \\
      --n 3 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lithos.data.labeling import (
    LabelRecord,
    agreement,
    build_prompt,
    parse_label,
    score_histogram,
)
from lithos.data.overlap import TEXT_FIELD_CANDIDATES, get_field

log = logging.getLogger("label")
RUBRICS = Path(__file__).resolve().parent.parent / "configs" / "quality" / "rubrics.yaml"


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_hf(spec: str) -> Iterator[dict]:
    """Stream docs from `repo`, `repo:config`, or `repo::data_dir` (shuffled)."""
    from datasets import load_dataset

    repo, config, data_dir = spec, None, None
    if "::" in spec:
        repo, data_dir = spec.split("::", 1)
    elif ":" in spec:
        repo, config = spec.split(":", 1)
    ds = load_dataset(repo, name=config, data_dir=data_dir, split="train", streaming=True)
    yield from ds.shuffle(seed=7, buffer_size=10_000)


def collect_docs(args) -> list[tuple[str, str, dict]]:
    """[(doc_id, text, extra)] — extra carries --carry-field values."""
    source = iter_jsonl(args.jsonl) if args.jsonl else iter_hf(args.hf)
    docs: list[tuple[str, str, dict]] = []
    for rec in source:
        text = next(
            (t for f in TEXT_FIELD_CANDIDATES if isinstance(t := get_field(rec, f), str)), None
        )
        if not text or len(text) < args.min_chars:
            continue
        doc_id = str(rec.get("id") or hashlib.sha1(text.encode()).hexdigest()[:16])
        extra = {f: get_field(rec, f) for f in (args.carry_field or []) if get_field(rec, f) is not None}
        docs.append((doc_id, text, extra))
        if len(docs) >= args.n:
            break
    return docs


class ChatEndpoint:
    """Minimal OpenAI-compatible /chat/completions client (stdlib only)."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout

    def generate(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "temperature": temperature,
             "max_tokens": 100}
        ).encode()
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.load(r)
        return data["choices"][0]["message"]["content"]


def label_docs(
    docs: list[tuple[str, str, dict]], rubric_cfg: dict, args, *, temperature: float = 0.0
) -> list[LabelRecord]:
    endpoint = ChatEndpoint(args.endpoint, args.model)
    dom = rubric_cfg["domains"][args.domain]
    records: list[LabelRecord] = []
    failures = 0
    for doc_id, text, extra in tqdm(docs, desc=f"label:{args.domain}", unit="doc"):
        messages = build_prompt(dom["rubric"], rubric_cfg["response_format"], text)
        if args.no_think:  # Qwen3 soft switch: suppress thinking mode for speed
            messages[0]["content"] += " /no_think"
        try:
            response = endpoint.generate(messages, temperature=temperature)
        except Exception as e:  # transient endpoint errors shouldn't kill a run
            log.warning("[%s] endpoint error: %s", doc_id, e)
            failures += 1
            continue
        parsed = parse_label(response)
        if parsed is None:
            log.warning("[%s] malformed response: %r", doc_id, response[:120])
            failures += 1
            continue
        score, why = parsed
        records.append(LabelRecord(
            doc_id=doc_id, domain=args.domain, rubric_version=int(rubric_cfg["version"]),
            score=score, justification=why, labeler=args.model,
            source=args.jsonl.name if args.jsonl else args.hf, extra=extra,
        ))
    if failures:
        log.warning("%d/%d docs failed", failures, len(docs))
    return records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=Path)
    src.add_argument("--hf", help="repo, repo:config, or repo::data_dir")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--endpoint", default="http://localhost:8000/v1")
    p.add_argument("--model", default="qwen3-32b")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--second-pass", type=int, default=0,
                   help="relabel first K docs at temp 0.7 for stability stats")
    p.add_argument("--carry-field", action="append",
                   help="copy source fields into records (e.g. score, for correlation)")
    p.add_argument("--save-texts", action="store_true", default=True,
                   help="write <out>.texts.jsonl alongside labels (classifier training input)")
    p.add_argument("--no-save-texts", dest="save_texts", action="store_false")
    p.add_argument("--no-think", action="store_true", default=True,
                   help="append Qwen3 /no_think soft switch (default on)")
    p.add_argument("--think", dest="no_think", action="store_false")
    p.add_argument("--dry-run", action="store_true", help="print prompts, no endpoint calls")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rubric_cfg = yaml.safe_load(RUBRICS.read_text())
    if args.domain not in rubric_cfg["domains"]:
        p.error(f"unknown domain {args.domain!r}; have {list(rubric_cfg['domains'])}")

    docs = collect_docs(args)
    log.info("collected %d docs", len(docs))

    if args.dry_run:
        dom = rubric_cfg["domains"][args.domain]
        for doc_id, text, _extra in docs:
            msgs = build_prompt(dom["rubric"], rubric_cfg["response_format"], text)
            print(f"=== {doc_id} ===\n[system] {msgs[0]['content'][:200]}...\n"
                  f"[user] {msgs[1]['content'][:400]}...\n")
        return 0

    records = label_docs(docs, rubric_cfg, args)
    out = args.out or Path(f"data/labels/{args.domain}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_texts:
        tpath = out.with_suffix(".texts.jsonl")
        with open(tpath, "w") as f:
            for doc_id, text, _ in docs:
                f.write(json.dumps({"id": doc_id, "text": text}) + "\n")
        log.info("texts → %s (classifier training needs these)", tpath)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_json()) + "\n")
    log.info("%d labels → %s", len(records), out)
    print("score histogram:", score_histogram([r.score for r in records]))

    if args.second_pass > 0:
        subset = docs[: args.second_pass]
        first_by_id = {r.doc_id: r.score for r in records}
        second = label_docs(subset, rubric_cfg, args, temperature=0.7)
        pairs = [(first_by_id[r.doc_id], r.score) for r in second if r.doc_id in first_by_id]
        if pairs:
            stats = agreement([a for a, _ in pairs], [b for _, b in pairs])
            print("stability (temp0 vs temp0.7):", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
