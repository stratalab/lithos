"""OpenStax ingestion — CC-BY textbook CNXML -> canonical Q&A-free documents.

The first `scrape`-route source of the physics/eng wave (docs/physics-eng-ingestion.md).
OpenStax ships its ~60 CC-BY textbooks as GitHub `osbooks-*` repos (per-module
`modules/<id>/index.cnxml`, MathML for math). We fetch each book as a **single
tarball pinned to the approved commit** (one request per book — no GitHub API
rate-limit problem), read every `index.cnxml` member, run it through EX-1
(`markup_text.cnxml_to_text`), and emit canonical `.jsonl.zst`.

Scope: the STEM books that map to our domains — physics, chemistry, math. OpenStax
has no engineering texts (those are the grey canon), and life-sciences books are
outside the four target domains, so both are excluded.

Network + tar are stdlib; lxml (via EX-1) and zstandard come from the `data` extra.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zstandard

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lithos.data.markup_text import cnxml_to_text  # noqa: E402

log = logging.getLogger("lithos.openstax")

APPROVED_LIST = ("https://raw.githubusercontent.com/openstax/"
                 "content-manager-approved-books/main/approved-book-list.json")
CODELOAD = "https://codeload.github.com/openstax/{repo}/tar.gz/{ref}"
LICENSE = "cc-by-4.0"
UA = {"User-Agent": "lithos-corpus-builder/0.1 (+https://github.com/stratalab/lithos)"}

# Curated STEM books (repo -> our domain). Physics + chemistry + math; life
# sciences and (absent) engineering excluded. Slugs live inside the bundles.
STEM_BOOKS: dict[str, str] = {
    "osbooks-university-physics-bundle": "physics",
    "osbooks-college-physics-bundle": "physics",
    "osbooks-physics": "physics",
    "osbooks-astronomy": "physics",
    "osbooks-chemistry-bundle": "chem",
    "osbooks-organic-chemistry": "chem",
    "osbooks-calculus-bundle": "math",
    "osbooks-college-algebra-bundle": "math",
    "osbooks-contemporary-mathematics": "math",
    "osbooks-prealgebra-bundle": "math",
    "osbooks-statistics": "math",
    "osbooks-introductory-statistics-bundle": "math",
}


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def latest_commits() -> dict[str, str]:
    """repo -> latest approved commit SHA (for reproducible pinning)."""
    data = json.loads(_get(APPROVED_LIST))
    out: dict[str, str] = {}
    for b in data.get("approved_books", []):
        versions = b.get("versions") or []
        if versions:
            out[b["repository_name"]] = versions[-1]["commit_sha"]
    return out


def _module_id(member_name: str) -> str:
    # ".../modules/<id>/index.cnxml" -> "<id>"
    parts = member_name.split("/")
    return parts[parts.index("modules") + 1] if "modules" in parts else member_name


def iter_records_from_tar(fileobj: Any, *, repo: str, commit: str, domain: str,
                          ) -> Iterator[dict[str, Any]]:
    """Yield one canonical record per non-empty ``index.cnxml`` in a book tarball."""
    with tarfile.open(fileobj=fileobj, mode="r:gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith("/index.cnxml")
                    and "/modules/" in member.name):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            text = cnxml_to_text(f.read())
            if not text.strip():
                continue
            mod = _module_id(member.name)
            yield {
                "id": f"openstax:{repo}:{mod}",
                "text": text,
                "source": "openstax",
                "subset": repo,
                "language": "en",
                "license": LICENSE,
                "metadata": {
                    "repo": repo, "commit": commit, "module": mod, "domain": domain,
                    "url": f"https://github.com/openstax/{repo}",
                },
            }


def extract_book(repo: str, commit: str, domain: str, out_dir: Path) -> dict[str, Any]:
    """Fetch a book tarball (pinned to ``commit``) and write ``<repo>.jsonl.zst``."""
    ref = commit or "refs/heads/main"
    log.info("[%s] fetching tarball @ %s", repo, ref[:10])
    with tempfile.TemporaryFile() as tmp:
        tmp.write(_get(CODELOAD.format(repo=repo, ref=ref)))
        tmp.seek(0)
        out_path = out_dir / f"{repo}.jsonl.zst"
        n = 0
        with open(out_path, "wb") as fh:
            w = zstandard.ZstdCompressor(level=10).stream_writer(fh)
            for rec in iter_records_from_tar(tmp, repo=repo, commit=commit, domain=domain):
                w.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
                n += 1
            w.close()
    if n == 0:
        out_path.unlink(missing_ok=True)
    log.info("[%s] %d modules -> %s", repo, n, out_path.name if n else "(empty, skipped)")
    return {"repo": repo, "commit": commit, "domain": domain, "modules": n}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path, help="output dir for <repo>.jsonl.zst")
    p.add_argument("--only", action="append", help="restrict to these repo(s) (repeatable)")
    p.add_argument("--limit-books", type=int, default=None, help="cap number of books (testing)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    commits = latest_commits()
    books = {r: d for r, d in STEM_BOOKS.items() if not args.only or r in args.only}
    repos = list(books)[: args.limit_books] if args.limit_books else list(books)
    log.info("extracting %d OpenStax book(s) -> %s", len(repos), args.out)

    sources: list[dict[str, Any]] = []
    failures: list[str] = []
    for repo in repos:
        try:
            sources.append(extract_book(repo, commits.get(repo, ""), books[repo], args.out))
        except Exception as e:  # keep going; report at the end
            log.error("[%s] FAILED: %s", repo, e)
            failures.append(repo)
    (args.out / "_sources.json").write_text(json.dumps(
        {"source": "openstax", "license": LICENSE, "fetched_at": datetime.now(UTC).isoformat(),
         "books": sources}, indent=2))
    if failures:
        log.error("failed: %s", ", ".join(failures))
        return 1
    log.info("done — %d books, %d modules", len(sources), sum(s["modules"] for s in sources))
    return 0


if __name__ == "__main__":
    sys.exit(main())
