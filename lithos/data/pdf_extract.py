"""EX-2 — PDF -> canonical, math-aware text via Docling.

The second extractor of the physics/eng ingestion wave (docs/physics-eng-ingestion.md):
the path every book flows through. Docling is layout-aware and, tuned for STEM,
converts display equations to LaTeX and detects tables and code/algorithm blocks —
far better than a plain text-layer dump that mangles math.

Tuning that matters (baked in):
* **OCR off by default** — canon PDFs are usually digitally produced (a real text
  layer); OCR is slow and adds errors. Pass ``ocr=True`` for scanned books.
* **Formula enrichment on** — display equations -> LaTeX (``$...$``). Inline math in
  running prose is still degraded (a Docling limitation) — a known caveat, not a bug.
* **Code enrichment on** — algorithm/pseudocode boxes -> fenced code where detected.

Output is one canonical record per book (the whole markdown), anchored to its
Lithos-Canon entry via ``metadata.source_id`` (Chisel CH-12), so the text is
auditable back to a bibliography row. Per-section segmentation is a later refinement.

Docling runs LOCALLY (self-hosted models, GPU-accelerated) — required for grey-tier
books, which must never be shipped to a cloud extraction service. It is a heavy
optional dependency (the ``pdf`` extra), imported lazily.

Note: Docling can throw a benign ``__del__`` AttributeError at interpreter teardown
(after the work is done). The written ``.jsonl.zst`` + manifest are the source of
truth, not the process exit code (same pattern as the tokenizer build).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zstandard

log = logging.getLogger("lithos.pdf_extract")


def pdf_to_markdown(pdf_path: str | Path, *, ocr: bool = False) -> tuple[str, int, str]:
    """Convert a PDF to math-aware markdown. Returns (markdown, n_pages, docling_version)."""
    import docling
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = ocr                    # digital PDFs have a text layer; OCR only for scans
    opts.do_formula_enrichment = True    # display equations -> LaTeX
    opts.do_code_enrichment = True       # algorithm/pseudocode boxes -> code
    opts.do_table_structure = True
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = conv.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()
    n_pages = len(getattr(result.document, "pages", []) or [])
    return markdown, n_pages, getattr(docling, "__version__", "unknown")


def make_record(markdown: str, *, source_id: str, title: str, domain: str, license: str,
                tier: str, n_pages: int, ocr: bool, docling_version: str) -> dict[str, Any]:
    """Assemble one canonical record for a book, anchored to its Canon entry (CH-12)."""
    return {
        "id": f"pdf:{source_id}",
        "text": markdown,
        "source": "canon",
        "subset": source_id,
        "language": "en",
        "license": license,
        "metadata": {
            "source_id": source_id,      # Lithos Canon anchor -> corpus/seed_index.csv id
            "title": title,
            "domain": domain,
            "tier": tier,
            "pages": n_pages,
            "extractor": "docling",
            "docling_version": docling_version,
            "ocr": ocr,
            "formula_enrichment": True,
        },
    }


def extract_pdf(pdf_path: str | Path, out_dir: str | Path, *, source_id: str, title: str,
                domain: str, license: str, tier: str, ocr: bool = False) -> dict[str, Any]:
    """Extract one book PDF -> ``<out>/<source_id>.jsonl.zst`` + a manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info("[%s] extracting %s (ocr=%s)", source_id, Path(pdf_path).name, ocr)
    markdown, n_pages, dv = pdf_to_markdown(pdf_path, ocr=ocr)
    if not markdown.strip():
        raise RuntimeError(f"{source_id}: docling produced no text")
    record = make_record(markdown, source_id=source_id, title=title, domain=domain,
                         license=license, tier=tier, n_pages=n_pages, ocr=ocr, docling_version=dv)

    out_path = out / f"{source_id}.jsonl.zst"
    with open(out_path, "wb") as fh:
        w = zstandard.ZstdCompressor(level=10).stream_writer(fh)
        w.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        w.close()
    manifest = {
        "source_id": source_id, "title": title, "domain": domain, "tier": tier,
        "source_pdf": Path(pdf_path).name, "pages": n_pages, "chars": len(markdown),
        "math_spans": markdown.count("$") // 2, "code_blocks": markdown.count("```") // 2,
        "extractor": "docling", "docling_version": dv, "ocr": ocr,
        "extracted_at": datetime.now(UTC).isoformat(),
    }
    (out / f"{source_id}.manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("[%s] %d pages -> %d chars, %d math spans, %d code blocks",
             source_id, n_pages, len(markdown), manifest["math_spans"], manifest["code_blocks"])
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", required=True, type=Path, help="the book PDF")
    p.add_argument("--out", required=True, type=Path, help="output dir for <source_id>.jsonl.zst")
    p.add_argument("--source-id", required=True, help="Lithos Canon id (corpus/seed_index.csv) to anchor to")
    p.add_argument("--title", default="", help="book title (else derived from source-id)")
    p.add_argument("--domain", default="xdomain")
    p.add_argument("--license", default="grey")
    p.add_argument("--tier", default="grey")
    p.add_argument("--ocr", action="store_true", help="enable OCR (scanned PDFs only)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    m = extract_pdf(args.pdf, args.out, source_id=args.source_id, title=args.title or args.source_id,
                    domain=args.domain, license=args.license, tier=args.tier, ocr=args.ocr)
    print(json.dumps(m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
