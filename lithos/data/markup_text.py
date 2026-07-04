"""EX-1 — structured markup (HTML / CNXML) -> clean, math-aware text.

The first extractor of the physics/eng ingestion wave (docs/physics-eng-ingestion.md).
One vocabulary-driven tree walker serves both:

* **HTML** — LibreTexts, gov reports, free-web lecture notes.
* **CNXML** — OpenStax's source format (its ~60 CC-BY textbooks ship as CNXML with
  MathML for math, no LaTeX annotation), so structure and math survive extraction
  instead of being scraped out of rendered pages.

Output preserves document structure (headings, paragraphs, lists, code) and, most
importantly for a STEM corpus, **math**: MathML is converted to inline LaTeX
(``v_{0}``, ``\\frac{a}{b}``) rather than dropped or dumped as tag soup. Uses an
existing LaTeX ``<annotation>`` when present (common in MathJax HTML), else converts
presentation MathML directly (OpenStax's case).

lxml is imported lazily so importing this module never requires the `data` extra.
"""

from __future__ import annotations

import re
from typing import Any

# Element roles the walker understands.
SKIP, HEADING, BLOCK, INLINE, LIST_ITEM, CODE_BLOCK, CODE_INLINE, MATH = (
    "skip", "heading", "block", "inline", "list_item", "code_block", "code_inline", "math"
)

# tag (local name, lowercased) -> role. Headings carry a level via HEADING_LEVEL.
HTML_VOCAB: dict[str, str] = {
    "script": SKIP, "style": SKIP, "nav": SKIP, "header": SKIP, "footer": SKIP,
    "aside": SKIP, "noscript": SKIP, "form": SKIP, "button": SKIP, "svg": SKIP,
    "h1": HEADING, "h2": HEADING, "h3": HEADING, "h4": HEADING, "h5": HEADING, "h6": HEADING,
    "p": BLOCK, "div": BLOCK, "section": BLOCK, "article": BLOCK, "blockquote": BLOCK,
    "figcaption": BLOCK, "tr": BLOCK, "br": BLOCK, "hr": BLOCK,
    "li": LIST_ITEM,
    "pre": CODE_BLOCK, "code": CODE_INLINE,
    "math": MATH,
}
HTML_LEVEL = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# CNXML (OpenStax): http://cnx.rice.edu/cnxml; math in the MathML namespace.
CNXML_VOCAB: dict[str, str] = {
    "title": HEADING,
    "para": BLOCK, "section": BLOCK, "note": BLOCK, "example": BLOCK,
    "exercise": BLOCK, "problem": BLOCK, "solution": BLOCK, "equation": BLOCK,
    "caption": BLOCK, "commentary": BLOCK, "rule": BLOCK, "statement": BLOCK,
    "item": LIST_ITEM,
    "code": CODE_INLINE, "preformat": CODE_BLOCK,
    "math": MATH,
    # dropped: figures/media/images (no text value), metadata, glossary cross-refs
    "figure": SKIP, "media": SKIP, "image": SKIP, "metadata": SKIP, "label": SKIP,
}

_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def _localname(tag: Any) -> str | None:
    """Local, lowercased tag name; None for comments/PIs (tag is not a str)."""
    if not isinstance(tag, str):
        return None
    return tag.rsplit("}", 1)[-1].lower()


# --------------------------------------------------------------------------- #
# MathML -> inline LaTeX-ish text
# --------------------------------------------------------------------------- #

def _mml_children(el: Any) -> list[str]:
    return [_mathml_to_text(c) for c in el]


def _mathml_to_text(el: Any) -> str:
    """Convert (presentation) MathML to compact inline LaTeX-ish text.

    Handles the elements that cover essentially all textbook math; unknown
    constructs fall back to concatenating their children, so nothing is dropped.
    """
    name = _localname(el.tag)
    if name is None:
        return ""
    # A LaTeX annotation, when present, is authoritative — use it verbatim.
    if name in ("math", "semantics"):
        for d in el.iter():
            if _localname(d.tag) == "annotation" and d.get("encoding") == "application/x-tex":
                return (d.text or "").strip()
    if name in ("mi", "mn", "mo", "mtext", "ms"):
        return (el.text or "").strip()
    if name in ("math", "mrow", "mstyle", "semantics", "mpadded", "mphantom"):
        return " ".join(p for p in _mml_children(el) if p)
    kids = _mml_children(el)
    if name == "msub" and len(kids) == 2:
        return f"{kids[0]}_{{{kids[1]}}}"
    if name == "msup" and len(kids) == 2:
        return f"{kids[0]}^{{{kids[1]}}}"
    if name == "msubsup" and len(kids) == 3:
        return f"{kids[0]}_{{{kids[1]}}}^{{{kids[2]}}}"
    if name == "mfrac" and len(kids) == 2:
        return f"\\frac{{{kids[0]}}}{{{kids[1]}}}"
    if name == "msqrt":
        return f"\\sqrt{{{' '.join(kids)}}}"
    if name == "mroot" and len(kids) == 2:
        return f"\\sqrt[{kids[1]}]{{{kids[0]}}}"
    if name in ("munder", "mover") and len(kids) == 2:
        return f"{kids[0]}_{{{kids[1]}}}" if name == "munder" else f"{kids[0]}^{{{kids[1]}}}"
    if name == "munderover" and len(kids) == 3:
        return f"{kids[0]}_{{{kids[1]}}}^{{{kids[2]}}}"
    if name == "mfenced":
        return f"({' '.join(kids)})"
    return " ".join(p for p in kids if p)


# --------------------------------------------------------------------------- #
# The walker
# --------------------------------------------------------------------------- #

def _render(el: Any, out: list[str], vocab: dict[str, str]) -> None:
    name = _localname(el.tag)
    if name is None:
        if getattr(el, "tail", None):
            out.append(el.tail)
        return
    role = vocab.get(name, INLINE)

    # <script type="math/tex">LaTeX</script> (MathJax source) is math, not skip.
    if name == "script":
        if (el.get("type") or "").startswith("math/tex"):
            out.append(f" ${(el.text or '').strip()}$ ")
        _tail(el, out)
        return
    if role == SKIP:
        _tail(el, out)
        return
    if role == MATH:
        out.append(f" ${_mathml_to_text(el)}$ ")
        _tail(el, out)
        return
    if role == CODE_BLOCK:
        out.append(f"\n```\n{el.text_content() if hasattr(el, 'text_content') else _text(el)}\n```\n")
        _tail(el, out)
        return
    if role == CODE_INLINE:
        inner = el.text_content() if hasattr(el, "text_content") else _text(el)
        out.append(f"\n```\n{inner}\n```\n" if "\n" in inner else f"`{inner}`")
        _tail(el, out)
        return

    if role == HEADING:
        level = HTML_LEVEL.get(name, 2)
        out.append(f"\n\n{'#' * level} ")
    elif role == BLOCK:
        out.append("\n\n")
    elif role == LIST_ITEM:
        out.append("\n- ")
    elif name == "sub":
        out.append("_{")
    elif name == "sup":
        out.append("^{")

    if el.text:
        out.append(el.text)
    for child in el:
        _render(child, out, vocab)

    if name in ("sub", "sup"):
        out.append("}")
    elif role in (HEADING, BLOCK):
        out.append("\n")
    _tail(el, out)


def _tail(el: Any, out: list[str]) -> None:
    if getattr(el, "tail", None):
        out.append(el.tail)


def _text(el: Any) -> str:
    return "".join(el.itertext())


def _clean(parts: list[str]) -> str:
    text = "".join(parts)
    text = _WS.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def html_to_text(html: str) -> str:
    """Flatten an HTML document/fragment to clean, math-aware text."""
    from lxml import etree
    from lxml import html as lxml_html

    if not html or not html.strip():
        return ""
    try:
        root = lxml_html.fromstring(html)
    except (etree.ParserError, etree.XMLSyntaxError, ValueError):
        return _clean([re.sub(r"<[^>]+>", "", html)])
    out: list[str] = []
    _render(root, out, HTML_VOCAB)
    return _clean(out)


def cnxml_to_text(cnxml: str | bytes) -> str:
    """Flatten an OpenStax CNXML module to clean, math-aware text."""
    from lxml import etree

    data = cnxml.encode("utf-8") if isinstance(cnxml, str) else cnxml
    if not data.strip():
        return ""
    root = etree.fromstring(data, parser=etree.XMLParser(recover=True, huge_tree=True))
    if root is None:
        return ""
    out: list[str] = []
    _render(root, out, CNXML_VOCAB)
    return _clean(out)
