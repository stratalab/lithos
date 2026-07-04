"""Tests for EX-1 (lithos/data/markup_text.py) — HTML/CNXML -> math-aware text."""

from __future__ import annotations

from lithos.data.markup_text import cnxml_to_text, html_to_text

# A CNXML fragment in OpenStax's real shape: cnxml + MathML namespaces, a section
# title, a paragraph with presentation MathML (no LaTeX annotation), and a list.
OPENSTAX_CNXML = """<?xml version="1.0"?>
<document xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML">
 <content>
  <section>
   <title>Motion Equations</title>
   <para>Consider <m:math><m:mrow><m:mi>y</m:mi><m:mo>=</m:mo>
     <m:msub><m:mi>v</m:mi><m:mn>0</m:mn></m:msub><m:mi>t</m:mi></m:mrow></m:math>,
     where <emphasis effect="italics">y</emphasis> is length.</para>
   <list><item>first point</item><item>second point</item></list>
  </section>
 </content>
</document>"""


# --------------------------------------------------------------------------- #
# CNXML (OpenStax) — the target format
# --------------------------------------------------------------------------- #

def test_cnxml_openstax_structure_and_math() -> None:
    out = cnxml_to_text(OPENSTAX_CNXML)
    assert "## Motion Equations" in out          # <title> -> heading
    assert "y = v_{0} t" in out                  # presentation MathML -> inline LaTeX
    assert "$" in out                            # math is delimited
    assert "y is length" in out                  # emphasis text kept inline
    assert "- first point" in out and "- second point" in out  # list items
    assert "xmlns" not in out and "m:math" not in out          # no tag/ns leakage


def test_cnxml_bytes_input() -> None:
    out = cnxml_to_text(OPENSTAX_CNXML.encode("utf-8"))
    assert "y = v_{0} t" in out


def test_cnxml_empty() -> None:
    assert cnxml_to_text("") == ""
    assert cnxml_to_text(b"") == ""


# --------------------------------------------------------------------------- #
# MathML conversion cases (via CNXML so namespaces parse cleanly)
# --------------------------------------------------------------------------- #

def _math(mml: str) -> str:
    doc = (f'<document xmlns="http://cnx.rice.edu/cnxml" '
           f'xmlns:m="http://www.w3.org/1998/Math/MathML"><para>{mml}</para></document>')
    return cnxml_to_text(doc)


def test_mathml_fraction() -> None:
    out = _math("<m:math><m:mfrac><m:mi>a</m:mi><m:mi>b</m:mi></m:mfrac></m:math>")
    assert "\\frac{a}{b}" in out


def test_mathml_superscript_and_sqrt() -> None:
    assert "c^{2}" in _math(
        "<m:math><m:msup><m:mi>c</m:mi><m:mn>2</m:mn></m:msup></m:math>")
    assert "\\sqrt{x}" in _math(
        "<m:math><m:msqrt><m:mi>x</m:mi></m:msqrt></m:math>")


def test_mathml_latex_annotation_wins() -> None:
    # When a LaTeX annotation is present (common in MathJax), use it verbatim.
    out = _math(
        '<m:math><m:semantics><m:mrow><m:mi>x</m:mi></m:mrow>'
        '<m:annotation encoding="application/x-tex">\\frac{1}{2}</m:annotation>'
        "</m:semantics></m:math>")
    assert "\\frac{1}{2}" in out


# --------------------------------------------------------------------------- #
# HTML — LibreTexts / gov reports / notes
# --------------------------------------------------------------------------- #

def test_html_structure() -> None:
    out = html_to_text(
        "<div><h2>Heading</h2><p>Text with <code>inline_code</code>.</p>"
        "<ul><li>a</li><li>b</li></ul><pre><code>x = 1\ny = 2</code></pre></div>")
    assert "## Heading" in out
    assert "`inline_code`" in out
    assert "- a" in out and "- b" in out
    assert "```" in out and "x = 1\ny = 2" in out


def test_html_skips_boilerplate() -> None:
    out = html_to_text(
        "<div><nav>menu links</nav><script>evil()</script>"
        "<p>real content</p><footer>copyright</footer></div>")
    assert "real content" in out
    assert "menu links" not in out and "evil()" not in out and "copyright" not in out


def test_html_math_tex_script() -> None:
    # MathJax LaTeX-source form.
    out = html_to_text('<p>Energy <script type="math/tex">E = mc^2</script> here.</p>')
    assert "E = mc^2" in out and "$" in out


def test_html_superscript() -> None:
    assert "cm^{2}" in html_to_text("<p>area in cm<sup>2</sup></p>")


def test_html_empty_and_malformed() -> None:
    assert html_to_text("") == ""
    assert "unterminated" in html_to_text("<p>unterminated <b>bold")
