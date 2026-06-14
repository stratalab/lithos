"""Eval report writer (PRD §11.3): results.json + results.md + config + reference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lithos.utils.io import atomic_write_text, ensure_dir, write_json, write_yaml


def _render_markdown(name: str, results: dict[str, Any], model_reference: dict[str, Any]) -> str:
    lines = [f"# Eval report: {name}", ""]
    lines.append("## Model")
    for key, value in model_reference.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    if "perplexity" in results:
        ppl = results["perplexity"]
        lines += [
            "## Perplexity",
            f"- loss: {ppl['loss']:.4f}",
            f"- perplexity: {ppl['perplexity']:.4f}",
            f"- tokens: {ppl['tokens']:,}",
            "",
        ]
    samples = results.get("samples")
    if samples:
        lines.append("## Samples")
        for s in samples:
            lines += [
                f"### {s['prompt']!r}",
                f"- repetition: {s['repetition']:.3f} ({s['n_new_tokens']} new tokens)",
                "",
                "```",
                s["completion"],
                "```",
                "",
            ]
    if results.get("notes"):
        lines += ["## Notes", results["notes"], ""]
    return "\n".join(lines)


def write_eval_report(
    output_dir: str | Path,
    *,
    name: str,
    results: dict[str, Any],
    model_reference: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    out = ensure_dir(output_dir)
    write_json(out / "results.json", results)
    write_json(out / "model_reference.json", model_reference)
    write_yaml(out / "config.yaml", config)
    atomic_write_text(out / "results.md", _render_markdown(name, results, model_reference))
    return out
