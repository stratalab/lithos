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
    bench = results.get("benchmarks")
    if bench and bench.get("tasks"):
        lines.append(
            f"## Benchmarks (battery {bench.get('battery_version', '?')}, "
            f"{bench.get('num_fewshot', 0)}-shot)"
        )
        for task, t in sorted(bench["tasks"].items()):
            value = t.get("value")
            shown = f"{value:.4f} ({t.get('metric')})" if value is not None else "n/a"
            lines.append(f"- **{task}**: {shown}")
        if bench.get("mean") is not None:
            lines.append(f"- **mean**: {bench['mean']:.4f}")
        lines.append("")
    tir = results.get("tir")
    if tir and tir.get("overall"):
        ov = tir["overall"]
        sig = "significant" if ov.get("significant") else "n.s."
        lines.append(
            f"## Tool-uplift (battery {tir.get('battery_version', tir.get('battery', '?'))}, "
            f"n={tir.get('n', 0)})"
        )
        lines.append(
            f"- **overall**: {ov['uplift']:+.3f} "
            f"[{ov['ci_low']:+.3f}, {ov['ci_high']:+.3f}] "
            f"(off {ov['solve_off']:.3f} → on {ov['solve_on']:.3f}; "
            f"McNemar p={ov['mcnemar_p']:.3g}, {sig})"
        )
        for tier, s in sorted(tir.get("per_tier", {}).items()):
            lines.append(
                f"  - {tier}: {s['uplift']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] "
                f"(n={s['n']}, off {s['solve_off']:.3f} → on {s['solve_on']:.3f})"
            )
        h = tir.get("health", {})
        lines.append(
            f"- health: tool-call {h.get('tool_call_rate', 0):.3f}, "
            f"malformed {h.get('malformed_call_rate', 0):.3f}, "
            f"truncation {h.get('truncation_rate_on', 0):.3f}, "
            f"calls/solve {h.get('tool_calls_per_solve', 0):.2f}"
        )
        lines.append("")
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
