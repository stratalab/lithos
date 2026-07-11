"""Paired statistics for the tool-uplift battery (docs/eval-tir-battery-plan.md).

Every problem is scored in both arms (tools off / on), so tool-uplift is a *paired*
comparison, and its uncertainty must respect two things (`eval-plan.md` §0.9):

- **Pairing** — the arms share the problem, so the signal lives in the *discordant*
  pairs (off-wrong→on-right vs on-wrong→off-right). That is McNemar's test, not two
  independent proportions; we use the exact (binomial) form because discordant counts
  are small.
- **Clustering** — near-duplicate problems (same ``family_id``) are not independent, so
  a naive standard error understates the truth; clustered SEs can run >3× larger. We
  report both so the gap is visible and the weight-class delta is judged on the honest
  interval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

_Z95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclass(frozen=True)
class UpliftStat:
    """Tool-uplift for one slice of problems, with a paired, cluster-robust interval."""

    n: int
    solve_off: float  # verified solve rate, tools off
    solve_on: float  # verified solve rate, tools on
    uplift: float  # solve_on - solve_off (the mean paired difference)
    se_naive: float  # SE of the mean difference treating rows as independent
    se_clustered: float  # cluster-robust SE (clusters = family_id); the one we trust
    ci_low: float  # uplift ± z * se_clustered
    ci_high: float
    mcnemar_on_gain: int  # b: off-wrong, on-right (tools helped)
    mcnemar_on_loss: int  # c: on-wrong, off-right (tools hurt)
    mcnemar_p: float  # two-sided exact McNemar p-value
    significant: bool  # clustered CI excludes 0 AND mcnemar_p < 0.05


def _exact_mcnemar_p(gain: int, loss: int) -> float:
    """Two-sided exact McNemar p-value: under H0 the ``gain`` discordant pairs are
    Binomial(gain+loss, 0.5). Exact rather than chi-square because discordant counts
    are typically small (that is where the continuity approximation is worst)."""
    n = gain + loss
    if n == 0:
        return 1.0
    k = min(gain, loss)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_uplift(
    off: Sequence[bool], on: Sequence[bool], clusters: Sequence[str | None]
) -> UpliftStat:
    """Paired tool-uplift over aligned per-problem verdicts.

    ``off[i]``/``on[i]`` are the two arms' correctness on problem ``i``; ``clusters[i]``
    is its ``family_id`` (``None`` → a singleton family). The uplift is the mean of the
    per-problem differences ``on - off`` ∈ {-1, 0, +1}; the interval is the
    cluster-robust (CR1) SE of that mean.
    """
    if not (len(off) == len(on) == len(clusters)):
        raise ValueError("off / on / clusters must be the same length")
    n = len(off)
    if n == 0:
        return UpliftStat(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 1.0, False)

    diffs = [int(o) - int(f) for f, o in zip(off, on, strict=True)]
    dbar = sum(diffs) / n
    solve_off = sum(1 for f in off if f) / n
    solve_on = sum(1 for o in on if o) / n

    # Naive SE of the mean paired difference (rows independent).
    var = sum((d - dbar) ** 2 for d in diffs) / n
    se_naive = math.sqrt(var / n)

    # Cluster-robust SE: sum over clusters of the squared within-cluster residual sum,
    # with the CR1 small-sample correction G/(G-1). A None family_id is its own cluster.
    residual_sums: dict[str, float] = {}
    for i, (d, cl) in enumerate(zip(diffs, clusters, strict=True)):
        key = cl if cl is not None else f"__singleton_{i}"
        residual_sums[key] = residual_sums.get(key, 0.0) + (d - dbar)
    g = len(residual_sums)
    ss = sum(s * s for s in residual_sums.values())
    correction = g / (g - 1) if g > 1 else 1.0
    se_clustered = math.sqrt(correction * ss / (n * n))

    ci_low = dbar - _Z95 * se_clustered
    ci_high = dbar + _Z95 * se_clustered

    gain = sum(1 for d in diffs if d > 0)
    loss = sum(1 for d in diffs if d < 0)
    p = _exact_mcnemar_p(gain, loss)
    significant = (ci_low > 0.0 or ci_high < 0.0) and p < 0.05

    return UpliftStat(
        n=n,
        solve_off=solve_off,
        solve_on=solve_on,
        uplift=dbar,
        se_naive=se_naive,
        se_clustered=se_clustered,
        ci_low=ci_low,
        ci_high=ci_high,
        mcnemar_on_gain=gain,
        mcnemar_on_loss=loss,
        mcnemar_p=p,
        significant=significant,
    )
