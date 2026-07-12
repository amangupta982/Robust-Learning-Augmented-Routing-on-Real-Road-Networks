"""Statistical validation for Phase 6: bootstrap confidence intervals,
paired Wilcoxon signed-rank tests with effect sizes, and Holm-Bonferroni
correction across multiple baseline comparisons.

Correctness of the statistics matters as much as the routing code --
every function here is validated in tests/test_eval_stats.py against
synthetic inputs with hand-computed known answers. That is the ONE place
synthetic data is legitimate in this project (CLAUDE.md rule 1: never for
reported results, only for unit-testing the math itself).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclasses.dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n: int
    n_boot: int


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable = np.mean,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI for `statistic` (default: the mean) over
    `values`.

    Assumption: `values` are exchangeable observations from the population
    of interest (e.g. i.i.d. per-query metric values within one seed's
    query sample, or i.i.d. per-seed means across independent seeds) --
    the percentile bootstrap makes NO assumption that they're normally
    distributed. That matters here: competitive ratios are bounded below
    (~1, the best achievable) and effectively unbounded above (a bad
    predictor can blow up arbitrarily), i.e. heavily right-skewed, so a
    normal-theory CI (mean +/- 1.96*SEM) would be a poor, potentially
    misleading approximation. `statistic` must accept an `axis` keyword
    (true of np.mean/np.median/np.std/np.max/np.min).

    NaN values are dropped before resampling (e.g. a `competitive_ratio`
    of NaN from a 0/0 unreachable-query edge case) rather than propagating
    NaN through the whole CI.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0, n_boot)

    point_estimate = float(statistic(values))
    if n == 1:
        # No resampling variance is estimable from a single observation --
        # a degenerate (zero-width) interval is honest; a fabricated one is not.
        return BootstrapResult(point_estimate, point_estimate, point_estimate, 1, n_boot)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_samples = values[indices]
    boot_stats = statistic(boot_samples, axis=1)

    alpha = 1 - ci
    lo, hi = np.quantile(boot_stats, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(point_estimate, float(lo), float(hi), n, n_boot)


def seed_level_summary(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    seed_col: str = "query_seed",
    n_boot: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Two-level aggregation: mean within each seed's queries, THEN a
    bootstrap CI across the >=5 seed-level means -- this is what "mean +/-
    95% bootstrap CI ... over >=5 seeds" means: an independent source of
    uncertainty (what if we'd drawn a different random query sample
    entirely), distinct from `bootstrap_ci` applied directly to pooled
    per-query values (uncertainty from resampling the SAME sample).

    One output row per unique combination of `group_cols`."""
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        seed_means = group.groupby(seed_col)[value_col].mean().to_numpy()
        result = bootstrap_ci(seed_means, n_boot=n_boot, seed=seed)
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "n_seeds": result.n,
                "mean": result.point_estimate,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


@dataclasses.dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    effect_size_r: float
    n_pairs: int
    n_zero_diffs_dropped: int


def paired_wilcoxon(x: Sequence[float], y: Sequence[float]) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test comparing `x` vs `y` on the SAME
    matched units (e.g. robust A*'s competitive ratio vs a baseline's, on
    the same queries).

    Assumptions: (1) `x[i]` and `y[i]` are a matched pair (the same
    query); (2) under the null hypothesis, the paired differences x - y
    are symmetric around zero -- NOT that x or y are individually
    normally distributed. Competitive ratios are not normal (bounded
    below, right-skewed), which is exactly why a paired Wilcoxon
    signed-rank test is used here instead of a paired t-test. Zero
    differences are dropped before ranking (the standard "wilcox"
    zero-method); ties among nonzero |differences| use scipy's default
    mid-rank handling.

    Effect size: the matched-pairs rank-biserial correlation
        r = (W+ - W-) / (W+ + W-)
    computed directly from the signed ranks (W+ / W- = sum of ranks of
    positive / negative differences) rather than scipy's normal
    approximation (which is a large-sample approximation of a z-score,
    not this test's natural effect size). r in [-1, 1]; positive means x
    tends to exceed y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"x and y must be paired (same length), got {len(x)} vs {len(y)}")

    diff = x - y
    nonzero = diff[diff != 0]
    n_dropped = len(diff) - len(nonzero)

    if len(nonzero) == 0:
        return WilcoxonResult(float("nan"), float("nan"), 0.0, 0, n_dropped)

    ranks = scipy_stats.rankdata(np.abs(nonzero))
    w_pos = ranks[nonzero > 0].sum()
    w_neg = ranks[nonzero < 0].sum()
    effect_size_r = (w_pos - w_neg) / (w_pos + w_neg) if (w_pos + w_neg) > 0 else 0.0

    try:
        scipy_result = scipy_stats.wilcoxon(
            nonzero, zero_method="wilcox", alternative="two-sided"
        )
        statistic, p_value = float(scipy_result.statistic), float(scipy_result.pvalue)
    except ValueError:
        # scipy raises e.g. when too few nonzero differences remain.
        statistic, p_value = float("nan"), float("nan")

    return WilcoxonResult(statistic, p_value, float(effect_size_r), len(nonzero), n_dropped)


@dataclasses.dataclass(frozen=True)
class HolmResult:
    p_adjusted: list[float]
    reject: list[bool]


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> HolmResult:
    """Holm step-down procedure controlling the family-wise error rate
    across `m` simultaneous hypothesis tests -- here, robust A* vs each of
    several baselines on the same query set.

    Assumption: `p_values` are valid p-values in [0, 1] with no NaNs
    (filter degenerate comparisons out before calling). Unlike plain
    Bonferroni, Holm's method does not require the tests to be
    independent, which matters here since the comparisons share the same
    underlying queries.

    Procedure: sort ascending as p_(1) <= ... <= p_(m). Reject H_(i) iff
    p_(j) <= alpha / (m - j + 1) for every j <= i -- i.e. stop rejecting at
    the first failure and reject nothing after that (the step-down rule).
    Adjusted p-values are the smallest family-wise alpha at which H_(i)
    would be rejected:
        p_adj_(i) = max_{j <= i} [ (m - j + 1) * p_(j) ], clipped to <= 1
    enforced monotone non-decreasing by construction (the running max).
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return HolmResult([], [])
    if np.isnan(p_values).any():
        raise ValueError("holm_bonferroni does not accept NaN p-values; filter them out first")

    order = np.argsort(p_values, kind="stable")
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m)
    running_max = 0.0
    for i in range(m):
        candidate = (m - i) * sorted_p[i]
        running_max = max(running_max, candidate)
        adjusted_sorted[i] = min(running_max, 1.0)

    p_adjusted = np.empty(m)
    p_adjusted[order] = adjusted_sorted

    reject_sorted = np.zeros(m, dtype=bool)
    for i in range(m):
        if sorted_p[i] <= alpha / (m - i):
            reject_sorted[i] = True
        else:
            break  # Holm's step-down rule: stop at the first non-rejection
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted

    return HolmResult(p_adjusted.tolist(), reject.tolist())


def compare_robust_vs_baselines(
    df: pd.DataFrame,
    baseline_methods: list[str],
    robust_method: str = "robust_astar",
    value_col: str = "competitive_ratio",
    pair_keys: tuple[str, ...] = ("query_seed", "query_id"),
    alpha: float = 0.05,
    n_boot: int = 10_000,
) -> pd.DataFrame:
    """For each baseline in `baseline_methods`, pairs `robust_method`'s
    `value_col` against that baseline's on the SAME queries (matched by
    `pair_keys`), runs a paired Wilcoxon signed-rank test + effect size,
    then Holm-Bonferroni-corrects across all baselines compared in one
    call (the "multiple baseline comparisons" CLAUDE.md flags). One output
    row per baseline; feeds Table 1's significance columns directly."""
    robust_df = df[df["method"] == robust_method].set_index(list(pair_keys))

    rows = []
    for baseline in baseline_methods:
        baseline_df = df[df["method"] == baseline].set_index(list(pair_keys))
        joined = robust_df[[value_col]].join(
            baseline_df[[value_col]], lsuffix="_robust", rsuffix="_baseline", how="inner"
        )
        x = joined[f"{value_col}_robust"].to_numpy()
        y = joined[f"{value_col}_baseline"].to_numpy()

        wilcoxon_result = paired_wilcoxon(x, y)
        robust_ci = bootstrap_ci(x, n_boot=n_boot)
        baseline_ci = bootstrap_ci(y, n_boot=n_boot)
        rows.append(
            {
                "baseline": baseline,
                "n_pairs": wilcoxon_result.n_pairs,
                "statistic": wilcoxon_result.statistic,
                "p_value": wilcoxon_result.p_value,
                "effect_size_r": wilcoxon_result.effect_size_r,
                "robust_mean": robust_ci.point_estimate,
                "robust_ci_low": robust_ci.ci_low,
                "robust_ci_high": robust_ci.ci_high,
                "baseline_mean": baseline_ci.point_estimate,
                "baseline_ci_low": baseline_ci.ci_low,
                "baseline_ci_high": baseline_ci.ci_high,
            }
        )
    result_df = pd.DataFrame(rows)

    valid = ~result_df["p_value"].isna()
    holm = holm_bonferroni(result_df.loc[valid, "p_value"].to_numpy(), alpha=alpha)
    result_df["p_adjusted"] = np.nan
    result_df["significant"] = False
    result_df.loc[valid, "p_adjusted"] = holm.p_adjusted
    result_df.loc[valid, "significant"] = holm.reject
    return result_df
