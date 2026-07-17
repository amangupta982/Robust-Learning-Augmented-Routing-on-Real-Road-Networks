"""Unit tests for roar/eval/stats.py, validated against synthetic inputs
with hand-computed known answers -- the ONE place synthetic data is
legitimate in this project (CLAUDE.md rule 1: never for reported results,
only for unit-testing the math itself). None of these numbers are ever
reported as a ROAR result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from roar.eval.stats import (
    bootstrap_ci,
    compare_robust_vs_baselines,
    holm_bonferroni,
    paired_wilcoxon,
    seed_level_summary,
)

# ---- bootstrap_ci ----


def test_bootstrap_ci_point_estimate_is_the_exact_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = bootstrap_ci(values, n_boot=2000, seed=1)
    assert result.point_estimate == 3.0
    assert result.ci_low <= 3.0 <= result.ci_high
    assert result.n == 5


def test_bootstrap_ci_constant_array_has_zero_width_interval():
    values = [7.0] * 20
    result = bootstrap_ci(values, n_boot=1000, seed=1)
    assert result.point_estimate == 7.0
    assert result.ci_low == 7.0
    assert result.ci_high == 7.0


def test_bootstrap_ci_single_value_is_degenerate_not_fabricated():
    result = bootstrap_ci([42.0], n_boot=1000, seed=1)
    assert result.point_estimate == 42.0
    assert result.ci_low == 42.0
    assert result.ci_high == 42.0
    assert result.n == 1


def test_bootstrap_ci_empty_input_is_nan():
    result = bootstrap_ci([], n_boot=1000, seed=1)
    assert result.n == 0
    assert result.point_estimate != result.point_estimate  # NaN


def test_bootstrap_ci_drops_nan_values():
    values = [1.0, 2.0, 3.0, float("nan")]
    result = bootstrap_ci(values, n_boot=2000, seed=1)
    assert result.n == 3
    assert result.point_estimate == 2.0


def test_bootstrap_ci_is_reproducible_given_the_same_seed():
    values = np.linspace(0, 10, 50).tolist()
    r1 = bootstrap_ci(values, n_boot=500, seed=7)
    r2 = bootstrap_ci(values, n_boot=500, seed=7)
    assert r1 == r2


def test_bootstrap_ci_supports_a_custom_statistic():
    values = [1.0, 2.0, 3.0, 4.0, 100.0]  # right-skewed -- median << mean
    mean_result = bootstrap_ci(values, statistic=np.mean, n_boot=2000, seed=1)
    median_result = bootstrap_ci(values, statistic=np.median, n_boot=2000, seed=1)
    assert median_result.point_estimate == 3.0
    assert mean_result.point_estimate == 22.0
    assert median_result.point_estimate < mean_result.point_estimate


# ---- paired_wilcoxon ----


def test_paired_wilcoxon_all_differences_negative_gives_effect_size_minus_one():
    # x always 1 less than y -> all diffs identical (tied |diff|) and
    # negative -> W+ = 0, W- = sum of all ranks -> r = (0 - W-)/(0 + W-) = -1.
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 3.0, 4.0, 5.0, 6.0]
    result = paired_wilcoxon(x, y)
    assert result.effect_size_r == pytest.approx(-1.0)
    assert result.n_pairs == 5
    assert result.n_zero_diffs_dropped == 0


def test_paired_wilcoxon_all_differences_positive_gives_effect_size_plus_one():
    x = [5.0, 5.0, 5.0]
    y = [1.0, 1.0, 1.0]
    result = paired_wilcoxon(x, y)
    assert result.effect_size_r == pytest.approx(1.0)
    assert result.n_pairs == 3


def test_paired_wilcoxon_drops_zero_differences():
    x = [1.0, 2.0, 3.0, 10.0]
    y = [1.0, 2.0, 3.0, 1.0]  # first three pairs are tied (zero diff)
    result = paired_wilcoxon(x, y)
    assert result.n_zero_diffs_dropped == 3
    assert result.n_pairs == 1
    assert result.effect_size_r == pytest.approx(1.0)


def test_paired_wilcoxon_all_zero_differences_is_undefined_not_fabricated():
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    result = paired_wilcoxon(x, y)
    assert result.n_pairs == 0
    assert result.effect_size_r == 0.0
    assert result.p_value != result.p_value  # NaN


def test_paired_wilcoxon_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="paired"):
        paired_wilcoxon([1.0, 2.0], [1.0])


def test_paired_wilcoxon_matches_scipy_pvalue_on_a_known_example():
    import scipy.stats as scipy_stats

    rng = np.random.default_rng(0)
    x = rng.normal(size=30).tolist()
    y = (np.array(x) + rng.normal(scale=2.0, size=30)).tolist()
    result = paired_wilcoxon(x, y)
    diff = np.array(x) - np.array(y)
    expected = scipy_stats.wilcoxon(diff[diff != 0], zero_method="wilcox")
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)


# ---- holm_bonferroni ----


def test_holm_bonferroni_textbook_example_sorted_input():
    # Hand-derived (see stats.py docstring for the step-down formula):
    # only the smallest p-value (0.01) clears its threshold (0.05/5=0.01);
    # the second (0.02) fails its threshold (0.05/4=0.0125), so the
    # step-down procedure stops rejecting from there on.
    p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    result = holm_bonferroni(p_values, alpha=0.05)
    assert result.reject == [True, False, False, False, False]
    assert result.p_adjusted == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])


def test_holm_bonferroni_preserves_original_order_for_unsorted_input():
    # Same five p-values, permuted -- only original index 1 (p=0.01) rejects.
    p_values = [0.03, 0.01, 0.05, 0.02, 0.04]
    result = holm_bonferroni(p_values, alpha=0.05)
    assert result.reject == [False, True, False, False, False]
    assert result.p_adjusted == pytest.approx([0.09, 0.05, 0.09, 0.08, 0.09])


def test_holm_bonferroni_single_hypothesis_reduces_to_unadjusted():
    result = holm_bonferroni([0.03], alpha=0.05)
    assert result.reject == [True]
    assert result.p_adjusted == pytest.approx([0.03])


def test_holm_bonferroni_empty_input():
    result = holm_bonferroni([], alpha=0.05)
    assert result.p_adjusted == []
    assert result.reject == []


def test_holm_bonferroni_rejects_nan_p_values():
    with pytest.raises(ValueError, match="NaN"):
        holm_bonferroni([0.01, float("nan")], alpha=0.05)


def test_holm_bonferroni_adjusted_p_values_are_monotone_nondecreasing_in_sorted_order():
    p_values = [0.2, 0.001, 0.15, 0.04, 0.5, 0.03]
    result = holm_bonferroni(p_values, alpha=0.05)
    order = np.argsort(p_values)
    adjusted_sorted = np.array(result.p_adjusted)[order]
    assert np.all(np.diff(adjusted_sorted) >= -1e-12)


# ---- seed_level_summary ----


def test_seed_level_summary_aggregates_within_seed_then_across_seeds():
    df = pd.DataFrame(
        {
            "method": ["robust_astar"] * 10,
            "query_seed": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            "value": [1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    # seed 1 mean = 2.0, seed 2 mean = 5.0, seed 3 mean = 10.0
    result = seed_level_summary(df, value_col="value", group_cols=["method"], n_boot=2000)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["n_seeds"] == 3
    assert row["mean"] == pytest.approx((2.0 + 5.0 + 10.0) / 3)
    assert row["ci_low"] <= row["mean"] <= row["ci_high"]


def test_seed_level_summary_groups_independently_per_group_col():
    df = pd.DataFrame(
        {
            "method": ["a", "a", "b", "b"],
            "query_seed": [1, 2, 1, 2],
            "value": [1.0, 1.0, 100.0, 100.0],
        }
    )
    result = seed_level_summary(df, value_col="value", group_cols=["method"], n_boot=500)
    means = dict(zip(result["method"], result["mean"], strict=True))
    assert means["a"] == pytest.approx(1.0)
    assert means["b"] == pytest.approx(100.0)


# ---- compare_robust_vs_baselines (integration of the pieces above) ----


def test_compare_robust_vs_baselines_end_to_end_on_synthetic_paired_data():
    rng = np.random.default_rng(0)
    n = 40
    query_ids = list(range(n))
    # robust_astar: ratio hovering near 1 (good); baseline_bad: much worse;
    # baseline_same: statistically indistinguishable from robust_astar.
    robust_vals = 1.0 + rng.normal(0, 0.01, size=n)
    bad_vals = 3.0 + rng.normal(0, 0.5, size=n)
    same_vals = robust_vals + rng.normal(0, 0.001, size=n)

    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "method": "robust_astar",
                    "query_seed": 1,
                    "query_id": query_ids,
                    "competitive_ratio": robust_vals,
                }
            ),
            pd.DataFrame(
                {
                    "method": "baseline_bad",
                    "query_seed": 1,
                    "query_id": query_ids,
                    "competitive_ratio": bad_vals,
                }
            ),
            pd.DataFrame(
                {
                    "method": "baseline_same",
                    "query_seed": 1,
                    "query_id": query_ids,
                    "competitive_ratio": same_vals,
                }
            ),
        ],
        ignore_index=True,
    )

    result = compare_robust_vs_baselines(
        df, baseline_methods=["baseline_bad", "baseline_same"], n_boot=2000
    )

    assert set(result["baseline"]) == {"baseline_bad", "baseline_same"}
    bad_row = result[result["baseline"] == "baseline_bad"].iloc[0]
    same_row = result[result["baseline"] == "baseline_same"].iloc[0]

    assert bad_row["n_pairs"] == n
    assert bad_row["significant"]  # robust_astar clearly beats baseline_bad
    assert bad_row["effect_size_r"] < -0.5  # robust < baseline_bad consistently
    assert bad_row["p_adjusted"] < 0.05

    # baseline_same is statistically indistinguishable from robust_astar --
    # this could occasionally be flaky by chance, but the effect size
    # should be small regardless of significance.
    assert abs(same_row["effect_size_r"]) < 0.9
