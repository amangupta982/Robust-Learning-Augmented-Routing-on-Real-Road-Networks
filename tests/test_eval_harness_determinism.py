"""Harness determinism (CLAUDE.md Phase 5): running the same tiny config
twice with the same seed must produce byte-identical numbers. Uses a small
in-memory config and cheap predictors (oracle, adversarial, noisy -- no
lightgbm, to keep this fast) so it can run as part of the regular test
suite, not just as a one-off manual check.

Skipped if the real graph/features haven't been built yet (the harness
fundamentally operates on the real LA graph and real METR-LA data).
"""

from __future__ import annotations

import pandas as pd
import pytest
from roar.eval.harness import build_eval_context, run_experiment
from roar.eval.metrics import compute_oracle_optimal_costs
from roar.eval.queries import generate_stratified_queries

from tests.robust_astar_fixtures import real_data_available

pytestmark = pytest.mark.skipif(
    not real_data_available(), reason="real graph/features not built yet; run `make data` first"
)

TINY_QUERY_CFG = {
    "n_queries": 15,
    "direct_fraction": 0.85,
    "distance_strata": ["short", "medium", "long"],
    "time_strata": {
        "night": [0, 6],
        "morning_peak": [6, 10],
        "midday": [10, 16],
        "evening_peak": [16, 20],
        "evening": [20, 24],
    },
}

TINY_EXP_CFG = {
    "experiment": "determinism_smoke_test",
    "run_id": "determinism_smoke_test",
    "seed": 42,
    "alpha": 0.3,
    "lambda_base": 1.0,
    "lambda_mode": "fixed",
    "methods": [
        "dijkstra",
        "astar",
        "bidirectional_dijkstra",
        "pure_ml_astar",
        "robust_astar",
        "robust_astar_no_guard",
    ],
    "sweep": {
        "param": "predictor",
        "values": [
            {"type": "oracle"},
            {"type": "adversarial", "budget": 0.9},
            {"type": "noisy", "sigma_level": 0.5, "seed": 42},
        ],
    },
}


@pytest.fixture(scope="module")
def ctx():
    return build_eval_context()


def _run_once(ctx):
    queries = generate_stratified_queries(ctx.graph, ctx.features_df, TINY_QUERY_CFG, seed=42)
    oracle_optimal = compute_oracle_optimal_costs(ctx.graph, ctx.oracle, queries)
    # A fixed provenance dict (not make_provenance()) sidesteps the one
    # legitimately non-deterministic field (run_timestamp, wall-clock time)
    # -- everything else must still match exactly.
    provenance = {"git_commit": "test", "library_versions": {}, "run_timestamp": "test"}
    return run_experiment(
        TINY_EXP_CFG, "tests/tiny_config.yaml", 42, queries, oracle_optimal, ctx, provenance
    )


def test_same_seed_produces_byte_identical_results(ctx):
    df1 = _run_once(ctx)
    df2 = _run_once(ctx)

    assert len(df1) > 0
    # latency_ms is real wall-clock timing -- it will never be bit-identical
    # across two runs and isn't part of the "same seed -> same numbers"
    # guarantee; every other column must match exactly.
    non_timing_cols = [c for c in df1.columns if c != "latency_ms"]
    pd.testing.assert_frame_equal(df1[non_timing_cols], df2[non_timing_cols])
    assert (df1["latency_ms"] >= 0).all()
    assert (df2["latency_ms"] >= 0).all()


def test_query_generation_alone_is_deterministic(ctx):
    q1 = generate_stratified_queries(ctx.graph, ctx.features_df, TINY_QUERY_CFG, seed=42)
    q2 = generate_stratified_queries(ctx.graph, ctx.features_df, TINY_QUERY_CFG, seed=42)
    assert q1 == q2


def test_different_seed_produces_different_queries(ctx):
    q1 = generate_stratified_queries(ctx.graph, ctx.features_df, TINY_QUERY_CFG, seed=42)
    q2 = generate_stratified_queries(ctx.graph, ctx.features_df, TINY_QUERY_CFG, seed=43)
    assert q1 != q2


def test_result_schema_has_the_expected_columns(ctx):
    df = _run_once(ctx)
    expected = {
        "run_id", "experiment", "method", "query_seed", "query_id", "origin", "dest", "depart_time",
        "distance_stratum", "time_stratum", "predictor_type", "predictor_params",
        "lambda_base", "lambda_mode", "alpha", "guard_enabled", "cost",
        "oracle_optimal_cost", "competitive_ratio", "node_expansions", "latency_ms",
        "guard_invoked", "seed", "git_commit", "library_versions", "config_path",
        "run_timestamp",
    }
    assert expected.issubset(set(df.columns))


def test_guard_ablation_rows_come_from_a_single_search_per_query(ctx):
    """robust_astar and robust_astar_no_guard must agree exactly whenever
    the guard wasn't invoked (same underlying search, only the reported
    cost differs when the guard actually intervened)."""
    df = _run_once(ctx)
    guard_on = df[df["method"] == "robust_astar"].set_index(["query_id", "predictor_params"])
    guard_off = df[df["method"] == "robust_astar_no_guard"].set_index(
        ["query_id", "predictor_params"]
    )
    merged = guard_on[["cost", "guard_invoked"]].join(
        guard_off[["cost"]], lsuffix="_on", rsuffix="_off"
    )
    not_invoked = merged[merged["guard_invoked"] == False]  # noqa: E712
    assert len(not_invoked) > 0, "fixture never exercised the not-invoked case"
    pd.testing.assert_series_equal(
        not_invoked["cost_on"], not_invoked["cost_off"], check_names=False
    )
