"""Improvement Phase Task 1c: path-identity proof.

`MaterializedLightGBMPredictor` is a PERFORMANCE optimization (see its
module docstring for what profiling actually found), never a different
model. On 500 real queries, RobustAStar guided by the materialized
predictor MUST return identical paths and costs to RobustAStar guided by
the unoptimized LightGBMPredictor. Per this phase's hard rules: if they
ever differ, that is a correctness bug in the optimization and must be
reported loudly, not averaged away or silently tolerated.
"""

from __future__ import annotations

import pandas as pd
import pytest
from roar.eval.metrics import resolve_model_dir
from roar.graph.features import FEATURES_PATH
from roar.predictor.lightgbm import LightGBMPredictor
from roar.predictor.materialized_lightgbm import MaterializedLightGBMPredictor
from roar.routing.baselines import predictor_cost_fn
from roar.routing.robust_astar import RobustAStar

from tests.robust_astar_fixtures import (
    cross_network_queries,
    direct_instrumented_queries,
    load_oracle,
    load_real_graph,
    real_data_available,
)

pytestmark = pytest.mark.skipif(
    not (real_data_available() and resolve_model_dir().exists()),
    reason="real graph/features/trained model not built yet; run `make data && make train` first",
)

ALPHA = 0.3
N_QUERIES = 500
N_DIRECT = 400
N_CROSS = N_QUERIES - N_DIRECT


def _load_full_features_df() -> pd.DataFrame:
    """The full feature table -- NOT tests/robust_astar_fixtures.py's
    load_features_df(), which only reads a trimmed column subset for
    OraclePredictor's needs. LightGBMPredictor.load() requires every
    engineered feature column."""
    return pd.read_parquet(FEATURES_PATH)


def _lightgbm_predictors(features_df: pd.DataFrame):
    model_dir = resolve_model_dir()
    base = LightGBMPredictor.load(model_dir, features_df)
    materialized = MaterializedLightGBMPredictor(base, features_df)
    return base, materialized


def test_materialized_predictor_matches_base_predictor_on_500_real_queries():
    graph = load_real_graph()
    features_df = _load_full_features_df()
    oracle = load_oracle(features_df)
    ground_truth_cost_fn = predictor_cost_fn(oracle)
    base, materialized = _lightgbm_predictors(features_df)

    queries = direct_instrumented_queries(
        features_df, n=N_DIRECT, seed=7007
    ) + cross_network_queries(features_df, n=N_CROSS, seed=7007)
    assert len(queries) == N_QUERIES

    robust_base = RobustAStar(graph, base, ground_truth_cost_fn, alpha=ALPHA, lambda_base=1.0)
    robust_materialized = RobustAStar(
        graph, materialized, ground_truth_cost_fn, alpha=ALPHA, lambda_base=1.0
    )

    mismatches = []
    n_reachable = 0
    for origin, dest, depart_time in queries:
        result_base = robust_base.search(origin, dest, depart_time)
        result_materialized = robust_materialized.search(origin, dest, depart_time)

        if result_base.path is None and result_materialized.path is None:
            continue
        n_reachable += 1

        same_path = result_base.path == result_materialized.path
        same_cost = abs(result_base.cost - result_materialized.cost) < 1e-6
        if not (same_path and same_cost):
            mismatches.append(
                {
                    "origin": origin,
                    "dest": dest,
                    "depart_time": depart_time,
                    "base_path": result_base.path,
                    "materialized_path": result_materialized.path,
                    "base_cost": result_base.cost,
                    "materialized_cost": result_materialized.cost,
                }
            )

    assert n_reachable > 0, "fixture produced no reachable queries -- test is vacuous"
    assert not mismatches, (
        f"MaterializedLightGBMPredictor diverged from the base predictor on "
        f"{len(mismatches)}/{n_reachable} reachable queries -- this is a "
        f"correctness bug in the optimization, not noise: first mismatch = "
        f"{mismatches[0]}"
    )


def test_materialized_predictor_matches_base_eta_directly_on_a_sample():
    """A finer-grained check than the full-search path identity above:
    the two predictors' raw eta/sigma must match on individual
    (edge_id, depart_time) pairs, not just in aggregate path outcomes."""
    features_df = _load_full_features_df()
    base, materialized = _lightgbm_predictors(features_df)

    sample = features_df[features_df["split"] == "test"].sample(200, random_state=1)
    for _, row in sample.iterrows():
        edge_id = row["edge_id"]
        depart_time = row["timestamp"].to_pydatetime()
        eta_base, sigma_base = base.eta_with_confidence(edge_id, depart_time)
        eta_mat, sigma_mat = materialized.eta_with_confidence(edge_id, depart_time)
        assert eta_base == pytest.approx(eta_mat, abs=1e-9)
        assert sigma_base == pytest.approx(sigma_mat, abs=1e-9)
