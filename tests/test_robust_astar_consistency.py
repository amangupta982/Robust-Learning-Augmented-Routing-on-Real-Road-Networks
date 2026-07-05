"""CRITICAL test (Phase 4): Consistency.

With a perfect predictor (OraclePredictor) and full trust (lambda_base=1.0),
RobustAStar's blended heuristic becomes exactly the EXACT remaining-cost
model (h(n) == ml_estimate(n), computed by a real backward Dijkstra over the
oracle's own -- here, real -- cost function). An exact heuristic is
trivially admissible and consistent, so the ML-guided search must find the
TRUE ground-truth-optimal path: robust A*'s realized cost should match a
reference oracle-cost shortest-path search almost exactly, and the guard
should never need to intervene.

Uses a modest number of real queries (not "thousands" -- that's the
robustness test's job) since this is a correctness/consistency check, not a
stress test. Skipped if the real data/graph hasn't been built yet.
"""

from __future__ import annotations

import pytest
from roar.routing.baselines import astar, predictor_cost_fn
from roar.routing.robust_astar import RobustAStar

from tests.robust_astar_fixtures import (
    direct_instrumented_queries,
    load_features_df,
    load_oracle,
    load_real_graph,
    real_data_available,
)

pytestmark = pytest.mark.skipif(
    not real_data_available(), reason="real graph/features not built yet; run `make data` first"
)


@pytest.fixture(scope="module")
def graph():
    return load_real_graph()


@pytest.fixture(scope="module")
def oracle():
    return load_oracle(load_features_df())


def test_oracle_guided_robust_astar_matches_true_optimal_cost(graph, oracle):
    features_df = load_features_df()
    queries = direct_instrumented_queries(features_df, n=30, seed=123)
    ground_truth_cost_fn = predictor_cost_fn(oracle)

    robust = RobustAStar(
        graph,
        planning_predictor=oracle,
        ground_truth_cost_fn=ground_truth_cost_fn,
        alpha=0.3,
        lambda_base=1.0,
    )

    ratios = []
    for origin, dest, depart_time in queries:
        true_optimal = astar(graph, origin, dest, depart_time, ground_truth_cost_fn)
        result = robust.search(origin, dest, depart_time)

        assert true_optimal.path is not None, "fixture query must be reachable"
        assert result.path is not None

        # Consistency: robust A* achieves the TRUE ground-truth-optimal
        # realized cost when the predictor is perfect and fully trusted --
        # "ratio ~1" relative to true_optimal (a different comparison than
        # robustness_bound.ratio, which is relative to the classical
        # free-flow baseline, not the true optimum). result.cost can never
        # be BELOW true_optimal.cost (that would be a mathematically
        # impossible "better than optimal" path).
        assert result.cost >= true_optimal.cost - 1e-6, (
            f"robust A* realized cost ({result.cost}) is below the TRUE "
            f"optimal ({true_optimal.cost}) -- impossible, investigate a bug"
        )
        ratio_to_true_optimal = result.cost / true_optimal.cost if true_optimal.cost > 0 else 1.0
        ratios.append(ratio_to_true_optimal)

        assert result.cost == pytest.approx(true_optimal.cost, abs=1e-6), (
            f"robust A* realized cost ({result.cost}) does not match the TRUE "
            f"optimal ({true_optimal.cost}) with a perfect predictor and full "
            "trust -- consistency should hold exactly here"
        )
        # With a perfect predictor the guard should never need to reject
        # the ML-guided candidate.
        assert result.robustness_bound.guard_invoked is False, (
            f"guard fired even though the predictor was a perfect oracle: {result}"
        )

    mean_ratio = sum(ratios) / len(ratios)
    assert mean_ratio == pytest.approx(1.0, abs=1e-6)
