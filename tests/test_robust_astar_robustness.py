"""CRITICAL test (Phase 4): Robustness.

With an AdversarialPredictor at (near-)maximum corruption guiding the
search, the REALIZED path cost (evaluated against ground truth, never
against what the adversary claims) must NEVER exceed (1 + alpha) times the
realized cost of the classical, predictor-free path -- across a large
random query set built from real graph nodes and real test-split
timestamps.

This holds BY CONSTRUCTION (see roar/routing/guard.py's docstring: the
guard is a global compare-and-fallback, not a probabilistic mechanism), so
this test is not "discovering" a statistical property -- it is exercising
the real implementation across thousands of diverse real queries to catch
any bug that would make the guarantee not actually hold as coded. Per
CLAUDE.md: if this test ever fails, the guard is broken and must be treated
as a genuine, paper-blocking bug -- not loosened or skipped.
"""

from __future__ import annotations

import pytest
from roar.predictor.adversarial import AdversarialPredictor
from roar.routing.baselines import predictor_cost_fn
from roar.routing.robust_astar import RobustAStar

from tests.robust_astar_fixtures import (
    cross_network_queries,
    direct_instrumented_queries,
    load_features_df,
    load_oracle,
    load_real_graph,
    real_data_available,
)

pytestmark = pytest.mark.skipif(
    not real_data_available(), reason="real graph/features not built yet; run `make data` first"
)

ALPHA = 0.3
# AdversarialPredictor requires budget < 1 (see roar/predictor/adversarial.py);
# 0.95 is "maximum corruption" within that constraint.
MAX_BUDGET = 0.95


def test_realized_cost_never_exceeds_the_guard_bound_under_adversarial_corruption():
    graph = load_real_graph()
    features_df = load_features_df()
    oracle = load_oracle(features_df)
    ground_truth_cost_fn = predictor_cost_fn(oracle)
    adversary = AdversarialPredictor(oracle, budget=MAX_BUDGET)

    robust = RobustAStar(
        graph,
        planning_predictor=adversary,
        ground_truth_cost_fn=ground_truth_cost_fn,
        alpha=ALPHA,
        lambda_base=1.0,  # fully trust the (adversarial) predictor -- worst case for the guard
    )

    # Mostly cheap, direct instrumented-edge queries (guaranteed adversarial
    # exposure, and fast thanks to RobustAStar's bounded heuristic radius)
    # plus a smaller batch of longer cross-network queries for diversity.
    queries = direct_instrumented_queries(features_df, n=2500, seed=2024) + cross_network_queries(
        features_df, n=250, seed=2024
    )

    violations = []
    guard_invocations = 0
    reachable = 0

    for origin, dest, depart_time in queries:
        result = robust.search(origin, dest, depart_time)
        bound = result.robustness_bound
        if result.path is None:
            continue  # unreachable under BOTH classical and candidate -- no guarantee to check
        reachable += 1
        if bound.guard_invoked:
            guard_invocations += 1
        if bound.ratio > 1 + ALPHA + 1e-6:
            violations.append((origin, dest, depart_time, bound))

    assert reachable > 0, "fixture produced no reachable queries -- test is vacuous"
    assert not violations, (
        f"guard bound VIOLATED on {len(violations)}/{reachable} reachable queries "
        f"(alpha={ALPHA}): first violation = {violations[0]} -- "
        "the robustness guarantee does not hold as implemented, this is a "
        "paper-blocking bug, not a flaky test"
    )
    # Sanity: an adversary at near-maximum corruption, fully trusted, should
    # actually trigger the guard sometimes -- if it never did, the test
    # fixture likely isn't exercising the adversary at all (a vacuously
    # "passing" test that never touches the mechanism under test).
    assert guard_invocations > 0, (
        "guard was never invoked across all queries -- the adversarial "
        "predictor may not be corrupting anything the search actually uses; "
        "this test would then be passing for the wrong reason"
    )


def test_zero_alpha_forces_the_classical_path_whenever_the_candidate_differs():
    """alpha=0 is the tightest possible bound: the guard should reject any
    ML-guided candidate whose realized cost isn't <= the classical path's,
    which (since classical is a real shortest-path search over the SAME
    graph) means it should almost always just fall back to classical
    exactly, even under a maximally corrupting adversary."""
    graph = load_real_graph()
    features_df = load_features_df()
    oracle = load_oracle(features_df)
    ground_truth_cost_fn = predictor_cost_fn(oracle)
    adversary = AdversarialPredictor(oracle, budget=MAX_BUDGET)

    robust = RobustAStar(
        graph,
        planning_predictor=adversary,
        ground_truth_cost_fn=ground_truth_cost_fn,
        alpha=0.0,
        lambda_base=1.0,
    )

    queries = direct_instrumented_queries(features_df, n=200, seed=99)
    for origin, dest, depart_time in queries:
        result = robust.search(origin, dest, depart_time)
        if result.path is None:
            continue
        assert result.robustness_bound.ratio <= 1.0 + 1e-6
