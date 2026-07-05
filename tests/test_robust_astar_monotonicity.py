"""CRITICAL test (Phase 4): Monotonicity smoke test.

With NoisyPredictor at increasing relative noise levels (sigma_level),
average path quality should never drop below the guard floor: no matter
how noisy the predictor gets, every single query's realized cost must still
satisfy ratio <= 1 + alpha (the same guarantee the robustness test checks
under adversarial corruption -- noise is a weaker, non-adversarial
perturbation, so if the guard holds against a worst-case adversary it must
also hold here; this test exists to catch a DIFFERENT class of bug: one
that only shows up with genuinely random, non-adversarial errors).

As a secondary smoke check, the guard should fire more often (not less) as
the predictor gets noisier -- a rough monotonic trend, checked loosely
(highest vs. lowest sigma) rather than strictly pointwise, since individual
noise draws aren't monotonic by construction.
"""

from __future__ import annotations

import pytest
from roar.predictor.noisy import NoisyPredictor
from roar.routing.baselines import predictor_cost_fn
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

ALPHA = 0.3
SIGMA_LEVELS = [0.0, 0.2, 0.5, 1.0, 2.0]


def _run_at_sigma_level(graph, oracle, ground_truth_cost_fn, queries, sigma_level):
    noisy = NoisyPredictor(oracle, sigma_level=sigma_level, seed=7)
    robust = RobustAStar(
        graph,
        planning_predictor=noisy,
        ground_truth_cost_fn=ground_truth_cost_fn,
        alpha=ALPHA,
        lambda_base=1.0,
    )
    ratios = []
    guard_invocations = 0
    for origin, dest, depart_time in queries:
        result = robust.search(origin, dest, depart_time)
        if result.path is None:
            continue
        ratios.append(result.robustness_bound.ratio)
        if result.robustness_bound.guard_invoked:
            guard_invocations += 1
    return ratios, guard_invocations


def test_guard_floor_holds_at_every_noise_level_and_invocation_rate_trends_up():
    graph = load_real_graph()
    features_df = load_features_df()
    oracle = load_oracle(features_df)
    ground_truth_cost_fn = predictor_cost_fn(oracle)
    queries = direct_instrumented_queries(features_df, n=400, seed=555)

    invocation_rates = []
    for sigma_level in SIGMA_LEVELS:
        ratios, guard_invocations = _run_at_sigma_level(
            graph, oracle, ground_truth_cost_fn, queries, sigma_level
        )
        assert ratios, f"sigma_level={sigma_level}: fixture produced no reachable queries"

        max_ratio = max(ratios)
        mean_ratio = sum(ratios) / len(ratios)
        assert max_ratio <= 1 + ALPHA + 1e-6, (
            f"sigma_level={sigma_level}: guard floor violated, max ratio={max_ratio} "
            f"> 1+alpha={1 + ALPHA} -- average path quality dropped below the "
            "guard floor, this is a paper-blocking bug"
        )
        assert mean_ratio <= 1 + ALPHA + 1e-6

        invocation_rates.append(guard_invocations / len(ratios))

    # Smoke check: noisier predictions should make the guard fire at least
    # as often at the highest sigma as at sigma=0 (perfect predictor -- the
    # guard should essentially never need to fire there).
    assert invocation_rates[-1] >= invocation_rates[0], (
        f"guard invocation rate did not trend up with noise: {invocation_rates} "
        f"for sigma levels {SIGMA_LEVELS} -- either noise isn't reaching the "
        "search, or the guard's triggering logic is miscalibrated"
    )
