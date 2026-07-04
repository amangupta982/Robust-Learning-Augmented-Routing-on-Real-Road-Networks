"""PureMLAStarBaseline: A* that trusts a TravelTimePredictor's eta directly,
with no robustness mechanism -- the "unsafe" baseline Phase 4's RobustAStar
must beat on the robustness sweep.
"""

from __future__ import annotations

from roar.routing.baselines import PureMLAStarBaseline, predictor_cost_fn

from tests.routing_fixtures import (
    DEPART_TIME,
    AlwaysMissingPredictor,
    StubPredictor,
    diamond_graph,
    recompute_path_cost,
)


def test_falls_back_to_free_flow_cost_when_predictor_has_no_coverage():
    g = diamond_graph()
    baseline = PureMLAStarBaseline(g, AlwaysMissingPredictor())
    result = baseline.search("A", "D", DEPART_TIME)
    # With zero predictor coverage every edge cost falls back to
    # static_free_flow_cost, so this must match the hand-computed optimum.
    assert result.cost == 10.0
    assert result.path == ["A", "C", "B", "D"]


def test_blindly_follows_a_predictor_that_disagrees_with_free_flow_cost():
    g = diamond_graph()
    # Free-flow shortest path is A-C-B-D (cost 10). Tell the predictor that
    # edge CB actually takes 100s (e.g. reporting real gridlock) -- a
    # predictor-trusting router with NO robustness mechanism must re-route
    # around it even though this could be a wildly wrong/adversarial reading.
    predictor = StubPredictor({"CB": 100.0})
    baseline = PureMLAStarBaseline(g, predictor)
    result = baseline.search("A", "D", DEPART_TIME)

    assert result.path == ["A", "B", "D"]  # now cheaper: 10 + 2 = 12 < 5+100+2=107
    assert result.cost == 12.0
    assert recompute_path_cost(g, result.path, predictor_cost_fn(predictor), DEPART_TIME) == (
        result.cost
    )
