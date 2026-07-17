"""Improvement Phase Task 2: unit tests for RoutingAwareAdversarialPredictor
on small, hand-checkable graphs (no real data needed) -- verifying the
threat model actually does what its docstring claims: identify the true
corridor and a genuine alternate trap, overestimate the former,
underestimate the latter, and (critically) that RobustAStar's guard still
holds even when a naive router would be lured onto the trap.
"""

from __future__ import annotations

import pytest
from roar.predictor.routing_aware_adversarial import RoutingAwareAdversarialPredictor
from roar.routing.baselines import PureMLAStarBaseline, predictor_cost_fn
from roar.routing.graph import Edge, build_routing_graph
from roar.routing.robust_astar import RobustAStar

from tests.routing_fixtures import DEPART_TIME, UNIT_SPEED_MPH, StubPredictor, diamond_graph


def _detour_graph():
    """A->B->D (cost 10) is the true optimum; A->C->D (cost 16) is a
    genuinely worse, fully edge-disjoint alternate path -- deliberately
    NOT diamond_graph, which has no complete detour once its shortest
    path's edges are blocked (see test_degenerate_graph_with_no_detour)."""
    nodes = {n: (0.0, 0.0) for n in ("A", "B", "C", "D")}
    edges = [
        Edge("AB", "A", "B", 5.0, UNIT_SPEED_MPH),
        Edge("BD", "B", "D", 5.0, UNIT_SPEED_MPH),
        Edge("AC", "A", "C", 8.0, UNIT_SPEED_MPH),
        Edge("CD", "C", "D", 8.0, UNIT_SPEED_MPH),
    ]
    graph = build_routing_graph(nodes, edges)
    oracle = StubPredictor({"AB": 5.0, "BD": 5.0, "AC": 8.0, "CD": 8.0})
    return graph, oracle


def test_identifies_the_true_corridor_and_a_genuine_trap():
    graph, oracle = _detour_graph()
    adversary = RoutingAwareAdversarialPredictor(
        graph, oracle, budget=1.0, origin="A", dest="D", depart_time=DEPART_TIME
    )
    assert adversary.corridor_edges == {"AB", "BD"}
    assert adversary.trap_edges == {"AC", "CD"}


def test_overestimates_corridor_and_underestimates_trap():
    graph, oracle = _detour_graph()
    adversary = RoutingAwareAdversarialPredictor(
        graph, oracle, budget=1.0, origin="A", dest="D", depart_time=DEPART_TIME
    )
    # corridor: true * (1 + budget)
    assert adversary.eta("AB", DEPART_TIME) == pytest.approx(5.0 * 2)
    assert adversary.eta("BD", DEPART_TIME) == pytest.approx(5.0 * 2)
    # trap: true / (1 + budget)
    assert adversary.eta("AC", DEPART_TIME) == pytest.approx(8.0 / 2)
    assert adversary.eta("CD", DEPART_TIME) == pytest.approx(8.0 / 2)


def test_budget_zero_is_truthful():
    graph, oracle = _detour_graph()
    adversary = RoutingAwareAdversarialPredictor(
        graph, oracle, budget=0.0, origin="A", dest="D", depart_time=DEPART_TIME
    )
    for edge_id, true_eta in [("AB", 5.0), ("BD", 5.0), ("AC", 8.0), ("CD", 8.0)]:
        assert adversary.eta(edge_id, DEPART_TIME) == pytest.approx(true_eta)


def test_negative_budget_is_rejected():
    graph, oracle = _detour_graph()
    with pytest.raises(ValueError, match="budget"):
        RoutingAwareAdversarialPredictor(
            graph, oracle, budget=-0.1, origin="A", dest="D", depart_time=DEPART_TIME
        )


def test_with_budget_reuses_corridor_and_trap_without_recomputation():
    graph, oracle = _detour_graph()
    adversary = RoutingAwareAdversarialPredictor(
        graph, oracle, budget=0.25, origin="A", dest="D", depart_time=DEPART_TIME
    )
    rebudgeted = adversary.with_budget(2.0)
    assert rebudgeted.corridor_edges == adversary.corridor_edges
    assert rebudgeted.trap_edges == adversary.trap_edges
    assert rebudgeted.eta("AC", DEPART_TIME) == pytest.approx(8.0 / 3)
    # original is untouched
    assert adversary.eta("AC", DEPART_TIME) == pytest.approx(8.0 / 1.25)


def test_degenerate_graph_reuses_a_penalized_corridor_edge_when_forced():
    """diamond_graph's shortest path A-C-B-D uses edges that make the
    corridor nearly a cut-set: once {AC, CB, BD} are blocked, the ONLY way
    to reach D at all is A-B-D, which itself must reuse BD (blocked). The
    large FINITE sentinel (see _compute_corridor_and_trap's docstring --
    NOT literal infinity, which would crash astar()) means the "detour"
    search still finds this path rather than reporting none: it is simply
    the cheapest of the (all heavily penalized) remaining options. The
    resulting trap_edges legitimately OVERLAPS corridor_edges on "BD" here
    -- an honest consequence of this graph having poor alternate-route
    diversity around D, not a bug (eta() resolves the overlap by treating
    a corridor/trap edge as trap -- see its docstring)."""
    graph = diamond_graph()
    oracle = StubPredictor({"AB": 10.0, "AC": 5.0, "CB": 3.0, "BD": 2.0, "CD": 20.0})
    adversary = RoutingAwareAdversarialPredictor(
        graph, oracle, budget=1.0, origin="A", dest="D", depart_time=DEPART_TIME
    )
    assert adversary.corridor_edges == {"AC", "CB", "BD"}
    assert adversary.trap_edges == {"AB", "BD"}


def test_pure_ml_astar_is_lured_onto_the_trap_but_robust_astar_is_not():
    graph, oracle = _detour_graph()
    ground_truth_cost_fn = predictor_cost_fn(oracle)
    adversary = RoutingAwareAdversarialPredictor(
        graph, oracle, budget=1.0, origin="A", dest="D", depart_time=DEPART_TIME
    )

    pure_ml = PureMLAStarBaseline(graph, adversary)
    lured = pure_ml.search("A", "D", DEPART_TIME)
    assert lured.path == ["A", "C", "D"]  # the trap: true cost 16, worse than the true optimum 10

    robust = RobustAStar(graph, adversary, ground_truth_cost_fn, alpha=0.1, lambda_base=1.0)
    protected = robust.search("A", "D", DEPART_TIME)
    assert protected.path == ["A", "B", "D"]  # the guard falls back to the TRUE optimum
    assert protected.cost == pytest.approx(10.0)
    assert protected.robustness_bound.guard_invoked is True
