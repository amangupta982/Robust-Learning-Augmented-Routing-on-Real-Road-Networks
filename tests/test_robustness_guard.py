"""RobustnessGuard.apply() tested in complete isolation: hand-built
SearchResults with known costs, no real search involved. This is possible
by design (see guard.py's docstring) -- `apply()` is a pure function of
(classical, candidate) plus a ground-truth cost function, which lets us
pin down the guard's decision logic exactly, independent of any search
algorithm or predictor.
"""

from __future__ import annotations

import pytest
from roar.routing.baselines import SearchResult
from roar.routing.guard import RobustnessGuard, path_realized_cost

from tests.routing_fixtures import DEPART_TIME, diamond_graph


def _result(path, cost):
    return SearchResult(path=path, cost=cost, node_expansions=0, latency_ms=0.0)


def test_candidate_within_bound_is_returned_unmodified():
    g = diamond_graph()
    guard = RobustnessGuard(g, ground_truth_cost_fn=lambda e, t: e.length_m, alpha=0.5)
    classical = _result(["A", "C", "B", "D"], 10.0)  # real cost 10
    candidate = _result(["A", "B", "D"], 12.0)  # real cost 12 <= 1.5*10=15

    path, bound = guard.apply(DEPART_TIME, classical, candidate)
    assert path == ["A", "B", "D"]
    assert bound.guard_invoked is False
    assert bound.classical_cost == 10.0
    assert bound.realized_cost == 12.0
    assert bound.ratio == 1.2


def test_candidate_violating_bound_is_rejected_in_favor_of_classical():
    g = diamond_graph()
    guard = RobustnessGuard(g, ground_truth_cost_fn=lambda e, t: e.length_m, alpha=0.2)
    classical = _result(["A", "C", "B", "D"], 10.0)
    candidate = _result(["A", "C", "D"], 25.0)  # real cost 25 > 1.2*10=12

    path, bound = guard.apply(DEPART_TIME, classical, candidate)
    assert path == ["A", "C", "B", "D"]
    assert bound.guard_invoked is True
    assert bound.classical_cost == 10.0
    assert bound.realized_cost == 10.0
    assert bound.ratio == 1.0


def test_candidate_exactly_at_the_bound_is_accepted():
    """guard.apply() ignores the (untrusted) SearchResult.cost field and
    independently recomputes realized cost from the graph's real edges --
    so this picks alpha such that A-B-D's REAL cost (10 + 2 = 12) lands
    exactly on (1 + alpha) * 10, to exercise the "<=" boundary precisely."""
    g = diamond_graph()
    guard = RobustnessGuard(g, ground_truth_cost_fn=lambda e, t: e.length_m, alpha=0.2)
    classical = _result(["A", "C", "B", "D"], 10.0)
    candidate = _result(["A", "B", "D"], 999.0)  # bogus self-reported cost, must be ignored

    path, bound = guard.apply(DEPART_TIME, classical, candidate)
    assert path == ["A", "B", "D"]
    assert bound.guard_invoked is False
    assert bound.realized_cost == 12.0
    assert bound.ratio == 1.2


def test_unreachable_candidate_falls_back_to_classical():
    g = diamond_graph()
    guard = RobustnessGuard(g, ground_truth_cost_fn=lambda e, t: e.length_m, alpha=0.5)
    classical = _result(["A", "C", "B", "D"], 10.0)
    candidate = _result(None, float("inf"))

    path, bound = guard.apply(DEPART_TIME, classical, candidate)
    assert path == ["A", "C", "B", "D"]
    assert bound.guard_invoked is True


def test_unreachable_classical_means_no_safe_path_at_all():
    g = diamond_graph()
    guard = RobustnessGuard(g, ground_truth_cost_fn=lambda e, t: e.length_m, alpha=0.5)
    classical = _result(None, float("inf"))
    candidate = _result(["A", "B", "D"], 12.0)

    path, bound = guard.apply(DEPART_TIME, classical, candidate)
    assert path is None
    assert bound.guard_invoked is False  # nothing to "invoke" -- no safe path exists at all


def test_origin_equals_dest_is_a_degenerate_zero_cost_case():
    g = diamond_graph()
    guard = RobustnessGuard(g, ground_truth_cost_fn=lambda e, t: e.length_m, alpha=0.5)
    classical = _result(["A"], 0.0)
    candidate = _result(["A"], 0.0)

    path, bound = guard.apply(DEPART_TIME, classical, candidate)
    assert path == ["A"]
    assert bound.ratio == 1.0
    assert bound.guard_invoked is False


def test_negative_alpha_is_rejected():
    g = diamond_graph()
    with pytest.raises(ValueError, match="alpha"):
        RobustnessGuard(g, ground_truth_cost_fn=lambda e, t: e.length_m, alpha=-0.1)


def test_path_realized_cost_matches_hand_computed_value():
    g = diamond_graph()
    cost = path_realized_cost(g, ["A", "C", "B", "D"], lambda e, t: e.length_m, DEPART_TIME)
    assert cost == 5.0 + 3.0 + 2.0


def test_path_realized_cost_of_none_is_infinite():
    g = diamond_graph()
    assert path_realized_cost(g, None, lambda e, t: e.length_m, DEPART_TIME) == float("inf")


def test_path_realized_cost_of_single_node_is_zero():
    g = diamond_graph()
    assert path_realized_cost(g, ["A"], lambda e, t: e.length_m, DEPART_TIME) == 0.0
