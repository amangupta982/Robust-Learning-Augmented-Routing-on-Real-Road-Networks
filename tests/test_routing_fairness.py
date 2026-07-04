"""Fairness test (CLAUDE.md Phase 3): every baseline must be run on the
exact same graph, costs, and queries -- otherwise a "wins on robustness"
result could just mean "got an easier graph."

PureMLAStarBaseline is given a predictor with zero coverage
(AlwaysMissingPredictor), so its effective edge costs fall back to the same
static_free_flow_cost the other three baselines use directly -- making all
four genuinely comparable on this fixture.
"""

from __future__ import annotations

from roar.routing.baselines import (
    AStarBaseline,
    BidirectionalDijkstraBaseline,
    DijkstraBaseline,
    PureMLAStarBaseline,
    static_free_flow_cost,
)

from tests.routing_fixtures import (
    DEPART_TIME,
    AlwaysMissingPredictor,
    diamond_graph,
    grid_graph,
    recompute_path_cost,
)


def _all_baselines(graph):
    return {
        "dijkstra": DijkstraBaseline(graph, static_free_flow_cost),
        "astar": AStarBaseline(graph, static_free_flow_cost),
        "bidirectional_dijkstra": BidirectionalDijkstraBaseline(graph, static_free_flow_cost),
        "pure_ml_astar": PureMLAStarBaseline(graph, AlwaysMissingPredictor()),
    }


def test_all_four_baselines_agree_on_cost_for_the_same_query_on_diamond_graph():
    graph = diamond_graph()
    results = {
        name: baseline.search("A", "D", DEPART_TIME)
        for name, baseline in _all_baselines(graph).items()
    }
    costs = {name: r.cost for name, r in results.items()}
    assert len(set(costs.values())) == 1, f"baselines disagree on cost: {costs}"
    assert costs["dijkstra"] == 10.0


def test_all_four_baselines_agree_on_cost_for_the_same_query_on_grid_graph():
    graph = grid_graph(n=5)
    results = {
        name: baseline.search("0_0", "4_4", DEPART_TIME)
        for name, baseline in _all_baselines(graph).items()
    }
    costs = {name: r.cost for name, r in results.items()}
    reference = costs["dijkstra"]
    for name, cost in costs.items():
        assert abs(cost - reference) < 1e-6, f"{name} disagrees: {cost} vs {reference}"


def test_every_baselines_returned_path_independently_recomputes_to_its_cost():
    graph = diamond_graph()
    for name, baseline in _all_baselines(graph).items():
        result = baseline.search("A", "D", DEPART_TIME)
        recomputed = recompute_path_cost(graph, result.path, static_free_flow_cost, DEPART_TIME)
        assert recomputed == result.cost, f"{name}: path doesn't match its own reported cost"


def test_multiple_queries_are_run_over_the_identical_graph_instance():
    """Same RoutingGraph object across queries and baselines -- guards
    against a baseline accidentally mutating shared graph state."""
    graph = grid_graph(n=4)
    baselines = _all_baselines(graph)
    queries = [
        ("0_0", "3_3", DEPART_TIME),
        ("0_3", "3_0", DEPART_TIME),
        ("1_1", "2_2", DEPART_TIME),
    ]

    for origin, dest, depart in queries:
        costs = {name: b.search(origin, dest, depart).cost for name, b in baselines.items()}
        reference = costs["dijkstra"]
        for name, cost in costs.items():
            assert abs(cost - reference) < 1e-6, f"{name} disagrees on {origin}->{dest}: {costs}"
