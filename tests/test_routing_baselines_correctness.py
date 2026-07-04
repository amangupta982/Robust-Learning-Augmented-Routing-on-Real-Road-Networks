"""Correctness of Dijkstra and A* on small, hand-checkable graphs.

CLAUDE.md: "These MUST be correct and fair -- weak baselines invalidate the
whole paper." Every expected value here is computed by hand in
tests/routing_fixtures.py's docstring, not derived from the code under test.
"""

from __future__ import annotations

from roar.routing.baselines import astar, dijkstra, static_free_flow_cost

from tests.routing_fixtures import (
    DEPART_TIME,
    diamond_graph,
    disconnected_graph,
    grid_graph,
    recompute_path_cost,
)


def test_dijkstra_matches_hand_computed_optimum_on_diamond_graph():
    g = diamond_graph()
    result = dijkstra(g, "A", "D", DEPART_TIME, static_free_flow_cost)
    assert result.cost == 10.0
    assert result.path == ["A", "C", "B", "D"]


def test_astar_matches_dijkstra_cost_on_diamond_graph():
    g = diamond_graph()
    dij = dijkstra(g, "A", "D", DEPART_TIME, static_free_flow_cost)
    ast = astar(g, "A", "D", DEPART_TIME, static_free_flow_cost)
    assert ast.cost == dij.cost == 10.0
    assert ast.path == dij.path


def test_returned_path_recomputes_to_the_reported_cost():
    g = diamond_graph()
    result = dijkstra(g, "A", "D", DEPART_TIME, static_free_flow_cost)
    assert recompute_path_cost(g, result.path, static_free_flow_cost, DEPART_TIME) == result.cost


def test_origin_equals_dest_is_zero_cost_single_node_path():
    g = diamond_graph()
    for search in (dijkstra, astar):
        result = search(g, "A", "A", DEPART_TIME, static_free_flow_cost)
        assert result.path == ["A"]
        assert result.cost == 0.0


def test_unreachable_destination_reports_none_path_and_infinite_cost():
    g = disconnected_graph()
    for search in (dijkstra, astar):
        result = search(g, "A", "C", DEPART_TIME, static_free_flow_cost)
        assert result.path is None
        assert result.cost == float("inf")


def test_astar_and_dijkstra_agree_on_grid_graph_and_astar_expands_no_more_nodes():
    g = grid_graph(n=5)
    dij = dijkstra(g, "0_0", "4_4", DEPART_TIME, static_free_flow_cost)
    ast = astar(g, "0_0", "4_4", DEPART_TIME, static_free_flow_cost)

    assert dij.path is not None
    assert abs(ast.cost - dij.cost) < 1e-6
    # A consistent, informative heuristic must never expand more nodes than
    # plain Dijkstra (it can only prune search, never require extra work).
    assert ast.node_expansions <= dij.node_expansions
    # On a 5x5 grid with a real geographic heuristic pointed at the goal,
    # the pruning should be more than a rounding artifact.
    assert ast.node_expansions < dij.node_expansions


def test_node_expansions_never_exceed_reachable_node_count():
    g = grid_graph(n=4)
    dij = dijkstra(g, "0_0", "3_3", DEPART_TIME, static_free_flow_cost)
    assert dij.node_expansions <= len(g.nodes)
