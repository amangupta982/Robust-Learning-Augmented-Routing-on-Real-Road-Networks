"""Correctness of bidirectional_dijkstra (the Contraction-Hierarchies
substitute -- see roar/routing/baselines.py's module docstring for why).
Must agree with plain Dijkstra's optimal cost on every hand-checkable graph,
using the same frozen-weight cost function so the comparison is apples to
apples."""

from __future__ import annotations

from roar.routing.baselines import bidirectional_dijkstra, dijkstra, static_free_flow_cost

from tests.routing_fixtures import (
    DEPART_TIME,
    diamond_graph,
    disconnected_graph,
    grid_graph,
    recompute_path_cost,
)


def test_matches_dijkstra_on_diamond_graph():
    g = diamond_graph()
    dij = dijkstra(g, "A", "D", DEPART_TIME, static_free_flow_cost)
    bd = bidirectional_dijkstra(g, "A", "D", DEPART_TIME, static_free_flow_cost)
    assert bd.cost == dij.cost == 10.0


def test_matches_dijkstra_on_grid_graph():
    g = grid_graph(n=5)
    dij = dijkstra(g, "0_0", "4_4", DEPART_TIME, static_free_flow_cost)
    bd = bidirectional_dijkstra(g, "0_0", "4_4", DEPART_TIME, static_free_flow_cost)
    assert abs(bd.cost - dij.cost) < 1e-6


def test_returned_path_is_valid_and_recomputes_to_the_reported_cost():
    g = diamond_graph()
    bd = bidirectional_dijkstra(g, "A", "D", DEPART_TIME, static_free_flow_cost)
    assert bd.path[0] == "A"
    assert bd.path[-1] == "D"
    assert recompute_path_cost(g, bd.path, static_free_flow_cost, DEPART_TIME) == bd.cost


def test_origin_equals_dest_is_zero_cost_single_node_path():
    g = diamond_graph()
    bd = bidirectional_dijkstra(g, "A", "A", DEPART_TIME, static_free_flow_cost)
    assert bd.path == ["A"]
    assert bd.cost == 0.0


def test_unreachable_destination_reports_none_path_and_infinite_cost():
    g = disconnected_graph()
    bd = bidirectional_dijkstra(g, "A", "C", DEPART_TIME, static_free_flow_cost)
    assert bd.path is None
    assert bd.cost == float("inf")
