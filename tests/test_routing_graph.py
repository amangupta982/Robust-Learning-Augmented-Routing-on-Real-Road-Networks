"""RoutingGraph construction and the real-graph loader.

The hand-built-graph assertions never touch real data. The real-graph load
is skipped if `make data` hasn't been run yet, matching the pattern in
tests/test_graph.py.
"""

from __future__ import annotations

import math

import pytest
from roar.graph.load_graph import GRAPH_PATH
from roar.routing.graph import haversine_m, load_road_graph

from tests.routing_fixtures import diamond_graph, grid_graph


def test_haversine_zero_distance_for_identical_points():
    assert haversine_m((34.0, -118.0), (34.0, -118.0)) == 0.0


def test_haversine_matches_known_one_degree_longitude_at_equator():
    # 1 degree of longitude at the equator is ~111.32 km.
    d = haversine_m((0.0, 0.0), (0.0, 1.0))
    assert math.isclose(d, 111_320, rel_tol=0.01)


def test_diamond_graph_adjacency_and_reverse_adjacency_are_consistent():
    g = diamond_graph()
    assert {e.edge_id for e in g.edges_from("A")} == {"AB", "AC"}
    assert {e.edge_id for e in g.edges_from("C")} == {"CB", "CD"}
    assert g.edges_from("D") == []

    # Every forward edge (u -> v) must have a matching reverse edge (v -> u)
    # with the same edge_id/length/speed, used by bidirectional_dijkstra.
    for node in ("A", "B", "C", "D"):
        for edge in g.edges_from(node):
            reverse_matches = [
                e for e in g.reverse_edges_from(edge.v) if e.edge_id == edge.edge_id
            ]
            assert len(reverse_matches) == 1
            rev = reverse_matches[0]
            assert rev.u == edge.v
            assert rev.v == edge.u
            assert rev.length_m == edge.length_m
            assert rev.speed_limit_mph == edge.speed_limit_mph


def test_max_speed_mph_is_the_graph_wide_maximum():
    g = diamond_graph()
    all_speeds = [e.speed_limit_mph for edges in g.adjacency.values() for e in edges]
    assert g.max_speed_mph == max(all_speeds)


def test_grid_graph_is_4_connected_interior_2_connected_corner():
    g = grid_graph(n=3)
    # Corner (0,0): only right and down neighbors -> 2 outgoing edges.
    assert len(g.edges_from("0_0")) == 2
    # Interior (1,1) of a 3x3 grid: up/down/left/right -> 4 outgoing edges.
    assert len(g.edges_from("1_1")) == 4


def test_load_road_graph_matches_real_data_if_built():
    if not GRAPH_PATH.exists():
        pytest.skip("real graph not built yet; run `make data` first")

    g = load_road_graph()
    assert len(g.nodes) > 0
    assert g.max_speed_mph > 0
    some_node = next(iter(g.nodes))
    # every node key must resolve in both adjacency dicts (even if empty)
    assert g.edges_from(some_node) is not None
    assert g.reverse_edges_from(some_node) is not None
