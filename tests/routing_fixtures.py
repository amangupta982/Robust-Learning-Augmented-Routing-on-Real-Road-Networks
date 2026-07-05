"""Hand-built, hand-checkable graphs shared by the routing baseline tests.

Not a test module itself (no test_ prefix) -- imported by the tests in this
package. Every edge uses a fixed speed_limit_mph of `1 / MPH_TO_MPS`, chosen
so that `travel_time_seconds(length_m, speed_limit_mph) == length_m`
numerically: this lets edge "length_m" double as a plain, hand-computable
edge weight in seconds while still going through the real production cost
path (roar.routing.baselines.static_free_flow_cost), not a test-only
shortcut.
"""

from __future__ import annotations

import datetime as dt

from roar.predictor.base import MPH_TO_MPS, TravelTimePredictor
from roar.routing.baselines import EdgeCostFn
from roar.routing.graph import Edge, RoutingGraph, build_routing_graph, haversine_m

UNIT_SPEED_MPH = 1 / MPH_TO_MPS  # travel_time_seconds(length_m, this) == length_m

DEPART_TIME = dt.datetime(2012, 6, 4, 8, 0)


def diamond_graph() -> RoutingGraph:
    """Four nodes, all at the same coordinate (heuristic h=0 everywhere --
    trivially admissible, and reduces A* to Dijkstra so their costs and
    expansion counts must match exactly).

    Edges (weight = length_m, in seconds):
      A->B: 10   A->C: 5   C->B: 3   B->D: 2   C->D: 20

    Hand-computed shortest paths from A:
      A->B direct:        10
      A->C->B:            5 + 3 = 8   (beats the direct A->B edge)
      A->C->B->D:         8 + 2 = 10
      A->B->D:            10 + 2 = 12
      A->C->D:            5 + 20 = 25
    So the shortest A->D path is A-C-B-D with cost 10.
    """
    nodes = {n: (0.0, 0.0) for n in ("A", "B", "C", "D")}
    edges = [
        Edge("AB", "A", "B", 10.0, UNIT_SPEED_MPH),
        Edge("AC", "A", "C", 5.0, UNIT_SPEED_MPH),
        Edge("CB", "C", "B", 3.0, UNIT_SPEED_MPH),
        Edge("BD", "B", "D", 2.0, UNIT_SPEED_MPH),
        Edge("CD", "C", "D", 20.0, UNIT_SPEED_MPH),
    ]
    return build_routing_graph(nodes, edges)


def disconnected_graph() -> RoutingGraph:
    """A and B are connected; C is an isolated node with no edges at all --
    dest=C from origin=A must be reported unreachable, not silently wrong."""
    nodes = {"A": (0.0, 0.0), "B": (0.0, 0.0), "C": (1.0, 1.0)}
    edges = [Edge("AB", "A", "B", 5.0, UNIT_SPEED_MPH)]
    return build_routing_graph(nodes, edges)


def grid_graph(n: int = 5, step_deg: float = 0.01) -> RoutingGraph:
    """An n x n grid with cardinal (4-connected) moves only and real
    lat/lon coordinates, so the great-circle heuristic is genuinely
    informative (not just trivially zero) -- this is the fixture used to
    check that A* expands <= Dijkstra's node count."""
    nodes: dict[str, tuple[float, float]] = {}
    for i in range(n):
        for j in range(n):
            nodes[f"{i}_{j}"] = (i * step_deg, j * step_deg)

    edges: list[Edge] = []
    for i in range(n):
        for j in range(n):
            node = f"{i}_{j}"
            if i + 1 < n:
                nb = f"{i + 1}_{j}"
                length = haversine_m(nodes[node], nodes[nb])
                edges.append(Edge(f"{node}-{nb}", node, nb, length, UNIT_SPEED_MPH))
                edges.append(Edge(f"{nb}-{node}", nb, node, length, UNIT_SPEED_MPH))
            if j + 1 < n:
                nb = f"{i}_{j + 1}"
                length = haversine_m(nodes[node], nodes[nb])
                edges.append(Edge(f"{node}-{nb}", node, nb, length, UNIT_SPEED_MPH))
                edges.append(Edge(f"{nb}-{node}", nb, node, length, UNIT_SPEED_MPH))
    return build_routing_graph(nodes, edges)


class AlwaysMissingPredictor(TravelTimePredictor):
    """A predictor with zero coverage -- every query raises KeyError, so
    roar.routing.baselines.predictor_cost_fn always falls back to
    static_free_flow_cost. Used to make PureMLAStarBaseline directly
    comparable to the other (predictor-free) baselines in the fairness
    test: same graph, same effective costs, same queries."""

    def eta(self, edge_id: str, depart_time: dt.datetime) -> float:
        raise KeyError(edge_id)

    def eta_with_confidence(self, edge_id: str, depart_time: dt.datetime) -> tuple[float, float]:
        raise KeyError(edge_id)


class StubPredictor(TravelTimePredictor):
    """Returns a fixed, hand-picked eta (and optional sigma) per edge_id
    regardless of depart_time -- used to prove PureMLAStarBaseline trusts
    the predictor directly (no clamping/robustness mechanism), and to drive
    RobustAStar's confidence-modulated lambda with known sigma values."""

    def __init__(
        self, eta_by_edge: dict[str, float], sigma_by_edge: dict[str, float] | None = None
    ):
        self._eta_by_edge = eta_by_edge
        self._sigma_by_edge = sigma_by_edge or {}

    def eta(self, edge_id: str, depart_time: dt.datetime) -> float:
        if edge_id not in self._eta_by_edge:
            raise KeyError(edge_id)
        return self._eta_by_edge[edge_id]

    def eta_with_confidence(self, edge_id: str, depart_time: dt.datetime) -> tuple[float, float]:
        return self.eta(edge_id, depart_time), self._sigma_by_edge.get(edge_id, 0.0)


def recompute_path_cost(
    graph: RoutingGraph, path: list[str], cost_fn: EdgeCostFn, depart_time: dt.datetime
) -> float:
    """Independently re-derives a path's total cost by walking its edges,
    used to sanity-check a SearchResult.path against its own SearchResult.cost."""
    total = 0.0
    # path[1:] is deliberately one element shorter than path -- this pairs
    # up consecutive (u, v) nodes along the path, not two independent
    # same-length sequences.
    for u, v in zip(path, path[1:], strict=False):
        candidates = [e for e in graph.edges_from(u) if e.v == v]
        assert candidates, f"path uses a non-existent edge {u} -> {v}"
        edge = min(candidates, key=lambda e: cost_fn(e, depart_time + dt.timedelta(seconds=total)))
        total += cost_fn(edge, depart_time + dt.timedelta(seconds=total))
    return total
