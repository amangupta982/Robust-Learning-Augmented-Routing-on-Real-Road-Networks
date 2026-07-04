"""Lightweight routing graph representation.

Deliberately decoupled from OSMnx/networkx graph objects (beyond the one
loader function below) so routing algorithms can be exercised against tiny,
hand-checkable graphs in tests as well as the real LA drive network -- the
same `RoutingGraph` type serves both.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pandas as pd

from roar.graph.load_graph import EDGES_PATH, GRAPH_PATH

EARTH_RADIUS_M = 6_371_000.0


@dataclasses.dataclass(frozen=True)
class Edge:
    edge_id: str
    u: str
    v: str
    length_m: float
    speed_limit_mph: float


@dataclasses.dataclass(frozen=True)
class RoutingGraph:
    """`nodes`: node_id -> (lat, lon) in degrees, used only by the A*
    heuristic. `adjacency`/`reverse_adjacency`: node_id -> outgoing/incoming
    Edge list (`reverse_adjacency` holds each edge with u/v swapped, for
    backward search -- edge_id/length_m/speed_limit_mph are unchanged, so
    cost lookups by edge_id are identical in either direction).
    `max_speed_mph`: the graph-wide max speed limit, used by the A*
    heuristic (see roar/routing/baselines.py for the admissibility proof)."""

    nodes: dict[str, tuple[float, float]]
    adjacency: dict[str, list[Edge]]
    reverse_adjacency: dict[str, list[Edge]]
    max_speed_mph: float

    def edges_from(self, node: str) -> list[Edge]:
        return self.adjacency.get(node, [])

    def reverse_edges_from(self, node: str) -> list[Edge]:
        return self.reverse_adjacency.get(node, [])


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in meters between two (lat, lon) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def build_routing_graph(
    nodes: dict[str, tuple[float, float]], edges: list[Edge]
) -> RoutingGraph:
    adjacency: dict[str, list[Edge]] = {n: [] for n in nodes}
    reverse_adjacency: dict[str, list[Edge]] = {n: [] for n in nodes}
    max_speed = 0.0
    for e in edges:
        adjacency.setdefault(e.u, []).append(e)
        adjacency.setdefault(e.v, [])
        reverse_adjacency.setdefault(
            e.v, []
        ).append(Edge(e.edge_id, e.v, e.u, e.length_m, e.speed_limit_mph))
        reverse_adjacency.setdefault(e.u, [])
        max_speed = max(max_speed, e.speed_limit_mph)
    return RoutingGraph(
        nodes=nodes, adjacency=adjacency, reverse_adjacency=reverse_adjacency,
        max_speed_mph=max_speed,
    )


def load_road_graph(
    graph_path: Path = GRAPH_PATH, edges_path: Path = EDGES_PATH
) -> RoutingGraph:
    """Loads the real Phase 1 LA drive network (`make data` output)."""
    import osmnx as ox  # imported lazily: only the real-data loader needs it

    if not graph_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f"{graph_path} / {edges_path} not found. Run `make data` first."
        )

    graph = ox.load_graphml(graph_path)
    nodes = {str(n): (float(data["y"]), float(data["x"])) for n, data in graph.nodes(data=True)}

    edges_df = pd.read_parquet(
        edges_path, columns=["edge_id", "u", "v", "length_m", "speed_limit_mph"]
    )
    edges = [
        Edge(row.edge_id, str(row.u), str(row.v), float(row.length_m), float(row.speed_limit_mph))
        for row in edges_df.itertuples()
    ]
    return build_routing_graph(nodes, edges)
