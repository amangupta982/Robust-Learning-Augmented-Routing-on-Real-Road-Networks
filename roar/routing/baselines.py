"""Classical + pure-ML-guided routing baselines (CLAUDE.md Phase 3). These
are the "must be correct and fair" comparison points every later result
(Phase 4's RobustAStar included) is measured against.

All four share one interface -- `search(origin, dest, depart_time) ->
SearchResult(path, cost, node_expansions, latency_ms)` -- via the
`RoutingBaseline` classes at the bottom of this file, so the experiment
harness (Phase 5) can run every baseline over the exact same graph, cost
function, and query list.

## Time-dependent costs

`dijkstra()` and `astar()` are genuinely time-dependent: the edge cost is
evaluated at the current accumulated arrival time (`depart_time +
elapsed`), not just at `depart_time`, using whatever `EdgeCostFn` is passed
in (`static_free_flow_cost`, or `predictor_cost_fn(predictor)` which
queries a `TravelTimePredictor` and falls back to free-flow for edges/times
it has no coverage for).

## Contraction Hierarchies substitution

We substitute **bidirectional Dijkstra** for full Contraction Hierarchies.
CH requires an offline contraction phase (node ordering + shortcut-edge
construction) that is a substantial piece of engineering on its own and
orthogonal to this paper's contribution (the robustness guarantee, not
query-speed engineering) -- building a correct CH implementation would risk
introducing exactly the kind of "impressive but subtly wrong" baseline this
phase is supposed to avoid. Bidirectional Dijkstra is simple to prove
correct, is a standard strong speed baseline in the routing literature, and
still meaningfully differentiates "a real speedup technique" from plain
Dijkstra/A*.

Caveat, stated plainly rather than silently glossed over: `bidirectional_dijkstra()`
freezes edge costs at `depart_time` for the whole query (a "frozen-weight
snapshot") rather than updating them as the search progresses. Exact
time-dependent bidirectional search requires additional machinery (e.g. a
consistent lower-bounding potential for the backward frontier) that is out
of scope here. This makes it a fast baseline for *routing-graph traversal
speed*, not a second fully time-dependent baseline -- `dijkstra()` and
`astar()` remain the time-dependent references.

## Pure-ML-guided A*

`PureMLAStarBaseline` is exactly `astar()` with `cost_fn =
predictor_cost_fn(predictor)` and no other change -- it trusts whatever the
injected `TravelTimePredictor` says, with no clamping or sanity check
against how wrong that predictor might be. That is deliberate: it is the
"unsafe" baseline Phase 4's `RobustAStar` (same predictor, wrapped in a
trust-parameter-bounded `RobustnessGuard`) needs to beat on the
robustness sweep.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import heapq
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from roar.predictor.base import MPH_TO_MPS, TravelTimePredictor, travel_time_seconds
from roar.routing.graph import Edge, RoutingGraph, haversine_m

EdgeCostFn = Callable[[Edge, dt.datetime], float]


@dataclasses.dataclass
class SearchResult:
    path: list[str] | None  # node ids, origin..dest inclusive; None if unreachable
    cost: float  # seconds; inf if unreachable
    node_expansions: int
    latency_ms: float


def static_free_flow_cost(edge: Edge, current_time: dt.datetime) -> float:
    """Free-flow travel time from the edge's posted speed limit -- the
    fallback for the ~99.8% of edges METR-LA doesn't instrument (see
    results/data_quality.md), and the only cost model bidirectional_dijkstra
    needs since it doesn't consult a predictor."""
    return travel_time_seconds(edge.length_m, edge.speed_limit_mph)


def predictor_cost_fn(predictor: TravelTimePredictor) -> EdgeCostFn:
    """Wraps any concrete TravelTimePredictor as an EdgeCostFn (CLAUDE.md
    rule 5: this is the only predictor-specific code in the module, and it
    depends on the interface, never a concrete class). Falls back to
    static_free_flow_cost on KeyError -- the documented behavior of
    OraclePredictor/LightGBMPredictor for edges or times they have no
    coverage for."""

    def cost(edge: Edge, current_time: dt.datetime) -> float:
        try:
            return predictor.eta(edge.edge_id, current_time)
        except KeyError:
            return static_free_flow_cost(edge, current_time)

    return cost


def classical_lower_bound_heuristic(graph: RoutingGraph, dest: str) -> Callable[[str], float]:
    """h(n) = great_circle_distance(n, dest) / max_speed_mph -- the
    admissible, consistent heuristic `astar()` uses by default (see its
    docstring for the proof). Exposed as a standalone function so
    `roar/routing/robust_astar.py` can reuse the exact same classical lower
    bound as one term of its blended heuristic, rather than risking a second,
    possibly-divergent copy of this logic."""
    max_speed_mps = graph.max_speed_mph * MPH_TO_MPS

    def h(node: str) -> float:
        return haversine_m(graph.nodes[node], graph.nodes[dest]) / max_speed_mps

    return h


def _reconstruct_path(prev: dict[str, str], origin: str, dest: str) -> list[str]:
    path = [dest]
    while path[-1] != origin:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def dijkstra(
    graph: RoutingGraph,
    origin: str,
    dest: str,
    depart_time: dt.datetime,
    cost_fn: EdgeCostFn,
) -> SearchResult:
    """Classic time-dependent Dijkstra: labels are arrival times, and each
    edge's cost is evaluated at the arrival time of the node it's relaxed
    from, so cost_fn sees the real depart-time-shifted query time."""
    start_perf = time.perf_counter()

    dist: dict[str, float] = {origin: 0.0}
    prev: dict[str, str] = {}
    visited: set[str] = set()
    heap: list[tuple[float, str]] = [(0.0, origin)]
    expansions = 0

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        expansions += 1
        if u == dest:
            break
        current_time = depart_time + dt.timedelta(seconds=d)
        for edge in graph.edges_from(u):
            if edge.v in visited:
                continue
            nd = d + cost_fn(edge, current_time)
            if edge.v not in dist or nd < dist[edge.v]:
                dist[edge.v] = nd
                prev[edge.v] = u
                heapq.heappush(heap, (nd, edge.v))

    latency_ms = (time.perf_counter() - start_perf) * 1000
    if dest not in dist:
        return SearchResult(None, float("inf"), expansions, latency_ms)
    return SearchResult(_reconstruct_path(prev, origin, dest), dist[dest], expansions, latency_ms)


def astar(
    graph: RoutingGraph,
    origin: str,
    dest: str,
    depart_time: dt.datetime,
    cost_fn: EdgeCostFn,
    heuristic: Callable[[str], float] | None = None,
) -> SearchResult:
    """A* search. Default heuristic (used whenever `heuristic` is not
    supplied): h(n) = great_circle_distance(n, dest) / max_speed_mph (the
    graph-wide max speed limit, converted to m/s) -- see
    `classical_lower_bound_heuristic`.

    Admissibility of the default heuristic: the driving-network shortest
    path between n and dest is always >= their great-circle distance (a
    straight line is the shortest path between two points; any real route
    can only be as long or longer). Every edge's free-flow speed is <=
    max_speed_mph by construction (max_speed_mph is defined as the max over
    all edges in the graph), so the true minimum travel time from n to dest
    satisfies
        true_time(n, dest) = sum(edge_length_i / edge_speed_i)
                           >= sum(edge_length_i) / max_speed_mph
                           >= great_circle(n, dest) / max_speed_mph
                           == h(n).
    h never overestimates the true remaining cost -> admissible.

    Consistency: great_circle(., .) satisfies the triangle inequality (it's
    a metric), so for any edge (n, m): great_circle(n, dest) <=
    great_circle(n, m) + great_circle(m, dest), hence
        h(n) <= great_circle(n, m) / max_speed_mph + h(m).
    And great_circle(n, m) / max_speed_mph <= actual_cost(n, m) by the same
    admissibility argument applied to a single edge. Substituting:
        h(n) <= actual_cost(n, m) + h(m)
    which is exactly the monotonicity / consistency condition. A consistent
    heuristic on a graph with non-negative edge weights never requires
    re-expanding an already-settled node (this is why `visited` below is a
    simple set, not a re-openable structure).

    A caller-supplied `heuristic` (see roar/routing/robust_astar.py) need
    NOT be admissible/consistent -- correctness of "return a valid path if
    one exists, else None" never depends on the heuristic (the loop only
    gives up when the heap empties, never based on h's value), only
    *optimality* of the returned path does. An inadmissible heuristic can
    make this function return a valid but suboptimal path; it can never
    make it return a wrong/invalid one.
    """
    start_perf = time.perf_counter()
    h = heuristic if heuristic is not None else classical_lower_bound_heuristic(graph, dest)

    g: dict[str, float] = {origin: 0.0}
    prev: dict[str, str] = {}
    visited: set[str] = set()
    heap: list[tuple[float, str]] = [(h(origin), origin)]
    expansions = 0

    while heap:
        _, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        expansions += 1
        if u == dest:
            break
        current_time = depart_time + dt.timedelta(seconds=g[u])
        for edge in graph.edges_from(u):
            if edge.v in visited:
                continue
            ng = g[u] + cost_fn(edge, current_time)
            if edge.v not in g or ng < g[edge.v]:
                g[edge.v] = ng
                prev[edge.v] = u
                heapq.heappush(heap, (ng + h(edge.v), edge.v))

    latency_ms = (time.perf_counter() - start_perf) * 1000
    if dest not in g:
        return SearchResult(None, float("inf"), expansions, latency_ms)
    return SearchResult(_reconstruct_path(prev, origin, dest), g[dest], expansions, latency_ms)


def bidirectional_dijkstra(
    graph: RoutingGraph,
    origin: str,
    dest: str,
    depart_time: dt.datetime,
    cost_fn: EdgeCostFn,
) -> SearchResult:
    """Bidirectional Dijkstra with frozen (depart-time-snapshot) edge costs
    -- see the module docstring for why this substitutes for Contraction
    Hierarchies and why the weights are frozen rather than time-dependent.

    Standard alternating-frontier algorithm: expand whichever of the
    forward/backward frontiers has the smaller top-of-heap key, track the
    best complete forward+backward path seen at any meeting node, and stop
    once the sum of both frontiers' smallest keys can no longer improve on
    it -- the classic correctness condition for bidirectional Dijkstra.
    """
    start_perf = time.perf_counter()
    if origin == dest:
        return SearchResult([origin], 0.0, 1, (time.perf_counter() - start_perf) * 1000)

    frozen_time = depart_time

    dist_f: dict[str, float] = {origin: 0.0}
    dist_b: dict[str, float] = {dest: 0.0}
    prev_f: dict[str, str] = {}
    prev_b: dict[str, str] = {}
    visited_f: set[str] = set()
    visited_b: set[str] = set()
    heap_f: list[tuple[float, str]] = [(0.0, origin)]
    heap_b: list[tuple[float, str]] = [(0.0, dest)]
    expansions = 0
    best = float("inf")
    meet_node: str | None = None

    def relax(u: str, edges: list[Edge], dist, prev, heap, other_dist) -> None:
        nonlocal best, meet_node
        for edge in edges:
            nd = dist[u] + cost_fn(edge, frozen_time)
            neighbor = edge.v
            if neighbor not in dist or nd < dist[neighbor]:
                dist[neighbor] = nd
                prev[neighbor] = u
                heapq.heappush(heap, (nd, neighbor))
            if neighbor in other_dist:
                total = dist[neighbor] + other_dist[neighbor]
                if total < best:
                    best = total
                    meet_node = neighbor

    while heap_f and heap_b:
        if heap_f[0][0] + heap_b[0][0] >= best:
            break
        if heap_f[0][0] <= heap_b[0][0]:
            d, u = heapq.heappop(heap_f)
            if u in visited_f:
                continue
            visited_f.add(u)
            expansions += 1
            relax(u, graph.edges_from(u), dist_f, prev_f, heap_f, dist_b)
        else:
            d, u = heapq.heappop(heap_b)
            if u in visited_b:
                continue
            visited_b.add(u)
            expansions += 1
            relax(u, graph.reverse_edges_from(u), dist_b, prev_b, heap_b, dist_f)

    latency_ms = (time.perf_counter() - start_perf) * 1000
    if meet_node is None:
        return SearchResult(None, float("inf"), expansions, latency_ms)

    path_to_meet = (
        [origin] if meet_node == origin else _reconstruct_path(prev_f, origin, meet_node)
    )
    meet_to_dest = (
        [dest] if meet_node == dest else _reconstruct_path(prev_b, dest, meet_node)
    )
    path = path_to_meet + list(reversed(meet_to_dest))[1:]
    return SearchResult(path, best, expansions, latency_ms)


class RoutingBaseline(ABC):
    @abstractmethod
    def search(self, origin: str, dest: str, depart_time: dt.datetime) -> SearchResult: ...


class DijkstraBaseline(RoutingBaseline):
    def __init__(self, graph: RoutingGraph, cost_fn: EdgeCostFn):
        self._graph = graph
        self._cost_fn = cost_fn

    def search(self, origin: str, dest: str, depart_time: dt.datetime) -> SearchResult:
        return dijkstra(self._graph, origin, dest, depart_time, self._cost_fn)


class AStarBaseline(RoutingBaseline):
    def __init__(self, graph: RoutingGraph, cost_fn: EdgeCostFn):
        self._graph = graph
        self._cost_fn = cost_fn

    def search(self, origin: str, dest: str, depart_time: dt.datetime) -> SearchResult:
        return astar(self._graph, origin, dest, depart_time, self._cost_fn)


class BidirectionalDijkstraBaseline(RoutingBaseline):
    def __init__(self, graph: RoutingGraph, cost_fn: EdgeCostFn):
        self._graph = graph
        self._cost_fn = cost_fn

    def search(self, origin: str, dest: str, depart_time: dt.datetime) -> SearchResult:
        return bidirectional_dijkstra(self._graph, origin, dest, depart_time, self._cost_fn)


class PureMLAStarBaseline(RoutingBaseline):
    """The "unsafe" baseline: A* with edge costs taken directly from the
    injected predictor, no robustness mechanism. See module docstring."""

    def __init__(self, graph: RoutingGraph, predictor: TravelTimePredictor):
        self._graph = graph
        self._cost_fn = predictor_cost_fn(predictor)

    def search(self, origin: str, dest: str, depart_time: dt.datetime) -> SearchResult:
        return astar(self._graph, origin, dest, depart_time, self._cost_fn)
