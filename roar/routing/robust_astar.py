"""RobustAStar: A* guided by a trust-parameter blend of a classical
admissible lower bound and an ML-model-based estimate, wrapped in a
`RobustnessGuard` (see guard.py) that delivers the actual provable
(1 + alpha) degradation bound.

## Heuristic

    h(n) = classical_lb(n) + lambda * (ml_estimate(n) - classical_lb(n))

  - `classical_lb(n)`: the admissible, predictor-free heuristic from
    `roar.routing.baselines.classical_lower_bound_heuristic` (great-circle
    distance / max speed limit) -- always available, needs no predictor.
  - `ml_estimate(n)`: the ML predictor's OWN belief about the shortest
    remaining cost from n to the destination, computed EXACTLY (not
    approximated) via a single backward Dijkstra run from the destination
    over the reversed graph, using the predictor's cost function
    (`predictor_cost_fn`). At lambda=1, h(n) == ml_estimate(n): the router
    fully trusts what its ML model believes about itself. At lambda=0,
    h(n) == classical_lb(n): pure classical A*, no ML influence at all.

IMPORTANT: this blended heuristic can be INADMISSIBLE (it may overestimate
true remaining cost whenever the predictor overestimates, and lambda > 0)
-- so the raw ML-guided search below is not guaranteed optimal, or even
guaranteed close to optimal. That is fine and expected: `astar()`'s
correctness ("returns a valid path if one exists, else None") never
depends on heuristic quality (see its docstring), only optimality does --
and optimality is not what delivers this module's safety property anyway.
**The (1 + alpha) guarantee comes entirely from RobustnessGuard, downstream
of this search, and is completely independent of how good or bad the
blended heuristic is.** A bad heuristic can only make the router slower or
propose a worse candidate (which the guard will then reject); it can never
break the guarantee.

## Lambda modes

  - **Fixed**: a single user-supplied `lambda_base` in [0, 1] used for
    every query, unmodified.
  - **Confidence-modulated** (`confidence_modulated=True`): computed ONCE
    per query (not per edge/node) as

        lambda_eff = lambda_base / (1 + mean_relative_sigma)

    where `mean_relative_sigma` is the predictor's average (sigma / eta)
    -- via `eta_with_confidence` -- over the classical path's edges that
    the predictor actually has coverage for. Higher predictor uncertainty
    on the route you'd actually consider taking -> lower effective trust.
    If the predictor has no coverage on any of those edges, there is
    nothing to be uncertain (or confident) about, so lambda_eff falls back
    to lambda_base unmodified.

## Frozen-weight approximation in `ml_estimate`

The backward Dijkstra that builds `ml_estimate` freezes edge costs at the
query's `depart_time` (the same approximation `bidirectional_dijkstra` uses
in baselines.py, for the same reason: exact time-dependent backward search
needs machinery out of scope here). As noted above, this only affects
search efficiency and candidate quality -- never the safety guarantee.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import heapq
import time

from roar.predictor.base import TravelTimePredictor
from roar.routing.baselines import (
    EdgeCostFn,
    astar,
    classical_lower_bound_heuristic,
    predictor_cost_fn,
    static_free_flow_cost,
)
from roar.routing.graph import RoutingGraph
from roar.routing.guard import RobustnessBound, RobustnessGuard


@dataclasses.dataclass
class RobustSearchResult:
    path: list[str] | None
    cost: float  # realized cost of the returned path (seconds); inf if unreachable
    node_expansions: int  # from the ML-guided candidate search
    latency_ms: float  # total wall-clock for the whole robust query
    lambda_used: float  # the (possibly confidence-modulated) lambda actually applied
    robustness_bound: RobustnessBound


def _ml_backward_distances(
    graph: RoutingGraph,
    dest: str,
    depart_time: dt.datetime,
    predictor: TravelTimePredictor,
    max_distance: float,
) -> dict[str, float]:
    """Exact shortest-path-to-`dest` cost under the ML predictor's own cost
    model, for every node within `max_distance` seconds of `dest` -- a
    single Dijkstra run backward from `dest` over the reversed graph (see
    module docstring for the frozen-weight caveat), stopped once the
    frontier passes `max_distance`.

    This is a heuristic table, not a correctness-critical value (see module
    docstring: the safety guarantee comes entirely from RobustnessGuard,
    independent of heuristic quality), so bounding it is safe: nodes beyond
    `max_distance` just fall back to `classical_lb` in the blended
    heuristic (via `.get(node, lb)`), exactly as if the ML model had no
    opinion there at all. Without this bound, a full unbounded run over the
    whole graph (tens of thousands of nodes) would dominate every query's
    latency for no benefit, since only nodes actually near the search
    ever matter to the forward A* run this table feeds."""
    cost_fn = predictor_cost_fn(predictor)
    dist: dict[str, float] = {dest: 0.0}
    visited: set[str] = set()
    heap: list[tuple[float, str]] = [(0.0, dest)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > max_distance:
            break
        if u in visited:
            continue
        visited.add(u)
        for edge in graph.reverse_edges_from(u):
            if edge.v in visited:
                continue
            nd = d + cost_fn(edge, depart_time)
            if edge.v not in dist or nd < dist[edge.v]:
                dist[edge.v] = nd
                heapq.heappush(heap, (nd, edge.v))
    return dist


class RobustAStar:
    def __init__(
        self,
        graph: RoutingGraph,
        planning_predictor: TravelTimePredictor,
        ground_truth_cost_fn: EdgeCostFn,
        alpha: float,
        lambda_base: float,
        confidence_modulated: bool = False,
        heuristic_radius_factor: float = 3.0,
    ):
        if not 0 <= lambda_base <= 1:
            raise ValueError(f"lambda_base must be in [0, 1], got {lambda_base}")
        self._graph = graph
        self._predictor = planning_predictor
        self._guard = RobustnessGuard(graph, ground_truth_cost_fn, alpha)
        self._lambda_base = lambda_base
        self._confidence_modulated = confidence_modulated
        # How far past the geometric (straight-line/max-speed) lower bound
        # between origin and dest the ml_estimate backward search bothers
        # to explore -- see _ml_backward_distances for why bounding this is
        # safe (it only trims a heuristic table, never the guarantee).
        self._heuristic_radius_factor = heuristic_radius_factor

    def _effective_lambda(
        self, classical_path: list[str] | None, depart_time: dt.datetime
    ) -> float:
        if not self._confidence_modulated:
            return self._lambda_base
        if classical_path is None or len(classical_path) < 2:
            return self._lambda_base

        relative_sigmas: list[float] = []
        elapsed = 0.0
        for u, v in zip(classical_path, classical_path[1:], strict=False):
            candidates = [e for e in self._graph.edges_from(u) if e.v == v]
            if not candidates:
                continue
            edge = candidates[0]
            current_time = depart_time + dt.timedelta(seconds=elapsed)
            try:
                eta, sigma = self._predictor.eta_with_confidence(edge.edge_id, current_time)
            except KeyError:
                elapsed += static_free_flow_cost(edge, current_time)
                continue
            relative_sigmas.append(sigma / max(eta, 1e-6))
            elapsed += eta

        if not relative_sigmas:
            return self._lambda_base
        mean_relative_sigma = sum(relative_sigmas) / len(relative_sigmas)
        return self._lambda_base / (1 + mean_relative_sigma)

    def search(self, origin: str, dest: str, depart_time: dt.datetime) -> RobustSearchResult:
        start_perf = time.perf_counter()

        classical = self._guard.classical_path(origin, dest, depart_time)
        lambda_eff = self._effective_lambda(classical.path, depart_time)

        classical_lb = classical_lower_bound_heuristic(self._graph, dest)
        max_distance = classical_lb(origin) * self._heuristic_radius_factor
        ml_estimate = _ml_backward_distances(
            self._graph, dest, depart_time, self._predictor, max_distance
        )

        def blended_heuristic(node: str) -> float:
            lb = classical_lb(node)
            ml = ml_estimate.get(node, lb)
            return lb + lambda_eff * (ml - lb)

        candidate = astar(
            self._graph,
            origin,
            dest,
            depart_time,
            predictor_cost_fn(self._predictor),
            heuristic=blended_heuristic,
        )

        final_path, bound = self._guard.apply(depart_time, classical, candidate)
        latency_ms = (time.perf_counter() - start_perf) * 1000
        return RobustSearchResult(
            path=final_path,
            cost=bound.realized_cost,
            node_expansions=candidate.node_expansions,
            latency_ms=latency_ms,
            lambda_used=lambda_eff,
            robustness_bound=bound,
        )
