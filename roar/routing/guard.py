"""RobustnessGuard: the mechanism that turns "A* guided by a possibly-wrong
ML predictor" into a router with a PROVABLE worst-case degradation bound --
the core contribution of this project (see CLAUDE.md: "the headline metric
is robustness, not prediction accuracy").

## The guarantee

For a configurable alpha >= 0, the path this guard returns always satisfies

    realized_cost(returned_path) <= (1 + alpha) * realized_cost(classical_path)

where:
  - `classical_path` is the path found by plain A*/Dijkstra using ONLY the
    static, free-flow (speed-limit-based) edge cost -- no predictor
    involved at all (`roar.routing.baselines.static_free_flow_cost`).
  - `realized_cost` is the TRUE cost of a *fixed* path, evaluated with a
    ground-truth cost function -- never the (possibly corrupted) predictor
    that proposed the candidate path. Checking a candidate against the very
    predictor that produced it would make the guarantee circular and
    vacuous: an adversarial predictor could simply under-report its own
    candidate's cost and always "pass".

## Why this is a compare-and-fallback, not a per-edge cost clamp

A tempting-sounding alternative is "clamp every edge's predicted cost to at
most (1 + alpha) times its own free-flow cost, then just run the ML-guided
search." This does NOT provably bound the guarantee above: per-edge
clamping only bounds each edge's LOCAL error relative to ITS OWN free-flow
cost. It says nothing about the GLOBAL path -- an ML-guided search could
still be lured onto a longer, differently-shaped path where every
individual edge is within its per-edge clamp, but the path visits enough
extra/longer edges that its TOTAL cost exceeds (1 + alpha) times the true
shortest path's cost. Bounding a sum by bounding its individual terms only
works if you also control how many terms there are and how that path
relates to the alternative -- a local, per-edge clamp cannot do that.

The only mechanism that provably bounds the aggregate is a **global
compare-and-fallback**: compute the ML-guided candidate path in full,
evaluate its REAL cost, and reject it in favor of the safe classical path
whenever it would violate the bound. That is exactly what `apply()` below
does -- this is the guard's "clamp," operating at the path level (the level
the guarantee is actually stated at) rather than the edge level.

## Assumption (stated explicitly, not glossed over)

This guard needs a ground-truth (or otherwise trusted) cost function to
evaluate a candidate path BEFORE deciding whether to return it. In this
project's offline historical-replay evaluation setting, that's
`OraclePredictor` (the real realized speeds, held out from training) --
appropriate here because every query replays a historical (origin, dest,
depart_time) whose outcome is already known. A live deployment has no such
foreknowledge and would need a different, online mechanism (e.g. monitoring
realized delays en route and replanning) -- that adaptation is out of scope
for this phase.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

from roar.routing.baselines import EdgeCostFn, SearchResult, astar, static_free_flow_cost
from roar.routing.graph import RoutingGraph


@dataclasses.dataclass(frozen=True)
class RobustnessBound:
    """`ratio` is realized_cost / classical_cost for the path actually
    returned -- by construction this is always <= 1 + alpha (see `apply`).
    `guard_invoked` is True iff the ML-guided candidate was rejected and the
    classical path was substituted."""

    alpha: float
    classical_cost: float
    realized_cost: float
    ratio: float
    guard_invoked: bool


def _safe_ratio(numerator: float, denominator: float) -> float:
    """0/0 (both paths cost nothing, e.g. origin == dest) is defined as a
    ratio of 1.0 -- trivially "met the bound exactly", not a division error."""
    if denominator == 0:
        return 1.0 if numerator == 0 else float("inf")
    return numerator / denominator


def path_realized_cost(
    graph: RoutingGraph,
    path: list[str] | None,
    cost_fn: EdgeCostFn,
    depart_time: dt.datetime,
) -> float:
    """Walks a *fixed* path edge by edge and sums `cost_fn` evaluated at
    each edge's real arrival time -- the single source of truth for "what
    did this path actually cost", independent of whichever cost model (if
    any) produced the path in the first place. `path=None` (unreachable)
    costs `inf`; a single-node path (origin == dest) costs 0."""
    if path is None:
        return float("inf")
    if len(path) <= 1:
        return 0.0

    total = 0.0
    for u, v in zip(path, path[1:], strict=False):
        candidates = [e for e in graph.edges_from(u) if e.v == v]
        if not candidates:
            return float("inf")  # not a valid edge in this graph
        current_time = depart_time + dt.timedelta(seconds=total)
        edge = min(candidates, key=lambda e: cost_fn(e, current_time))
        total += cost_fn(edge, current_time)
    return total


class RobustnessGuard:
    def __init__(self, graph: RoutingGraph, ground_truth_cost_fn: EdgeCostFn, alpha: float):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self._graph = graph
        self._ground_truth_cost_fn = ground_truth_cost_fn
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def classical_path(self, origin: str, dest: str, depart_time: dt.datetime) -> SearchResult:
        """The safe, predictor-free fallback: A* using only the static
        free-flow cost -- exactly Phase 3's `AStarBaseline`. Exposed as a
        convenience for callers (RobustAStar, tests) that don't already
        have a classical SearchResult on hand."""
        return astar(self._graph, origin, dest, depart_time, static_free_flow_cost)

    def apply(
        self,
        depart_time: dt.datetime,
        classical: SearchResult,
        candidate: SearchResult,
    ) -> tuple[list[str] | None, RobustnessBound]:
        """Decides whether `candidate` (an ML-guided search's result) is
        safe to return, or whether the guard must fall back to
        `classical`. Both must be SearchResults for the SAME (origin, dest,
        depart_time) query -- this function does not re-run any search, it
        only evaluates and compares already-computed candidates, so it can
        be tested in complete isolation with hand-built SearchResults (see
        tests/test_robustness_guard.py).

        Returns (final_path, bound) where `bound.ratio <= 1 + alpha` always
        holds for the returned path (the guarantee), by construction: if
        the candidate would violate it, classical is substituted instead,
        and classical trivially satisfies ratio == 1 <= 1 + alpha.
        """
        classical_realized = path_realized_cost(
            self._graph, classical.path, self._ground_truth_cost_fn, depart_time
        )

        if classical.path is None:
            # No path exists even under a real, non-ML search -- there is
            # no safe path to fall back to, and no meaningful ratio.
            return None, RobustnessBound(
                self._alpha, float("inf"), float("inf"), float("nan"), False
            )

        candidate_realized = path_realized_cost(
            self._graph, candidate.path, self._ground_truth_cost_fn, depart_time
        )
        bound_value = (1 + self._alpha) * classical_realized

        if candidate.path is not None and candidate_realized <= bound_value:
            ratio = _safe_ratio(candidate_realized, classical_realized)
            return candidate.path, RobustnessBound(
                self._alpha, classical_realized, candidate_realized, ratio, False
            )

        ratio = _safe_ratio(classical_realized, classical_realized)
        return classical.path, RobustnessBound(
            self._alpha, classical_realized, classical_realized, ratio, True
        )
