"""Routing-aware adversarial predictor -- Improvement Phase Task 2.

## Why this exists

The existing `AdversarialPredictor` (roar/predictor/adversarial.py) applies
a STATIC, per-edge multiplicative underestimate uniformly to every edge,
regardless of the query. Phase 5/6 found this too weak an instrument: at
budget=0.95 it produced no significant separation between guarded and
unguarded routing on this project's query set (results/figures/
table1_headline_metrics.csv's robustness_sweep row is marked "ns"),
because a uniform corruption doesn't specifically target the one thing
that actually changes a routing decision -- the RELATIVE cost between the
true-optimal corridor and its alternatives.

## Threat model (stated explicitly, per this phase's hard rules)

This adversary is deliberately much stronger and far less realistic than
any real ML miscalibration: it has FULL KNOWLEDGE of the query (origin,
destination, depart_time) and of the TRUE ground-truth costs (an oracle).
Given that knowledge, for THIS specific query it:

  1. Computes the true oracle-optimal path (the "good corridor").
  2. Computes a genuinely worse alternate path that avoids every good-
     corridor edge (the "trap" -- found by re-routing with the good
     corridor's edges' cost set to infinity).
  3. UNDERESTIMATES every edge on the trap (by budget B, via eta/(1+B) --
     see below for why not eta*(1-B)), making the genuinely bad detour
     look artificially cheap.
  4. OVERESTIMATES every edge on the good corridor (via eta*(1+B)), making
     the genuinely good path look artificially expensive.
  5. Reports every edge elsewhere at its TRUE cost -- this adversary does
     not blanket-lie; it spends its entire budget concentrating the attack
     where it can do the most damage, which is the point of "routing-aware"
     and "greedy."

If the (1+alpha) guard bound holds even against an adversary that KNOWS
the truth and lies with surgical, query-specific precision, that is much
stronger evidence for the guarantee than holding against a uniform
corruption. If it is EVER violated, that is a genuine correctness bug in
the guard (roar/routing/guard.py) -- outranking everything else in this
phase (Task 2c) -- not a property of this adversary.

## Why eta/(1+B), not eta*(1-B), for the underestimate direction

The original AdversarialPredictor requires budget < 1 (eta*(1-B) goes
negative/zero at B >= 1). This task sweeps B up to 2.0, so a formula that
stays strictly positive for any B >= 0 is needed: eta/(1+B) is always > 0
and monotonically decreasing in B (eta/2 at B=1, eta/3 at B=2), the same
qualitative "lure" direction as the original predictor, just defined for
this wider budget range.

## Scope note

Unlike every other predictor in this project, this one is NOT a stateless
per-edge function of (edge_id, depart_time) alone -- it is constructed
PER QUERY (it needs to know origin/destination to compute the corridor),
so it cannot be built once and reused across a whole sweep the way
roar/eval/harness.py's other predictors are. See
roar/eval/adversarial_experiment.py, which constructs a fresh instance per
query rather than retrofitting roar/eval/harness.py's per-sweep-point
predictor-construction assumption.
"""

from __future__ import annotations

import copy
import datetime as dt

from roar.predictor.base import TravelTimePredictor
from roar.predictor.oracle import OraclePredictor
from roar.routing.baselines import astar, predictor_cost_fn
from roar.routing.graph import RoutingGraph

_MIN_ETA_S = 1e-3


def _path_edge_ids(graph: RoutingGraph, path: list[str], cost_fn, depart_time) -> set[str]:
    """Same u->v parallel-edge disambiguation as
    roar.routing.guard.path_realized_cost (a real OSM graph can have more
    than one directed edge between the same two nodes) -- here used to
    identify WHICH edges a path uses, not to accumulate its cost."""
    edge_ids: set[str] = set()
    for u, v in zip(path, path[1:], strict=False):
        candidates = [e for e in graph.edges_from(u) if e.v == v]
        if not candidates:
            continue
        edge = min(candidates, key=lambda e: cost_fn(e, depart_time))
        edge_ids.add(edge.edge_id)
    return edge_ids


class RoutingAwareAdversarialPredictor(TravelTimePredictor):
    def __init__(
        self,
        graph: RoutingGraph,
        oracle: OraclePredictor,
        budget: float,
        origin: str,
        dest: str,
        depart_time: dt.datetime,
    ):
        if budget < 0:
            raise ValueError(f"budget must be >= 0, got {budget}")
        self._oracle = oracle
        self._budget = budget
        self.corridor_edges, self.trap_edges = self._compute_corridor_and_trap(
            graph, oracle, origin, dest, depart_time
        )

    def with_budget(self, budget: float) -> RoutingAwareAdversarialPredictor:
        """A new adversary for the SAME query (same corridor/trap edge
        sets, computed once) at a different budget -- sweeping budget over
        a fixed query would otherwise redundantly re-run the two A*
        searches in `_compute_corridor_and_trap` once per budget value for
        no reason, since the corridor/trap depend only on the query, never
        on the budget."""
        if budget < 0:
            raise ValueError(f"budget must be >= 0, got {budget}")
        clone = copy.copy(self)
        clone._budget = budget
        return clone

    @staticmethod
    def _compute_corridor_and_trap(
        graph: RoutingGraph,
        oracle: OraclePredictor,
        origin: str,
        dest: str,
        depart_time: dt.datetime,
    ) -> tuple[set[str], set[str]]:
        ground_truth_cost_fn = predictor_cost_fn(oracle)
        optimal = astar(graph, origin, dest, depart_time, ground_truth_cost_fn)
        if optimal.path is None or len(optimal.path) < 2:
            return set(), set()
        corridor_edges = _path_edge_ids(graph, optimal.path, ground_truth_cost_fn, depart_time)

        def blocked_cost_fn(edge, t):
            if edge.edge_id in corridor_edges:
                # A literal float("inf") here crashes astar()/dijkstra():
                # they compute `depart_time + timedelta(seconds=g[u])` for
                # every popped node, and timedelta cannot represent an
                # infinite duration (OverflowError) -- a latent gap in
                # roar/routing/baselines.py that no prior caller ever hit,
                # since nothing before this adversary used an infinite
                # edge cost. Rather than change that tested Phase 3/4 code
                # (this phase's hard rule 2), use a large FINITE sentinel:
                # no real edge costs more than a few minutes, so 1e7
                # seconds (~116 days) is never preferred over any real
                # alternative, and stays safely within datetime's range
                # even summed across many blocked edges on one path.
                return 1e7
            return ground_truth_cost_fn(edge, t)

        detour = astar(graph, origin, dest, depart_time, blocked_cost_fn)
        if detour.path is None or len(detour.path) < 2:
            # No detour exists that avoids every corridor edge (e.g. a
            # corridor edge is a cut/bridge) -- the adversary still
            # overestimates the corridor below, it just has no
            # complementary trap to lure onto for this query.
            return corridor_edges, set()
        trap_edges = _path_edge_ids(graph, detour.path, ground_truth_cost_fn, depart_time)
        return corridor_edges, trap_edges

    def eta(self, edge_id: str, depart_time: dt.datetime) -> float:
        """Trap is checked before corridor: on a graph where the corridor
        is nearly a cut-set, the "detour" search may be forced to reuse a
        penalized corridor edge (see _compute_corridor_and_trap's
        docstring), making that one edge_id a member of BOTH sets. In that
        case it is treated as trap (underestimated) -- an honest
        consequence of the graph having poor alternate-route diversity
        there, not an arbitrary tie-break (see
        tests/test_routing_aware_adversarial.py's degenerate-graph test)."""
        true_eta = self._oracle.eta(edge_id, depart_time)
        if edge_id in self.trap_edges:
            return max(true_eta / (1 + self._budget), _MIN_ETA_S)
        if edge_id in self.corridor_edges:
            return true_eta * (1 + self._budget)
        return true_eta

    def eta_with_confidence(self, edge_id: str, depart_time: dt.datetime) -> tuple[float, float]:
        return self.eta(edge_id, depart_time), 0.0
