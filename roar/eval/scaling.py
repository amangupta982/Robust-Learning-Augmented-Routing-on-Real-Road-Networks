"""Fig 3 data (latency vs. graph size, deployability): a real graph-size
scaling experiment.

Phase 5's per-query result parquets (roar/eval/harness.py) all ran on the
single, full-size LA graph -- there is no existing data with multiple graph
sizes to plot latency against. Rather than fabricate one, this builds
genuine SUBGRAPHS of the real LA network at increasing sizes via BFS
expansion from a fixed root node (a standard, honest technique for
scalability studies: real topology, just a real subset of it), and times
every method on the same queries at each size.

Run directly: `python -m roar.eval.scaling` (writes
results/latency_vs_graph_size.parquet). Included in `make experiments`.
"""

from __future__ import annotations

import datetime as dt
import random
from collections import deque

import pandas as pd

from roar.eval.metrics import make_provenance
from roar.graph.config import REPO_ROOT
from roar.graph.features import FEATURES_PATH
from roar.predictor.oracle import OraclePredictor
from roar.routing.baselines import (
    AStarBaseline,
    BidirectionalDijkstraBaseline,
    DijkstraBaseline,
    predictor_cost_fn,
    static_free_flow_cost,
)
from roar.routing.graph import RoutingGraph, build_routing_graph, load_road_graph
from roar.routing.robust_astar import RobustAStar

RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "latency_vs_graph_size.parquet"

# None = the full graph, at whatever size it actually is.
GRAPH_SIZES: list[int | None] = [1000, 2000, 5000, 10000, 20000, None]
N_QUERIES_PER_SIZE = 15
MAX_SAMPLE_ATTEMPTS_PER_QUERY = 20
SEED = 909
ALPHA = 0.3
LAMBDA_BASE = 1.0


def bfs_subgraph(graph: RoutingGraph, root: str, max_nodes: int) -> RoutingGraph:
    """The induced subgraph on the first `max_nodes` nodes reached by a
    breadth-first expansion from `root` over the real forward adjacency --
    a real, connected piece of the actual LA network, not a fabricated
    graph."""
    visited = {root}
    order = [root]
    queue = deque([root])
    while queue and len(order) < max_nodes:
        u = queue.popleft()
        for edge in graph.edges_from(u):
            if edge.v not in visited:
                visited.add(edge.v)
                order.append(edge.v)
                queue.append(edge.v)
                if len(order) >= max_nodes:
                    break

    node_set = set(order)
    nodes = {n: graph.nodes[n] for n in node_set}
    edges = [edge for n in node_set for edge in graph.edges_from(n) if edge.v in node_set]
    return build_routing_graph(nodes, edges)


def sample_reachable_queries(
    graph: RoutingGraph, n: int, rng: random.Random
) -> list[tuple[str, str]]:
    """Random (origin, dest) pairs verified reachable (via plain Dijkstra)
    -- a truncated BFS subgraph isn't guaranteed strongly connected, so
    unreachable draws are discarded rather than silently included as
    zero-cost/failed rows."""
    nodes = list(graph.nodes.keys())
    dijkstra = DijkstraBaseline(graph, static_free_flow_cost)
    depart_time = dt.datetime(2012, 6, 4, 8, 0)
    pairs: list[tuple[str, str]] = []
    attempts = 0
    while len(pairs) < n and attempts < n * MAX_SAMPLE_ATTEMPTS_PER_QUERY:
        attempts += 1
        origin, dest = rng.choice(nodes), rng.choice(nodes)
        if origin == dest:
            continue
        if dijkstra.search(origin, dest, depart_time).path is not None:
            pairs.append((origin, dest))
    return pairs


def run_scaling_experiment() -> pd.DataFrame:
    print("Loading the real LA graph and oracle predictor for the scaling experiment ...")
    full_graph = load_road_graph()
    features_df = pd.read_parquet(FEATURES_PATH)
    oracle = OraclePredictor(features_df)
    ground_truth_cost_fn = predictor_cost_fn(oracle)
    provenance = make_provenance()
    depart_time = dt.datetime(2012, 6, 4, 8, 0)

    rng = random.Random(SEED)
    root = rng.choice(list(full_graph.nodes.keys()))

    rows: list[dict] = []
    for size in GRAPH_SIZES:
        subgraph = full_graph if size is None else bfs_subgraph(full_graph, root, size)
        actual_size = len(subgraph.nodes)
        print(f"  graph_size={actual_size} (requested {size or 'full'}) ...")

        queries = sample_reachable_queries(subgraph, N_QUERIES_PER_SIZE, rng)
        if not queries:
            print(f"    no reachable query pairs found at size {actual_size}, skipping")
            continue

        classical_methods = {
            "dijkstra": DijkstraBaseline(subgraph, static_free_flow_cost),
            "astar": AStarBaseline(subgraph, static_free_flow_cost),
            "bidirectional_dijkstra": BidirectionalDijkstraBaseline(
                subgraph, static_free_flow_cost
            ),
        }
        robust = RobustAStar(
            subgraph, oracle, ground_truth_cost_fn, alpha=ALPHA, lambda_base=LAMBDA_BASE
        )

        for query_index, (origin, dest) in enumerate(queries):
            for name, baseline in classical_methods.items():
                result = baseline.search(origin, dest, depart_time)
                rows.append(
                    {
                        "graph_size": actual_size,
                        "requested_size": size,
                        "method": name,
                        "query_index": query_index,
                        "latency_ms": result.latency_ms,
                        "node_expansions": result.node_expansions,
                        "seed": SEED,
                        "git_commit": provenance["git_commit"],
                        "run_timestamp": provenance["run_timestamp"],
                    }
                )
            robust_result = robust.search(origin, dest, depart_time)
            rows.append(
                {
                    "graph_size": actual_size,
                    "requested_size": size,
                    "method": "robust_astar",
                    "query_index": query_index,
                    "latency_ms": robust_result.latency_ms,
                    "node_expansions": robust_result.node_expansions,
                    "seed": SEED,
                    "git_commit": provenance["git_commit"],
                    "run_timestamp": provenance["run_timestamp"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = run_scaling_experiment()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"Saved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
