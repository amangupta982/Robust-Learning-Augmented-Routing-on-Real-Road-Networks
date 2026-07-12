"""Improvement Phase Task 1d: re-run the latency-vs-graph-size sweep using
the REAL LightGBM predictor (both the unoptimized base predictor and the
materialized/optimized one from roar/predictor/materialized_lightgbm.py),
on the SAME graph sizes, query-sampling seed, and query count as the
original scaling experiment (roar/eval/scaling.py) -- imported from there,
not re-typed, so this is a fair like-for-like comparison.

This does NOT modify or overwrite roar/eval/scaling.py or its output
(results/latency_vs_graph_size.parquet, which used OraclePredictor and
backs the already-reported Fig 3) -- CLAUDE.md's improvement-phase rules
forbid altering a previously reported metric's definition. This is a
genuinely NEW measurement (LightGBM's real per-query cost, not Oracle's),
written to a NEW file: results/latency_vs_graph_size_lightgbm.parquet.

Why re-run with LightGBM at all: profiling (Task 1a, see
roar/predictor/materialized_lightgbm.py's docstring and IMPROVEMENTS.md)
was done against LightGBM specifically, since that is the predictor a real
deployment would use (OraclePredictor is not deployable -- it requires
already knowing the ground truth). The original Fig 3 measured Oracle's
latency, which is real but not the "deployability" number the optimization
in this phase actually targets; this file is the honest LightGBM-based
counterpart.

Run directly: `python -m roar.eval.scaling_lightgbm`.
"""

from __future__ import annotations

import datetime as dt
import random

import pandas as pd

from roar.eval.metrics import make_provenance, resolve_model_dir
from roar.eval.scaling import (
    ALPHA,
    GRAPH_SIZES,
    LAMBDA_BASE,
    N_QUERIES_PER_SIZE,
    SEED,
    bfs_subgraph,
    sample_reachable_queries,
)
from roar.graph.config import REPO_ROOT
from roar.graph.features import FEATURES_PATH
from roar.predictor.lightgbm import LightGBMPredictor
from roar.predictor.materialized_lightgbm import MaterializedLightGBMPredictor
from roar.routing.baselines import predictor_cost_fn
from roar.routing.graph import load_road_graph
from roar.routing.robust_astar import RobustAStar

RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "latency_vs_graph_size_lightgbm.parquet"

DEPART_TIME = dt.datetime(2012, 6, 4, 8, 0)


def run_scaling_experiment_lightgbm() -> pd.DataFrame:
    print("Loading the real LA graph, features, and trained LightGBM model ...")
    full_graph = load_road_graph()
    features_df = pd.read_parquet(FEATURES_PATH)
    model_dir = resolve_model_dir()
    base_predictor = LightGBMPredictor.load(model_dir, features_df)

    print("Materializing the LightGBM predictor (Task 1b) -- one-time cost, timed separately ...")
    t0 = dt.datetime.now(dt.UTC)
    materialized_predictor = MaterializedLightGBMPredictor(base_predictor, features_df)
    materialization_seconds = (dt.datetime.now(dt.UTC) - t0).total_seconds()
    print(f"  materialization took {materialization_seconds:.1f}s (a one-time startup cost, "
          "not a per-query cost -- reported separately, not folded into latency_ms below)")

    provenance = make_provenance()

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

        for predictor_variant, predictor in (
            ("base", base_predictor),
            ("materialized", materialized_predictor),
        ):
            ground_truth_cost_fn = predictor_cost_fn(predictor)
            robust = RobustAStar(
                subgraph, predictor, ground_truth_cost_fn, alpha=ALPHA, lambda_base=LAMBDA_BASE
            )
            for query_index, (origin, dest) in enumerate(queries):
                result = robust.search(origin, dest, DEPART_TIME)
                rows.append(
                    {
                        "graph_size": actual_size,
                        "requested_size": size,
                        "method": "robust_astar",
                        "predictor_variant": predictor_variant,
                        "query_index": query_index,
                        "latency_ms": result.latency_ms,
                        "node_expansions": result.node_expansions,
                        "materialization_seconds": (
                            materialization_seconds if predictor_variant == "materialized" else 0.0
                        ),
                        "seed": SEED,
                        "git_commit": provenance["git_commit"],
                        "run_timestamp": provenance["run_timestamp"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    df = run_scaling_experiment_lightgbm()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"Saved {len(df)} rows to {OUT_PATH}")

    summary = df.groupby(["graph_size", "predictor_variant"])["latency_ms"].mean().unstack()
    print(summary)


if __name__ == "__main__":
    main()
