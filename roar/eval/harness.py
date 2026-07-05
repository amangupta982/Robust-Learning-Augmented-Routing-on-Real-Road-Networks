"""Phase 5 experiment harness: reads experiments/configs/*.yaml, runs every
baseline + RobustAStar variant over the SAME stratified query set, and logs
one clean parquet per experiment to results/<run_id>.parquet.

`make experiments` runs `python -m roar.eval.harness`, which:
  1. Builds the query set ONCE from experiments/configs/query_set.yaml
     (real graph nodes, real METR-LA-instrumented edges, real test-split
     timestamps -- see roar/eval/queries.py) and reuses that exact
     in-memory list for every experiment below, so "every method ran on
     identical queries" holds by construction, not by matching seeds.
  2. Computes the ground-truth-optimal cost per query ONCE (the
     competitive-ratio denominator for every method).
  3. Reads experiments/configs/experiments.yaml (the ordered list of
     experiment configs to run) and, for each, runs every requested method
     over the shared query set and writes results/<run_id>.parquet.

No plotting here -- Phase 6 turns these parquet logs into figures/tables
(CLAUDE.md rule 4: every reported number must be regenerable from this
command, no hand-edited numbers anywhere).

Runtime note: `experiments/configs/query_set.yaml` ships with n_queries=200
(not literally "thousands") so a full `make experiments` finishes in
tens of minutes rather than hours in a typical dev environment --
increase `n_queries` there for a full-scale paper run; nothing else in
this module changes.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from roar.eval.metrics import (
    EvalContext,
    build_predictor,
    compute_oracle_optimal_costs,
    make_provenance,
    make_row,
)
from roar.eval.queries import Query, generate_stratified_queries
from roar.graph.config import REPO_ROOT, load_config
from roar.graph.features import FEATURES_PATH
from roar.predictor.lightgbm import PREDICTOR_CONFIG_PATH
from roar.predictor.oracle import OraclePredictor
from roar.routing.baselines import (
    AStarBaseline,
    BidirectionalDijkstraBaseline,
    DijkstraBaseline,
    PureMLAStarBaseline,
    static_free_flow_cost,
)
from roar.routing.graph import load_road_graph
from roar.routing.guard import path_realized_cost
from roar.routing.robust_astar import RobustAStar

CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"
RESULTS_DIR = REPO_ROOT / "results"
QUERY_SET_CONFIG_PATH = CONFIGS_DIR / "query_set.yaml"
EXPERIMENTS_META_PATH = CONFIGS_DIR / "experiments.yaml"

# Predictor-independent methods: computed ONCE per query, never per sweep
# point (their result can't change with the predictor).
CLASSICAL_METHOD_FACTORIES = {
    "dijkstra": lambda graph: DijkstraBaseline(graph, static_free_flow_cost),
    "astar": lambda graph: AStarBaseline(graph, static_free_flow_cost),
    "bidirectional_dijkstra": lambda graph: BidirectionalDijkstraBaseline(
        graph, static_free_flow_cost
    ),
}


def build_eval_context() -> EvalContext:
    data_cfg = load_config()
    predictor_cfg = load_config(PREDICTOR_CONFIG_PATH)
    predictor_cfg.setdefault("time_bucket_minutes", data_cfg["time_bucket_minutes"])

    graph = load_road_graph()
    features_df = pd.read_parquet(FEATURES_PATH)
    oracle = OraclePredictor(
        features_df, time_bucket_minutes=predictor_cfg["time_bucket_minutes"]
    )
    return EvalContext(
        graph=graph, features_df=features_df, oracle=oracle, predictor_cfg=predictor_cfg
    )


def _resolve_sweep_points(exp_cfg: dict) -> list[dict]:
    """A sweep varies exactly one of {predictor, lambda_base, lambda_mode}
    across `sweep.values`; every other parameter stays at the experiment's
    base value. No `sweep` key at all means a single (base-parameters-only)
    "sweep point"."""
    base = {
        "predictor": exp_cfg.get("predictor"),
        "lambda_base": exp_cfg.get("lambda_base"),
        "lambda_mode": exp_cfg.get("lambda_mode", "fixed"),
        "alpha": exp_cfg["alpha"],
    }
    sweep = exp_cfg.get("sweep")
    if sweep is None:
        return [dict(base)]

    param = sweep["param"]
    points = []
    for value in sweep["values"]:
        point = dict(base)
        point[param] = value
        points.append(point)
    return points


def _realized_cost(ctx: EvalContext, query: Query, path: list[str] | None) -> float:
    """The ground-truth REALIZED cost of a (fixed) path -- never that
    method's own cost model. This is what "competitive ratio vs the
    oracle-optimal path" must be measured against: classical methods plan
    with free-flow costs, and pure_ml_astar plans with a possibly
    adversarial predictor's costs, but what matters operationally (and for
    a fair comparison against oracle_optimal_cost) is what the path
    actually costs in reality -- exactly the same principle
    roar/routing/guard.py's RobustnessGuard is built on, reused here via
    the same `path_realized_cost` helper."""
    return path_realized_cost(ctx.graph, path, ctx.ground_truth_cost_fn(), query.depart_time)


def _run_classical_methods(
    exp_cfg: dict,
    config_path: str,
    query_seed: int,
    queries: list[Query],
    oracle_optimal: dict[int, float],
    ctx: EvalContext,
    provenance: dict,
) -> list[dict]:
    method_names = [m for m in exp_cfg["methods"] if m in CLASSICAL_METHOD_FACTORIES]
    if not method_names:
        return []

    baselines = {name: CLASSICAL_METHOD_FACTORIES[name](ctx.graph) for name in method_names}
    rows = []
    for query in queries:
        oracle_optimal_cost = oracle_optimal[query.query_id]
        for name, baseline in baselines.items():
            result = baseline.search(query.origin, query.dest, query.depart_time)
            rows.append(
                make_row(
                    run_id=exp_cfg["run_id"],
                    experiment=exp_cfg["experiment"],
                    method=name,
                    query=query,
                    query_seed=query_seed,
                    predictor_spec={"type": "none"},
                    lambda_base=None,
                    lambda_mode=None,
                    alpha=None,
                    guard_enabled=None,
                    cost=_realized_cost(ctx, query, result.path),
                    oracle_optimal_cost=oracle_optimal_cost,
                    node_expansions=result.node_expansions,
                    latency_ms=result.latency_ms,
                    guard_invoked=None,
                    seed=exp_cfg["seed"],
                    provenance=provenance,
                    config_path=config_path,
                )
            )
    return rows


def _run_sweep_point(
    point: dict,
    exp_cfg: dict,
    config_path: str,
    query_seed: int,
    queries: list[Query],
    oracle_optimal: dict[int, float],
    ctx: EvalContext,
    provenance: dict,
) -> list[dict]:
    methods = exp_cfg["methods"]
    predictor_spec = point["predictor"]
    predictor = build_predictor(predictor_spec, ctx) if predictor_spec is not None else None
    lambda_base = point["lambda_base"]
    lambda_mode = point["lambda_mode"]
    alpha = point["alpha"]

    needs_robust = predictor is not None and (
        "robust_astar" in methods or "robust_astar_no_guard" in methods
    )
    needs_pure_ml = predictor is not None and "pure_ml_astar" in methods

    robust = (
        RobustAStar(
            ctx.graph,
            predictor,
            ctx.ground_truth_cost_fn(),
            alpha=alpha,
            lambda_base=lambda_base,
            confidence_modulated=(lambda_mode == "confidence_modulated"),
        )
        if needs_robust
        else None
    )
    pure_ml = PureMLAStarBaseline(ctx.graph, predictor) if needs_pure_ml else None

    rows: list[dict] = []
    for query in queries:
        oracle_optimal_cost = oracle_optimal[query.query_id]
        common = {
            "run_id": exp_cfg["run_id"],
            "experiment": exp_cfg["experiment"],
            "query": query,
            "query_seed": query_seed,
            "predictor_spec": predictor_spec,
            "oracle_optimal_cost": oracle_optimal_cost,
            "seed": exp_cfg["seed"],
            "provenance": provenance,
            "config_path": config_path,
        }

        if needs_pure_ml:
            result = pure_ml.search(query.origin, query.dest, query.depart_time)
            rows.append(
                make_row(
                    **common,
                    method="pure_ml_astar",
                    lambda_base=None,
                    lambda_mode=None,
                    alpha=None,
                    guard_enabled=None,
                    cost=_realized_cost(ctx, query, result.path),
                    node_expansions=result.node_expansions,
                    latency_ms=result.latency_ms,
                    guard_invoked=None,
                )
            )

        if needs_robust:
            # ONE search serves both "robust_astar" and "robust_astar_no_guard"
            # rows (the guard-on/off ablation) -- never re-run the search.
            result = robust.search(query.origin, query.dest, query.depart_time)
            if "robust_astar" in methods:
                rows.append(
                    make_row(
                        **common,
                        method="robust_astar",
                        lambda_base=lambda_base,
                        lambda_mode=lambda_mode,
                        alpha=alpha,
                        guard_enabled=True,
                        cost=result.cost,
                        node_expansions=result.node_expansions,
                        latency_ms=result.latency_ms,
                        guard_invoked=result.robustness_bound.guard_invoked,
                    )
                )
            if "robust_astar_no_guard" in methods:
                rows.append(
                    make_row(
                        **common,
                        method="robust_astar_no_guard",
                        lambda_base=lambda_base,
                        lambda_mode=lambda_mode,
                        alpha=alpha,
                        guard_enabled=False,
                        cost=result.robustness_bound.candidate_cost,
                        node_expansions=result.node_expansions,
                        latency_ms=result.latency_ms,
                        guard_invoked=None,
                    )
                )
    return rows


def run_experiment(
    exp_cfg: dict,
    config_path: str,
    query_seed: int,
    queries: list[Query],
    oracle_optimal: dict[int, float],
    ctx: EvalContext,
    provenance: dict,
) -> pd.DataFrame:
    """Runs one experiment config's classical methods (once) plus every
    sweep point's predictor-guided methods, over the SAME `queries` list
    (one query-generation seed's worth), and returns the combined result
    rows as a DataFrame."""
    rows = _run_classical_methods(
        exp_cfg, config_path, query_seed, queries, oracle_optimal, ctx, provenance
    )
    for point in _resolve_sweep_points(exp_cfg):
        rows.extend(
            _run_sweep_point(
                point, exp_cfg, config_path, query_seed, queries, oracle_optimal, ctx, provenance
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Runs every experiment once per query-generation seed in
    `query_set.yaml`'s `seeds` list (>=5, per CLAUDE.md Phase 6's "mean +/-
    95% bootstrap CI ... over >=5 seeds" requirement) and writes ONE
    combined parquet per experiment (all seeds' rows, distinguished by the
    `query_seed` column) -- so Phase 6's stats.py can bootstrap over
    queries WITHIN a seed and separately over the >=5 seed-level means,
    from the same file.

    The graph, features, and oracle predictor (and any cached LightGBM
    model/ablation retrain) are built ONCE for the whole run, not once per
    seed -- only query generation and search repeat per seed.
    """
    provenance = make_provenance()
    print("Building graph, features, and oracle predictor ...")
    ctx = build_eval_context()

    query_cfg = load_config(QUERY_SET_CONFIG_PATH)
    seeds = query_cfg["seeds"]
    if len(seeds) < 5:
        raise ValueError(
            f"query_set.yaml must list >= 5 seeds for seed-level CIs (Phase 6), got {len(seeds)}"
        )

    meta = load_config(EXPERIMENTS_META_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    exp_frames: dict[str, list[pd.DataFrame]] = {exp_file: [] for exp_file in meta["experiments"]}

    for query_seed in seeds:
        print(
            f"--- query_seed={query_seed}: generating {query_cfg['n_queries']} "
            "stratified queries ..."
        )
        queries = generate_stratified_queries(ctx.graph, ctx.features_df, query_cfg, query_seed)
        oracle_optimal = compute_oracle_optimal_costs(ctx.graph, ctx.oracle, queries)

        for exp_file in meta["experiments"]:
            exp_cfg_path = CONFIGS_DIR / exp_file
            exp_cfg = load_config(exp_cfg_path)
            config_path = str(exp_cfg_path.relative_to(REPO_ROOT))

            print(f"  Running experiment '{exp_cfg['experiment']}' ({config_path}) ...")
            df = run_experiment(
                exp_cfg, config_path, query_seed, queries, oracle_optimal, ctx, provenance
            )
            exp_frames[exp_file].append(df)

    for exp_file, frames in exp_frames.items():
        exp_cfg = load_config(CONFIGS_DIR / exp_file)
        combined = pd.concat(frames, ignore_index=True)
        out_path = RESULTS_DIR / f"{exp_cfg['run_id']}.parquet"
        combined.to_parquet(out_path)
        print(f"  -> {len(combined)} rows ({len(seeds)} seeds) -> {out_path}")

    print(f"Done at {dt.datetime.now(dt.UTC).isoformat()}.")


if __name__ == "__main__":
    main()
