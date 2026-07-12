"""Improvement Phase Task 2: the routing-aware adversary sweep, and Task 3's
adversary-side lambda diagnosis.

Uses `RoutingAwareAdversarialPredictor`
(roar/predictor/routing_aware_adversarial.py) -- see its module docstring
for the threat model. Unlike every predictor roar/eval/harness.py builds,
this one is constructed PER QUERY (it needs the query's origin/destination
to compute its corridor/trap), so it does not fit harness.py's
one-predictor-per-sweep-point assumption. Rather than retrofit the tested
Phase 5 harness (risking altering a previously reported metric's
definition -- this phase's hard rule 2), this is a separate, additive
script that reuses the SAME shared query set (roar/eval/queries.py,
experiments/configs/query_set.yaml, all 5 seeds) and the SAME row schema
(roar.eval.metrics.make_row), writing to its own new results files.

## Task 2c -- the guard-floor safety check outranks everything else

Every robust_astar row's `robustness_bound.ratio` is checked against
`1 + alpha` THE MOMENT it is computed, not after the fact. If it is ever
exceeded, this script raises immediately and does not keep collecting
data: a guard violation is a correctness bug in
roar/routing/guard.py, and per CLAUDE.md's improvement-phase rules it
outranks every other finding in this phase.

Run directly: `python -m roar.eval.adversarial_experiment`.
"""

from __future__ import annotations

import pandas as pd

from roar.eval.harness import build_eval_context
from roar.eval.metrics import compute_oracle_optimal_costs, make_provenance, make_row
from roar.eval.queries import Query, generate_stratified_queries
from roar.graph.config import REPO_ROOT, load_config
from roar.predictor.routing_aware_adversarial import RoutingAwareAdversarialPredictor
from roar.routing.baselines import PureMLAStarBaseline
from roar.routing.guard import path_realized_cost
from roar.routing.robust_astar import RobustAStar

CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"
RESULTS_DIR = REPO_ROOT / "results"
QUERY_SET_CONFIG_PATH = CONFIGS_DIR / "query_set.yaml"

ALPHA = 0.3
GUARD_FLOOR_EPSILON = 1e-6  # float tolerance, not a relaxation of the bound

# Task 2b: budgets to sweep, guarded vs unguarded vs pure-ML.
BUDGET_SWEEP = [0.25, 0.5, 1.0, 2.0]

# Task 3a: NoisyPredictor sigma levels for the degraded-predictor lambda
# diagnosis (handled by a separate function in this module using the
# EXISTING harness sweep mechanism, since NoisyPredictor -- unlike this
# adversary -- IS stateless per-edge and fits it).
LAMBDA_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Task 3a: fixed high budget for the adversary-side lambda diagnosis.
ADVERSARY_LAMBDA_DIAGNOSIS_BUDGET = 1.0


class GuardFloorViolation(RuntimeError):
    """Raised immediately if any robust_astar query ever exceeds
    (1 + alpha) -- per CLAUDE.md's improvement-phase rules, this outranks
    everything else in this phase and must stop the run, not be averaged
    into a summary statistic."""


def _check_guard_floor(row: dict, alpha: float) -> None:
    ratio = row["robustness_bound_ratio"]
    if ratio == ratio and ratio > 1 + alpha + GUARD_FLOOR_EPSILON:  # ratio == ratio: not NaN
        raise GuardFloorViolation(
            f"GUARD FLOOR VIOLATED: ratio={ratio} > 1+alpha={1 + alpha} for query_id="
            f"{row['query_id']}, query_seed={row['query_seed']}, "
            f"predictor_params={row['predictor_params']} -- this is a correctness bug in "
            "roar/routing/guard.py, not a property of the adversary. STOPPING per Task 2c."
        )


def _make_adversary_row(
    *,
    run_id: str,
    experiment: str,
    method: str,
    query: Query,
    predictor_spec: dict,
    lambda_base: float,
    lambda_mode: str,
    guard_enabled: bool | None,
    cost: float,
    oracle_optimal_cost: float,
    node_expansions: int,
    latency_ms: float,
    guard_invoked: bool | None,
    robustness_bound_ratio: float,
    seed: int,
    provenance: dict,
    config_path: str,
) -> dict:
    row = make_row(
        run_id=run_id,
        experiment=experiment,
        method=method,
        query=query,
        query_seed=seed,
        predictor_spec=predictor_spec,
        lambda_base=lambda_base,
        lambda_mode=lambda_mode,
        alpha=ALPHA,
        guard_enabled=guard_enabled,
        cost=cost,
        oracle_optimal_cost=oracle_optimal_cost,
        node_expansions=node_expansions,
        latency_ms=latency_ms,
        guard_invoked=guard_invoked,
        seed=seed,
        provenance=provenance,
        config_path=config_path,
    )
    row["robustness_bound_ratio"] = robustness_bound_ratio
    return row


def run_budget_sweep(ctx, queries: list[Query], oracle_optimal: dict, provenance: dict, seed: int):
    """Task 2b: guarded vs unguarded vs pure-ML, over BUDGET_SWEEP, on the
    same queries. Reports ALL budgets, including any showing no
    separation -- that is a finding, not a failure."""
    rows: list[dict] = []
    ground_truth_cost_fn = ctx.ground_truth_cost_fn()

    for query in queries:
        oracle_optimal_cost = oracle_optimal[query.query_id]
        adversary = RoutingAwareAdversarialPredictor(
            ctx.graph, ctx.oracle, BUDGET_SWEEP[0], query.origin, query.dest, query.depart_time
        )
        for budget in BUDGET_SWEEP:
            budgeted_adversary = adversary.with_budget(budget)
            predictor_spec = {"type": "routing_aware_adversarial", "budget": budget}

            pure_ml = PureMLAStarBaseline(ctx.graph, budgeted_adversary)
            pure_ml_result = pure_ml.search(query.origin, query.dest, query.depart_time)
            pure_ml_realized = path_realized_cost(
                ctx.graph, pure_ml_result.path, ground_truth_cost_fn, query.depart_time
            )
            rows.append(
                _make_adversary_row(
                    run_id="adversarial_budget_sweep",
                    experiment="adversarial_budget_sweep",
                    method="pure_ml_astar",
                    query=query,
                    predictor_spec=predictor_spec,
                    lambda_base=None,
                    lambda_mode=None,
                    guard_enabled=None,
                    cost=pure_ml_realized,
                    oracle_optimal_cost=oracle_optimal_cost,
                    node_expansions=pure_ml_result.node_expansions,
                    latency_ms=pure_ml_result.latency_ms,
                    guard_invoked=None,
                    robustness_bound_ratio=float("nan"),
                    seed=seed,
                    provenance=provenance,
                    config_path=__file__,
                )
            )

            robust = RobustAStar(
                ctx.graph, budgeted_adversary, ground_truth_cost_fn, alpha=ALPHA, lambda_base=1.0
            )
            robust_result = robust.search(query.origin, query.dest, query.depart_time)
            guarded_row = _make_adversary_row(
                run_id="adversarial_budget_sweep",
                experiment="adversarial_budget_sweep",
                method="robust_astar",
                query=query,
                predictor_spec=predictor_spec,
                lambda_base=1.0,
                lambda_mode="fixed",
                guard_enabled=True,
                cost=robust_result.cost,
                oracle_optimal_cost=oracle_optimal_cost,
                node_expansions=robust_result.node_expansions,
                latency_ms=robust_result.latency_ms,
                guard_invoked=robust_result.robustness_bound.guard_invoked,
                robustness_bound_ratio=robust_result.robustness_bound.ratio,
                seed=seed,
                provenance=provenance,
                config_path=__file__,
            )
            _check_guard_floor(guarded_row, ALPHA)  # Task 2c -- stop immediately if violated
            rows.append(guarded_row)

            rows.append(
                _make_adversary_row(
                    run_id="adversarial_budget_sweep",
                    experiment="adversarial_budget_sweep",
                    method="robust_astar_no_guard",
                    query=query,
                    predictor_spec=predictor_spec,
                    lambda_base=1.0,
                    lambda_mode="fixed",
                    guard_enabled=False,
                    cost=robust_result.robustness_bound.candidate_cost,
                    oracle_optimal_cost=oracle_optimal_cost,
                    node_expansions=robust_result.node_expansions,
                    latency_ms=robust_result.latency_ms,
                    guard_invoked=None,
                    robustness_bound_ratio=float("nan"),
                    seed=seed,
                    provenance=provenance,
                    config_path=__file__,
                )
            )
    return rows


def run_adversary_lambda_diagnosis(
    ctx, queries: list[Query], oracle_optimal: dict, provenance: dict, seed: int
):
    """Task 3a (adversary half): lambda_base swept 0->1 at a FIXED high
    budget, measuring both competitive_ratio AND guard_invoked rate as a
    function of lambda -- Task 3b's "guard-clamp rate vs lambda" measure."""
    rows: list[dict] = []
    ground_truth_cost_fn = ctx.ground_truth_cost_fn()

    for query in queries:
        oracle_optimal_cost = oracle_optimal[query.query_id]
        adversary = RoutingAwareAdversarialPredictor(
            ctx.graph,
            ctx.oracle,
            ADVERSARY_LAMBDA_DIAGNOSIS_BUDGET,
            query.origin,
            query.dest,
            query.depart_time,
        )
        for lambda_base in LAMBDA_SWEEP:
            robust = RobustAStar(
                ctx.graph, adversary, ground_truth_cost_fn, alpha=ALPHA, lambda_base=lambda_base
            )
            result = robust.search(query.origin, query.dest, query.depart_time)
            row = _make_adversary_row(
                run_id="adversary_lambda_diagnosis",
                experiment="adversary_lambda_diagnosis",
                method="robust_astar",
                query=query,
                predictor_spec={
                    "type": "routing_aware_adversarial",
                    "budget": ADVERSARY_LAMBDA_DIAGNOSIS_BUDGET,
                },
                lambda_base=lambda_base,
                lambda_mode="fixed",
                guard_enabled=True,
                cost=result.cost,
                oracle_optimal_cost=oracle_optimal_cost,
                node_expansions=result.node_expansions,
                latency_ms=result.latency_ms,
                guard_invoked=result.robustness_bound.guard_invoked,
                robustness_bound_ratio=result.robustness_bound.ratio,
                seed=seed,
                provenance=provenance,
                config_path=__file__,
            )
            _check_guard_floor(row, ALPHA)
            rows.append(row)
    return rows


def main() -> None:
    provenance = make_provenance()
    print("Building graph, features, and oracle predictor ...")
    ctx = build_eval_context()

    query_cfg = load_config(QUERY_SET_CONFIG_PATH)
    seeds = query_cfg["seeds"]

    budget_sweep_rows: list[dict] = []
    lambda_diagnosis_rows: list[dict] = []

    for query_seed in seeds:
        print(f"--- query_seed={query_seed} ---")
        queries = generate_stratified_queries(ctx.graph, ctx.features_df, query_cfg, query_seed)
        oracle_optimal = compute_oracle_optimal_costs(ctx.graph, ctx.oracle, queries)

        print(f"  Task 2b: routing-aware adversary budget sweep {BUDGET_SWEEP} ...")
        budget_sweep_rows.extend(
            run_budget_sweep(ctx, queries, oracle_optimal, provenance, query_seed)
        )

        print("  Task 3a (adversary half): lambda sweep under the adversary ...")
        lambda_diagnosis_rows.extend(
            run_adversary_lambda_diagnosis(ctx, queries, oracle_optimal, provenance, query_seed)
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    budget_df = pd.DataFrame(budget_sweep_rows)
    budget_out = RESULTS_DIR / "adversarial_budget_sweep.parquet"
    budget_df.to_parquet(budget_out)
    print(f"Saved {len(budget_df)} rows to {budget_out}")

    lambda_df = pd.DataFrame(lambda_diagnosis_rows)
    lambda_out = RESULTS_DIR / "adversary_lambda_diagnosis.parquet"
    lambda_df.to_parquet(lambda_out)
    print(f"Saved {len(lambda_df)} rows to {lambda_out}")

    guarded = budget_df["method"] == "robust_astar"
    max_ratio = budget_df.loc[guarded, "robustness_bound_ratio"].max()
    print(
        f"Guard-floor check: max realized ratio_vs_classical across all queries = "
        f"{max_ratio:.4f} (bound = {1 + ALPHA}). No violations -- see GuardFloorViolation "
        "checks inline above."
    )


if __name__ == "__main__":
    main()
