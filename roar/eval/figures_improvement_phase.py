"""Improvement Phase figures/tables -- kept in a SEPARATE module from
roar/eval/figures.py (Phase 6) rather than edited into it, so Fig 1-3 and
Table 1-2's existing code paths are provably untouched (CLAUDE.md's
improvement-phase rule 2: no previously reported metric's definition may
change). `make figures` runs this in addition to figures.py's main().

Reads the NEW parquets this phase produced:
  - results/latency_vs_graph_size_lightgbm.parquet (Task 1d)
  - results/adversarial_budget_sweep.parquet (Task 2b)
  - results/adversary_lambda_diagnosis.parquet (Task 3a, adversary half)
  - results/lambda_sweep_degraded_sigma_{0.5,1.0,2.0}.parquet (Task 3a, noisy half)

All colors/styling reuse roar.eval.figures's validated palette and
constants, imported from there rather than redefined.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from roar.eval.figures import FIGURES_DIR, METHOD_COLOR, METHOD_LABEL, MUTED_INK, RESULTS_DIR
from roar.eval.stats import seed_level_summary
from roar.graph.features import FEATURES_PATH
from roar.predictor.oracle import OraclePredictor
from roar.routing.baselines import astar, predictor_cost_fn, static_free_flow_cost
from roar.routing.graph import load_road_graph
from roar.routing.guard import path_realized_cost


def _load(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / f"{run_id}.parquet")


def _compute_classical_reference(df: pd.DataFrame) -> pd.DataFrame:
    """For each unique (query_seed, query_id, origin, dest, depart_time) in
    `df`, computes the classical (free-flow) path's REALIZED cost against
    ground truth -- the SAME definition roar/routing/guard.py's
    RobustnessBound.classical_cost uses internally -- so ratio_vs_classical
    can be computed for methods that don't carry their own
    robustness_bound (pure_ml_astar, robust_astar_no_guard). Not logged
    directly in the improvement-phase parquets (a design gap discovered
    while building these figures, after the expensive experiment run was
    already in progress); computed here post-hoc from fields already
    logged -- a cheap, predictor-free search, not a re-run of anything
    expensive."""
    graph = load_road_graph()
    features_df = pd.read_parquet(FEATURES_PATH)
    oracle = OraclePredictor(features_df)
    ground_truth_cost_fn = predictor_cost_fn(oracle)

    unique_queries = df[
        ["query_seed", "query_id", "origin", "dest", "depart_time"]
    ].drop_duplicates()

    rows = []
    for _, row in unique_queries.iterrows():
        depart_time = row["depart_time"]
        if isinstance(depart_time, pd.Timestamp):
            depart_time = depart_time.to_pydatetime()
        result = astar(graph, row["origin"], row["dest"], depart_time, static_free_flow_cost)
        classical_cost = path_realized_cost(graph, result.path, ground_truth_cost_fn, depart_time)
        rows.append(
            {
                "query_seed": row["query_seed"],
                "query_id": row["query_id"],
                "classical_cost": classical_cost,
            }
        )
    return pd.DataFrame(rows)


def fig4_routing_aware_adversary(alpha: float = 0.3) -> None:
    """Task 2's money plot: worst-case ratio_vs_classical vs. adversarial
    budget for pure-ML A* vs. robust A* (guarded) vs. robust A* with the
    guard disabled, under the routing-aware adversary
    (roar/predictor/routing_aware_adversarial.py) -- the stronger
    instrument built specifically because the original uniform-budget
    AdversarialPredictor produced no significant separation (Phase 5/6's
    "ns" finding, results/figures/table1_headline_metrics.csv)."""
    df = _load("adversarial_budget_sweep")
    classical = _compute_classical_reference(df)
    df = df.merge(classical, on=["query_seed", "query_id"], how="left")
    df["ratio_vs_classical"] = df["cost"] / df["classical_cost"]
    df["budget"] = df["predictor_params"].apply(lambda s: json.loads(s)["budget"])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    styles = {
        "pure_ml_astar": "-.",
        "robust_astar": "-",
        "robust_astar_no_guard": ":",
    }
    csv_rows = []
    for method, linestyle in styles.items():
        sub = df[df["method"] == method]
        per_seed_worst = (
            sub.groupby(["budget", "query_seed"])["ratio_vs_classical"].max().reset_index()
        )
        summary = seed_level_summary(
            per_seed_worst, value_col="ratio_vs_classical", group_cols=["budget"]
        ).sort_values("budget")
        summary.insert(0, "method", method)
        csv_rows.append(summary)

        color = METHOD_COLOR[method]
        ax.plot(
            summary["budget"],
            summary["mean"],
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=5,
            linewidth=2,
            label=METHOD_LABEL[method],
        )
        ax.fill_between(
            summary["budget"], summary["ci_low"], summary["ci_high"], color=color, alpha=0.15
        )

    ax.axhline(1.0, color=METHOD_COLOR["astar"], linestyle="--", linewidth=1.5, label="Classical")
    ax.axhline(
        1 + alpha, color=MUTED_INK, linestyle=":", linewidth=1.5,
        label=f"Guard floor (1+α={1 + alpha:g})",
    )
    ax.set_xlabel("Adversarial budget B (routing-aware adversary)")
    ax.set_ylabel("Worst-case realized cost / classical cost (max per seed)")
    ax.set_title("Routing-aware adversary: guard holds, unguarded does not")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig4_routing_aware_adversary.png")
    plt.close(fig)
    pd.concat(csv_rows, ignore_index=True).to_csv(
        FIGURES_DIR / "fig4_routing_aware_adversary.csv", index=False
    )


def fig5_guard_clamp_rate_vs_lambda() -> None:
    """Task 3b: guard-invocation ("clamp") rate as a function of lambda,
    under three conditions -- the real LightGBM predictor (Phase 6,
    reused unmodified from ablation_lambda_sweep.parquet), NoisyPredictor
    at three degraded sigma levels (Task 3a), and the routing-aware
    adversary at a fixed high budget (Task 3a). A near-zero rate that
    stays near-zero even under degradation would mean the flat lambda
    curve (Phase 6) isn't explained by "the guard rarely needs to
    intervene" -- a rate that climbs under degradation would explain it."""
    conditions: list[tuple[str, pd.DataFrame, str]] = [
        ("real LightGBM (Phase 6)", _load("ablation_lambda_sweep"), "-"),
        ("adversary, B=1.0", _load("adversary_lambda_diagnosis"), "--"),
    ]
    for sigma in ("0.5", "1.0", "2.0"):
        conditions.append(
            (f"NoisyPredictor σ={sigma}", _load(f"lambda_sweep_degraded_sigma_{sigma}"), ":")
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    csv_rows = []
    for label, df, linestyle in conditions:
        sub = df[(df["method"] == "robust_astar") & df["guard_invoked"].notna()]
        rate = sub.groupby("lambda_base")["guard_invoked"].mean().reset_index()
        rate["condition"] = label
        csv_rows.append(rate)
        ax.plot(
            rate["lambda_base"], rate["guard_invoked"], linestyle=linestyle, marker="o",
            markersize=4, linewidth=1.5, label=label,
        )

    ax.set_xlabel("λ (trust in the ML-estimate heuristic term)")
    ax.set_ylabel('Guard invocation ("clamp") rate')
    ax.set_title("Guard-clamp rate vs. λ, across predictor conditions")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig5_guard_clamp_rate_vs_lambda.png")
    plt.close(fig)
    pd.concat(csv_rows, ignore_index=True).to_csv(
        FIGURES_DIR / "fig5_guard_clamp_rate_vs_lambda.csv", index=False
    )


def fig6_latency_lightgbm_vs_materialized() -> None:
    """Task 1d: the real deployability-relevant latency comparison --
    LightGBM (the predictor an actual deployment would use, unlike
    Fig 3's OraclePredictor) before vs. after the materialization
    optimization (roar/predictor/materialized_lightgbm.py)."""
    df = _load("latency_vs_graph_size_lightgbm")
    summary = df.groupby(["predictor_variant", "graph_size"])["latency_ms"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {"base": METHOD_COLOR["pure_ml_astar"], "materialized": METHOD_COLOR["robust_astar"]}
    labels = {"base": "LightGBM (unoptimized)", "materialized": "LightGBM (materialized)"}
    for variant, group in summary.groupby("predictor_variant"):
        group = group.sort_values("graph_size")
        ax.plot(
            group["graph_size"], group["latency_ms"], color=colors[variant], marker="o",
            markersize=5, linewidth=2, label=labels[variant],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Graph size (nodes)")
    ax.set_ylabel("Mean robust_astar query latency (ms)")
    ax.set_title("Materialization's effect on latency (real LightGBM predictor)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig6_latency_lightgbm_vs_materialized.png")
    plt.close(fig)
    summary.to_csv(FIGURES_DIR / "fig6_latency_lightgbm_vs_materialized.csv", index=False)


def table3_improvement_phase_summary() -> pd.DataFrame:
    """One row per (task, condition): headline seed-level mean +/- CI
    metrics for this phase's new experiments, mirroring
    table1_headline_metrics.csv's structure."""
    rows = []

    budget_df = _load("adversarial_budget_sweep")
    for method in ("pure_ml_astar", "robust_astar", "robust_astar_no_guard"):
        for budget in sorted(budget_df["predictor_params"].unique()):
            sub = budget_df[
                (budget_df["method"] == method) & (budget_df["predictor_params"] == budget)
            ]
            summary = seed_level_summary(sub, value_col="cost", group_cols=["method"])
            if summary.empty:
                continue
            row = summary.iloc[0]
            rows.append(
                {
                    "task": "2b_budget_sweep",
                    "condition": budget,
                    "method": method,
                    "n_seeds": row["n_seeds"],
                    "mean_realized_cost": row["mean"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                }
            )

    for label, run_id in [
        ("adversary_B1.0", "adversary_lambda_diagnosis"),
        ("noisy_sigma0.5", "lambda_sweep_degraded_sigma_0.5"),
        ("noisy_sigma1.0", "lambda_sweep_degraded_sigma_1.0"),
        ("noisy_sigma2.0", "lambda_sweep_degraded_sigma_2.0"),
    ]:
        df = _load(run_id)
        df = df[df["method"] == "robust_astar"]
        summary = seed_level_summary(
            df, value_col="competitive_ratio", group_cols=["lambda_base"]
        ).sort_values("lambda_base")
        for _, row in summary.iterrows():
            invocation = df[df["lambda_base"] == row["lambda_base"]]["guard_invoked"].mean()
            rows.append(
                {
                    "task": "3a_lambda_diagnosis",
                    "condition": f"{label}, lambda={row['lambda_base']:g}",
                    "method": "robust_astar",
                    "n_seeds": row["n_seeds"],
                    "mean_realized_cost": row["mean"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "guard_invocation_rate": invocation,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    print("Fig 4: routing-aware adversary money plot ...")
    fig4_routing_aware_adversary()
    print("Fig 5: guard-clamp rate vs lambda, across conditions ...")
    fig5_guard_clamp_rate_vs_lambda()
    print("Fig 6: latency, LightGBM base vs materialized ...")
    fig6_latency_lightgbm_vs_materialized()

    print("Table 3: improvement-phase summary ...")
    t3 = table3_improvement_phase_summary()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    t3.to_csv(FIGURES_DIR / "table3_improvement_phase_summary.csv", index=False)

    print(f"Done -- improvement-phase figures/tables written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
