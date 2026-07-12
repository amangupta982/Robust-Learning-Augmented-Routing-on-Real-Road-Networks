"""Phase 6: turns roar/eval/harness.py's logged parquets into the paper's
figures and tables. `make figures` runs this and regenerates every output
from scratch -- no manual steps, no hand-edited numbers (CLAUDE.md rule 4).

Color: a fixed categorical color (and a distinct line style, so the
figures survive grayscale printing) is assigned per METHOD across every
figure -- the same entity always gets the same color, never reassigned
based on which methods happen to appear in a given plot. Colors are the
validated palette from the dataviz skill (references/palette.md),
CVD-checked via `node scripts/validate_palette.js` for this exact 6-color
set (see this module's git history / PR description for the validator
output).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from roar.eval.stats import compare_robust_vs_baselines, seed_level_summary
from roar.graph.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

METHOD_COLOR = {
    "dijkstra": "#eda100",
    "astar": "#2a78d6",
    "bidirectional_dijkstra": "#4a3aa7",
    "pure_ml_astar": "#e34948",
    "robust_astar": "#008300",
    "robust_astar_no_guard": "#eb6834",
}
METHOD_LABEL = {
    "dijkstra": "Dijkstra",
    "astar": "Classical A*",
    "bidirectional_dijkstra": "Bidirectional Dijkstra",
    "pure_ml_astar": "Pure-ML A* (unsafe)",
    "robust_astar": "Robust A* (guarded)",
    "robust_astar_no_guard": "Robust A* (guard off)",
}

GRIDLINE = "#e1e0d9"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.edgecolor": MUTED_INK,
        "axes.labelcolor": PRIMARY_INK,
        "text.color": PRIMARY_INK,
        "xtick.color": MUTED_INK,
        "ytick.color": MUTED_INK,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_results(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / f"{run_id}.parquet")


def add_ratio_vs_classical(df: pd.DataFrame, classical_method: str = "astar") -> pd.DataFrame:
    """Adds `ratio_vs_classical` = cost / classical's realized cost on the
    SAME (query_seed, query_id) query -- the metric RobustnessGuard's
    (1 + alpha) guarantee is actually stated in terms of (see
    roar/routing/guard.py), as opposed to `competitive_ratio`
    (cost / oracle_optimal_cost), a different, harder benchmark against
    the unattainable hindsight-optimal path."""
    classical = df[df["method"] == classical_method][
        ["query_seed", "query_id", "cost"]
    ].rename(columns={"cost": "classical_cost"})
    merged = df.merge(classical, on=["query_seed", "query_id"], how="left")
    merged["ratio_vs_classical"] = merged["cost"] / merged["classical_cost"]
    return merged


def _significance_stars(p_adjusted: float) -> str:
    if p_adjusted != p_adjusted:  # NaN
        return ""
    if p_adjusted < 0.001:
        return "***"
    if p_adjusted < 0.01:
        return "**"
    if p_adjusted < 0.05:
        return "*"
    return "ns"


def fig1_money_plot(alpha: float = 0.3) -> None:
    """The money plot: WORST-CASE realized-cost ratio vs. prediction error
    (NoisyPredictor's sigma_level, smoothness_sweep.parquet) for robust A*
    vs pure-ML A* vs classical, on the guard-relevant ratio_vs_classical
    scale -- showing robust A* pinned at/under the (1 + alpha) guard floor
    while pure-ML A* is not.

    Worst-case (max per seed), not mean, is the right statistic here: the
    guard's guarantee is itself a worst-case bound, and it is real outlier
    queries -- not the typical/average query -- where an untrusted
    predictor does the most damage. Plotting the mean would understate the
    story: most queries in this project's query set are short single-edge
    ones with limited alternate-route opportunity (see
    experiments/configs/query_set.yaml), so the MEAN ratio stays close to
    1 regardless of predictor quality; it is specifically the queries where
    a bad prediction can lure the router onto a genuinely worse path where
    the divergence between pure-ML A* and robust A* shows up.
    """
    df = load_results("smoothness_sweep")
    df = add_ratio_vs_classical(df)
    df["sigma_level"] = df["predictor_params"].apply(
        lambda s: json.loads(s).get("sigma_level")
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    csv_rows = []

    for method, linestyle in [("pure_ml_astar", "-."), ("robust_astar", "-")]:
        sub = df[df["method"] == method].dropna(subset=["sigma_level"])
        per_seed_worst = (
            sub.groupby(["sigma_level", "query_seed"])["ratio_vs_classical"].max().reset_index()
        )
        summary = seed_level_summary(
            per_seed_worst, value_col="ratio_vs_classical", group_cols=["sigma_level"]
        ).sort_values("sigma_level")
        summary.insert(0, "method", method)
        csv_rows.append(summary)

        color = METHOD_COLOR[method]
        ax.plot(
            summary["sigma_level"],
            summary["mean"],
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=5,
            linewidth=2,
            label=METHOD_LABEL[method],
        )
        ax.fill_between(
            summary["sigma_level"], summary["ci_low"], summary["ci_high"], color=color, alpha=0.15
        )

    ax.axhline(1.0, color=METHOD_COLOR["astar"], linestyle="--", linewidth=1.5, label="Classical")
    ax.axhline(
        1 + alpha,
        color=MUTED_INK,
        linestyle=":",
        linewidth=1.5,
        label=f"Guard floor (1+α={1 + alpha:g})",
    )

    ax.set_xlabel("Prediction error (NoisyPredictor σ level)")
    ax.set_ylabel("Worst-case realized cost / classical cost (max per seed)")
    ax.set_title("Robust A* stays at the guard floor; pure-ML A* does not")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig1_money_plot.png")
    plt.close(fig)
    pd.concat(csv_rows, ignore_index=True).to_csv(FIGURES_DIR / "fig1_money_plot.csv", index=False)


def fig2_pareto_front() -> None:
    """Effect of lambda_base on search efficiency vs. path quality
    (ablation_lambda_sweep.parquet, real trained LightGBM predictor).

    HONEST FINDING, not the naively-expected consistency-vs-robustness
    Pareto front: with this real (non-adversarial, reasonably-behaved)
    predictor, mean AND worst-case competitive_ratio are statistically flat
    across the entire lambda_base in [0, 1] sweep -- there is no
    accuracy/robustness trade-off to plot. This is mathematically expected
    given this project's RobustAStar design: `ml_estimate(n)` (the lambda=1
    term of the blended heuristic) is computed EXACTLY via a real backward
    Dijkstra under the SAME cost function used for the actual forward
    search (see roar/routing/robust_astar.py) -- so it is never a harmful,
    inadmissible-in-practice heuristic term for a well-behaved predictor;
    blending it with the classical lower bound at any lambda still guides
    A* to the same optimal-under-that-cost-model path. What DOES vary
    smoothly and substantially with lambda is search efficiency: a more
    informative heuristic (higher lambda) prunes the search dramatically
    (see node_expansions below) without spending any path quality to do
    it. That is the real, honest lambda trade-off this experiment reveals
    for a real predictor -- the accuracy/robustness tension shows up
    instead against a genuinely adversarial or heavily noisy predictor
    (Fig 1), not against lambda for a trustworthy one.
    """
    df = load_results("ablation_lambda_sweep")
    df = df[df["method"] == "robust_astar"]

    per_seed_expansions = (
        df.groupby(["lambda_base", "query_seed"])["node_expansions"].mean().reset_index()
    )
    expansions_summary = seed_level_summary(
        per_seed_expansions, value_col="node_expansions", group_cols=["lambda_base"]
    ).sort_values("lambda_base")

    per_seed_ratio = (
        df.groupby(["lambda_base", "query_seed"])["competitive_ratio"].mean().reset_index()
    )
    ratio_summary = seed_level_summary(
        per_seed_ratio, value_col="competitive_ratio", group_cols=["lambda_base"]
    )
    overall_ratio = ratio_summary["mean"].mean()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    color = METHOD_COLOR["robust_astar"]
    ax.plot(
        expansions_summary["lambda_base"],
        expansions_summary["mean"],
        color=color,
        linestyle="-",
        marker="o",
        markersize=7,
        linewidth=2,
    )
    ax.fill_between(
        expansions_summary["lambda_base"],
        expansions_summary["ci_low"],
        expansions_summary["ci_high"],
        color=color,
        alpha=0.15,
    )

    ax.set_xlabel("λ (trust in the ML-estimate heuristic term)")
    ax.set_ylabel("Mean node expansions per query")
    ax.set_title("Higher λ speeds up search without spending path quality")
    ax.text(
        0.02,
        0.03,
        f"Mean competitive ratio is flat at {overall_ratio:.3f} across all λ\n"
        "(no accuracy/robustness trade-off with this real predictor -- see docstring)",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED_INK,
        va="bottom",
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig2_pareto_front.png")
    plt.close(fig)

    csv_df = expansions_summary.rename(
        columns={"mean": "mean_node_expansions", "ci_low": "expansions_ci_low",
                 "ci_high": "expansions_ci_high"}
    ).merge(
        ratio_summary.rename(
            columns={"mean": "mean_competitive_ratio", "ci_low": "ratio_ci_low",
                     "ci_high": "ratio_ci_high"}
        ),
        on="lambda_base",
        suffixes=("", "_ratio"),
    )
    csv_df.to_csv(FIGURES_DIR / "fig2_pareto_front.csv", index=False)


def fig3_latency_vs_graph_size() -> None:
    """Latency vs. graph size (deployability): real BFS-subgraph scaling
    experiment (roar/eval/scaling.py), log-log."""
    df = pd.read_parquet(RESULTS_DIR / "latency_vs_graph_size.parquet")
    summary = df.groupby(["method", "graph_size"])["latency_ms"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    linestyles = {
        "dijkstra": ":",
        "astar": "--",
        "bidirectional_dijkstra": "-.",
        "robust_astar": "-",
    }
    for method, group in summary.groupby("method"):
        group = group.sort_values("graph_size")
        ax.plot(
            group["graph_size"],
            group["latency_ms"],
            color=METHOD_COLOR[method],
            linestyle=linestyles.get(method, "-"),
            marker="o",
            markersize=5,
            linewidth=2,
            label=METHOD_LABEL[method],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Graph size (nodes)")
    ax.set_ylabel("Mean query latency (ms)")
    ax.set_title("Latency vs. graph size")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig3_latency_vs_graph_size.png")
    plt.close(fig)
    summary.to_csv(FIGURES_DIR / "fig3_latency_vs_graph_size.csv", index=False)


def table1_headline_metrics(
    experiments: tuple[str, ...] = ("consistency_sweep", "robustness_sweep", "smoothness_sweep"),
    baselines: tuple[str, ...] = ("astar", "pure_ml_astar"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Headline metrics per experiment per method: seed-level mean +/- 95%
    bootstrap CI of competitive_ratio, plus Holm-Bonferroni-corrected
    significance markers for robust A* vs each baseline (pooled across
    every sweep point within the experiment, for a compact top-line
    table -- per-sweep-point detail belongs to the ablation tables /
    Fig 1-2, not this headline summary)."""
    rows = []
    for experiment in experiments:
        df = load_results(experiment)
        methods_present = [m for m in df["method"].unique()]

        comparison = None
        if "robust_astar" in methods_present:
            present_baselines = [b for b in baselines if b in methods_present]
            if present_baselines:
                comparison = compare_robust_vs_baselines(
                    df, baseline_methods=list(present_baselines), alpha=alpha
                ).set_index("baseline")

        summary = seed_level_summary(df, value_col="competitive_ratio", group_cols=["method"])
        for _, row in summary.iterrows():
            method = row["method"]
            sig = ""
            if comparison is not None and method in comparison.index:
                sig = _significance_stars(comparison.loc[method, "p_adjusted"])
            elif method == "robust_astar":
                sig = ""
            rows.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "n_seeds": row["n_seeds"],
                    "mean_competitive_ratio": row["mean"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "significance_vs_robust_astar": sig,
                }
            )
    return pd.DataFrame(rows)


def table2_ablations(
    ablation_run_ids: tuple[str, ...] = (
        "ablation_guard",
        "ablation_lambda_mode",
        "ablation_lambda_sweep",
        "ablation_feature",
    ),
) -> pd.DataFrame:
    """Per-ablation, per-method(/sweep-point) summary: seed-level mean +/-
    95% CI of competitive_ratio, and guard invocation rate where
    applicable."""
    rows = []
    for run_id in ablation_run_ids:
        df = load_results(run_id)
        group_cols = ["method", "predictor_params", "lambda_mode", "lambda_base"]
        group_cols = [c for c in group_cols if df[c].notna().any() or c == "method"]
        summary = seed_level_summary(df, value_col="competitive_ratio", group_cols=group_cols)

        invocation_rate = (
            df[df["guard_invoked"].notna()]
            .groupby(group_cols)["guard_invoked"]
            .mean()
            .rename("guard_invocation_rate")
        )
        summary = summary.merge(invocation_rate, on=group_cols, how="left")
        summary.insert(0, "ablation", run_id)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    print("Fig 1: money plot (smoothness_sweep) ...")
    fig1_money_plot()
    print("Fig 2: Pareto front (ablation_lambda_sweep) ...")
    fig2_pareto_front()
    print("Fig 3: latency vs graph size ...")
    fig3_latency_vs_graph_size()

    print("Table 1: headline metrics ...")
    t1 = table1_headline_metrics()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    t1.to_csv(FIGURES_DIR / "table1_headline_metrics.csv", index=False)

    print("Table 2: ablation results ...")
    t2 = table2_ablations()
    t2.to_csv(FIGURES_DIR / "table2_ablations.csv", index=False)

    print(f"Done -- figures and tables written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
