"""Improvement Phase Task 3a (NoisyPredictor half): runs the three
exp_lambda_sweep_degraded_sigma_*.yaml configs -- the SAME lambda_base
sweep mechanism as Phase 6's ablation_lambda_sweep (roar/eval/harness.py,
completely unmodified), just with NoisyPredictor at a fixed degraded sigma
instead of the real LightGBM predictor.

Deliberately NOT added to experiments/configs/experiments.yaml: that file
is Phase 5's committed meta-list, and re-running `make experiments` reruns
every experiment in it (a ~30+ minute job) to regenerate results that are
already good. This script runs just the 3 new configs, reusing
roar.eval.harness's exact, unmodified functions (build_eval_context,
generate_stratified_queries, compute_oracle_optimal_costs, run_experiment)
so the mechanism is provably identical to Phase 6's -- CLAUDE.md's
improvement-phase rule 2 (no metric-definition changes).

Run directly: `python -m roar.eval.lambda_diagnosis_noisy`.
"""

from __future__ import annotations

import pandas as pd

from roar.eval.harness import CONFIGS_DIR, RESULTS_DIR, build_eval_context, run_experiment
from roar.eval.metrics import compute_oracle_optimal_costs, make_provenance
from roar.eval.queries import generate_stratified_queries
from roar.graph.config import REPO_ROOT, load_config

QUERY_SET_CONFIG_PATH = CONFIGS_DIR / "query_set.yaml"

CONFIG_FILES = [
    "exp_lambda_sweep_degraded_sigma_0.5.yaml",
    "exp_lambda_sweep_degraded_sigma_1.0.yaml",
    "exp_lambda_sweep_degraded_sigma_2.0.yaml",
]


def main() -> None:
    provenance = make_provenance()
    print("Building graph, features, and oracle predictor ...")
    ctx = build_eval_context()

    query_cfg = load_config(QUERY_SET_CONFIG_PATH)
    seeds = query_cfg["seeds"]

    frames: dict[str, list[pd.DataFrame]] = {f: [] for f in CONFIG_FILES}

    for query_seed in seeds:
        print(f"--- query_seed={query_seed}: generating queries ...")
        queries = generate_stratified_queries(ctx.graph, ctx.features_df, query_cfg, query_seed)
        oracle_optimal = compute_oracle_optimal_costs(ctx.graph, ctx.oracle, queries)

        for config_file in CONFIG_FILES:
            exp_cfg_path = CONFIGS_DIR / config_file
            exp_cfg = load_config(exp_cfg_path)
            config_path = str(exp_cfg_path.relative_to(REPO_ROOT))
            print(f"  Running '{exp_cfg['experiment']}' ({config_path}) ...")
            df = run_experiment(
                exp_cfg, config_path, query_seed, queries, oracle_optimal, ctx, provenance
            )
            frames[config_file].append(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for config_file, dfs in frames.items():
        exp_cfg = load_config(CONFIGS_DIR / config_file)
        combined = pd.concat(dfs, ignore_index=True)
        out_path = RESULTS_DIR / f"{exp_cfg['run_id']}.parquet"
        combined.to_parquet(out_path)
        print(f"  -> {len(combined)} rows ({len(seeds)} seeds) -> {out_path}")


if __name__ == "__main__":
    main()
