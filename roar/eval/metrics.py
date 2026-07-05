"""Per-query metrics, predictor construction, and run provenance for the
Phase 5 experiment harness (roar/eval/harness.py).

CLAUDE.md rule 5 ("routing code must never import a concrete predictor
class") applies to `roar/routing/`, not here: something has to construct a
concrete predictor from a config entry. The routing code itself
(baselines.py, robust_astar.py, guard.py) still only ever receives a
`TravelTimePredictor` via dependency injection from this module, never
imports a concrete class.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from roar.eval.queries import Query
from roar.graph.config import REPO_ROOT
from roar.predictor.adversarial import AdversarialPredictor
from roar.predictor.base import TravelTimePredictor
from roar.predictor.lightgbm import LightGBMPredictor
from roar.predictor.noisy import NoisyPredictor
from roar.predictor.oracle import OraclePredictor
from roar.routing.baselines import astar, predictor_cost_fn
from roar.routing.graph import RoutingGraph

MODELS_DIR = REPO_ROOT / "results" / "models"

# Package versions recorded with every run (CLAUDE.md rule 4: reproducibility).
_PACKAGES_TO_RECORD = [
    "numpy",
    "pandas",
    "scipy",
    "networkx",
    "osmnx",
    "lightgbm",
    "scikit-learn",
]


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def library_versions() -> dict[str, str]:
    versions = {}
    for pkg in _PACKAGES_TO_RECORD:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator != denominator or numerator != numerator:  # either is NaN
        return float("nan")
    if denominator == 0:
        return 1.0 if numerator == 0 else float("inf")
    return numerator / denominator


@dataclasses.dataclass
class EvalContext:
    """Shared, expensive-to-build objects every experiment in a harness run
    reuses -- built ONCE by the harness, never re-built per sweep point.
    `_predictor_cache` memoizes lightgbm loads/retrains, which are the only
    expensive predictor constructions (oracle/noisy/adversarial are cheap
    wrappers)."""

    graph: RoutingGraph
    features_df: pd.DataFrame
    oracle: OraclePredictor
    predictor_cfg: dict
    _predictor_cache: dict[str, TravelTimePredictor] = dataclasses.field(default_factory=dict)

    def ground_truth_cost_fn(self):
        return predictor_cost_fn(self.oracle)


def resolve_model_dir(model_version: str = "latest") -> Path:
    """`model_version="latest"` reads results/models/latest.json (written by
    `make train`) to find the most recently trained model; any other value
    is treated as a literal version directory name under results/models/.
    Reused by roar/api/app.py at startup, not just this eval harness."""
    if model_version == "latest":
        latest = json.loads((MODELS_DIR / "latest.json").read_text())
        model_version = latest["version"]
    return MODELS_DIR / model_version


def _ablated_predictor_cfg(base_cfg: dict, excluded_features: list[str]) -> dict:
    """Drops `excluded_features` from every feature-column group -- the
    "feature ablation hook": point this at a real training run to see
    which engineered features the predictor actually depends on."""
    cfg = copy.deepcopy(base_cfg)
    fc = cfg["feature_columns"]
    for group in ("numeric", "categorical", "boolean"):
        fc[group] = [c for c in fc[group] if c not in excluded_features]
    return cfg


def build_predictor(spec: dict, ctx: EvalContext) -> TravelTimePredictor:
    """spec['type'] in {"oracle", "lightgbm", "lightgbm_ablation", "noisy",
    "adversarial"}. Memoized on ctx for the lifetime of one harness run."""
    kind = spec["type"]

    if kind == "oracle":
        return ctx.oracle

    if kind == "noisy":
        return NoisyPredictor(
            ctx.oracle, sigma_level=spec["sigma_level"], seed=spec.get("seed", 42)
        )

    if kind == "adversarial":
        return AdversarialPredictor(ctx.oracle, budget=spec["budget"])

    if kind == "lightgbm":
        model_version = spec.get("model_version", "latest")
        cache_key = f"lightgbm:{model_version}"
        if cache_key not in ctx._predictor_cache:
            model_dir = resolve_model_dir(model_version)
            ctx._predictor_cache[cache_key] = LightGBMPredictor.load(model_dir, ctx.features_df)
        return ctx._predictor_cache[cache_key]

    if kind == "lightgbm_ablation":
        excluded = sorted(spec.get("excluded_features", []))
        cache_key = f"lightgbm_ablation:{excluded}"
        if cache_key not in ctx._predictor_cache:
            ablated_cfg = _ablated_predictor_cfg(ctx.predictor_cfg, excluded)
            ctx._predictor_cache[cache_key] = LightGBMPredictor.train(
                ctx.features_df, cfg=ablated_cfg
            )
        return ctx._predictor_cache[cache_key]

    raise ValueError(f"unknown predictor spec type: {kind!r}")


def compute_oracle_optimal_costs(
    graph: RoutingGraph, oracle: OraclePredictor, queries: list[Query]
) -> dict[int, float]:
    """The TRUE ground-truth-optimal cost per query -- the denominator for
    every method's competitive ratio. Computed ONCE per query set and
    reused across every experiment/sweep point (it never depends on which
    method or predictor is being evaluated)."""
    ground_truth_cost_fn = predictor_cost_fn(oracle)
    return {
        q.query_id: astar(graph, q.origin, q.dest, q.depart_time, ground_truth_cost_fn).cost
        for q in queries
    }


def make_row(
    *,
    run_id: str,
    experiment: str,
    method: str,
    query: Query,
    query_seed: int,
    predictor_spec: dict,
    lambda_base: float | None,
    lambda_mode: str | None,
    alpha: float | None,
    guard_enabled: bool | None,
    cost: float,
    oracle_optimal_cost: float,
    node_expansions: int,
    latency_ms: float,
    guard_invoked: bool | None,
    seed: int,
    provenance: dict,
    config_path: str,
) -> dict:
    """One row of the logged result schema (CLAUDE.md rule 4: every number
    in the paper must be regenerable from this table). `query_seed` is the
    query-generation seed this query came from (roar/eval/queries.py) --
    `query_id` alone is only unique WITHIN one seed's query list, so
    `(query_seed, query_id)` together is the real primary key across a
    multi-seed run; `seed` is the experiment config's own seed (e.g. a
    NoisyPredictor's noise seed), a separate thing."""
    return {
        "run_id": run_id,
        "experiment": experiment,
        "method": method,
        "query_seed": query_seed,
        "query_id": query.query_id,
        "origin": query.origin,
        "dest": query.dest,
        "depart_time": query.depart_time,
        "distance_stratum": query.distance_stratum,
        "time_stratum": query.time_stratum,
        "predictor_type": predictor_spec.get("type"),
        "predictor_params": json.dumps(
            {k: v for k, v in predictor_spec.items() if k != "type"}, sort_keys=True
        ),
        "lambda_base": lambda_base,
        "lambda_mode": lambda_mode,
        "alpha": alpha,
        "guard_enabled": guard_enabled,
        "cost": cost,
        "oracle_optimal_cost": oracle_optimal_cost,
        "competitive_ratio": _safe_ratio(cost, oracle_optimal_cost),
        "node_expansions": node_expansions,
        "latency_ms": latency_ms,
        "guard_invoked": guard_invoked,
        "seed": seed,
        "git_commit": provenance["git_commit"],
        "library_versions": json.dumps(provenance["library_versions"], sort_keys=True),
        "config_path": config_path,
        "run_timestamp": provenance["run_timestamp"],
    }


def make_provenance() -> dict:
    return {
        "git_commit": git_commit_hash(),
        "library_versions": library_versions(),
        "run_timestamp": dt.datetime.now(dt.UTC).isoformat(),
    }
