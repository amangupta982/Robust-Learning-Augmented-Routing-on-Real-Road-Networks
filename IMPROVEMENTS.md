# Improvement Phase

Driven by weaknesses found in Phase 5/6's results. Per this phase's hard
rules: negative results below are reported as FINDINGS, not hidden or
tuned away; every run executed is logged (new parquets, never overwriting
old ones); every evaluation-code change is justified inline and does not
alter any previously reported metric's definition.

## Task 1 — Latency engineering

### 1a. Profiling (the premise didn't survive contact with the data)

cProfile on `RobustAStar.search()` for a single real query on the full
32,696-node graph, using the real trained LightGBM predictor:

```
3.648s total
  3.312s (91%)  _ml_backward_distances (the bounded-radius backward Dijkstra)
    3.500s cumulative, 31,789 calls   predictor_cost_fn's cost() wrapper
      3.077s cumulative, 31,609 calls   LightGBMPredictor.eta() / _predicted_times()
        ~2.157s   pandas MultiIndex .loc[] lookup chain (__getitem__ ->
                  _getitem_tuple -> _getitem_lowerdim -> _get_label -> xs ->
                  _get_loc_level)
        0.664s cumulative, 79 calls      predict_speed_quantiles (the actual
                                         model-invocation wrapper)
          0.508-0.514s, 395 calls          lightgbm.basic.Booster.predict()
                                           (~14% of total time)
```

**The task brief's premise was "confirm per-edge LightGBM inference
dominates." It doesn't.** Of 31,609 `eta()` calls, only 79 (0.25%) ever
reach `predict_speed_quantiles` / the LightGBM model at all — the other
31,530 (99.75%) hit a `KeyError` fallback (the edge has no METR-LA sensor
coverage; only 186/90,171 edges do) via a **failed** pandas `.loc[]`
lookup, and pandas does not resolve a failed MultiIndex lookup any faster
than a successful one. The real bottleneck is repeated pandas index
lookups in the search's hot loop — overwhelmingly failed ones — not model
inference (only ~14% of total time). Reported here exactly as found, per
this phase's hard rules, rather than silently confirming the stated
premise.

The fix below still addresses this directly: a plain Python dict is O(1)
whether the key is present or absent, unlike a pandas MultiIndex.

### 1b. Implementation

- `LightGBMPredictor.predict_times_and_sigma(df)` (`roar/predictor/lightgbm.py`,
  additive method, no existing method changed): the same per-row eta/sigma
  math as `eta_with_confidence()`, vectorized over an entire DataFrame in
  one batched call.
- `MaterializedLightGBMPredictor` (`roar/predictor/materialized_lightgbm.py`,
  new module): calls `predict_times_and_sigma()` ONCE at construction over
  every row of the feature table (every instrumented edge x every 5-minute
  bucket — the "per-time-bucket edge-cost materialization" the brief asked
  for), stores the result in a plain dict, and looks answers up there at
  query time. Anything not covered by the table (e.g. a smaller
  `features_df` was used to build it) falls back to the wrapped
  predictor's own per-row computation, LRU-style (capped at 4,096 entries).

### 1c. Path-identity proof

`tests/test_materialized_lightgbm.py`: 500 real queries (400 direct
instrumented-edge + 100 cross-network, the same stratification Phase 4/5
use), comparing `RobustAStar` guided by the base predictor vs. the
materialized one. **Zero divergence** in path or cost across all 500
queries (8m21s), plus 200 direct eta/sigma spot-checks matching to 1e-9.
This is a performance optimization, not a different model — proven, not
assumed.

### 1d. Re-run latency sweep — real numbers, target not met

New file (does NOT overwrite `results/latency_vs_graph_size.parquet`,
which used `OraclePredictor` and backs the existing Fig 3):
`results/latency_vs_graph_size_lightgbm.parquet`, via
`roar/eval/scaling_lightgbm.py`. Same graph sizes, seed, and query-sampling
methodology as the original scaling experiment (imported from there, not
retyped).

Why re-run with LightGBM at all, not just trust Fig 3: `OraclePredictor`
is not deployable (it requires already knowing the ground truth — see
`roar/api/app.py`'s docstring and `LIMITATIONS.md`). LightGBM is the
predictor a real deployment would actually use, so it's the honest
"deployability" number.

| Graph size | base (ms) | materialized (ms) | reduction |
|---|---|---|---|
| 1,000 | 49.1 | 31.1 | 37% |
| 2,000 | 52.0 | 46.4 | 11% |
| 5,000 | 151.7 | 110.0 | 27% |
| 10,000 | 400.5 | 262.0 | 35% |
| 20,000 | 897.9 | 519.2 | 42% |
| 32,696 (full) | 1,513.3 | 1,021.0 | 32.5% |

One-time materialization cost: **149.3s**. This is a real, substantial
startup cost (not a per-query cost) — a genuine tradeoff, not free: an API
process (`roar/api/app.py`) would pay this once at startup, meaning a
~2.5-minute delay before the first request can be served, in exchange for
~30-40% lower latency on every query thereafter.

**Target not met.** `906ms -> <200ms at 32,696 nodes` was the stated goal;
the materialized result is **1,021ms**, roughly 5x over target — a real,
substantial improvement (32.5% reduction from the unoptimized 1,513ms),
just not enough to hit the stated bar. Per this phase's hard rules, this
is reported as-is, not adjusted or re-run with different parameters to
look better.

**An open question, noted honestly rather than explained with an
unverified story:** the materialized LightGBM result (1,021ms) is still
*slower* than the original Oracle-based Fig 3 measurement (906ms) at the
same graph size, despite materialization replacing a pandas lookup with a
plain dict lookup — which should, in principle, be at least as fast as
Oracle's own pandas-based lookup. The two runs used the same seed and
query-sampling sequence (verified by code inspection: `scaling_lightgbm.py`
imports and calls `sample_reachable_queries` identically to
`scaling.py`), so this isn't an artifact of different queries. This gap
was not isolated further in this phase; it's flagged as something to
investigate before trusting a precise cross-predictor latency comparison,
not papered over with a plausible-sounding but unverified explanation.

## Task 2 — Stronger adversary

### Threat model

`RoutingAwareAdversarialPredictor` (`roar/predictor/routing_aware_adversarial.py`,
full threat model documented in its module docstring). Unlike the original
`AdversarialPredictor` (a static, per-edge, query-independent
underestimate), this adversary has full knowledge of the specific query
and the ground truth, and for that query: computes the true optimal
corridor, computes a genuinely worse alternate path avoiding it entirely
(the "trap"), overestimates every corridor edge, and underestimates every
trap edge — concentrating its entire budget where it can do the most
damage, rather than spending it uniformly.

Budget parametrization changed from the original predictor's `eta*(1-B)`
(requires B<1) to `eta/(1+B)` for the underestimate direction (always
positive for any B>=0), since this task sweeps B up to 2.0.

**A real bug was found and fixed while building this:** using a literal
`float("inf")` for blocked corridor edges crashed `astar()`'s
time-dependent cost evaluation (`depart_time + timedelta(seconds=inf)` ->
`OverflowError`) — a latent gap in `roar/routing/baselines.py` that no
prior caller had ever hit, since nothing before this adversary used an
infinite edge cost. Fixed with a large *finite* sentinel (1e7 seconds)
confined to the new module, not by changing the tested Phase 3/4 search
code (this phase's hard rule 2).

`tests/test_routing_aware_adversarial.py` verifies the mechanism on small,
hand-checkable graphs, including an end-to-end demonstration that
`PureMLAStarBaseline` is genuinely lured onto the trap while
`RobustAStar`'s guard falls back to the true optimum.

### 2b. Budget sweep — real separation this time

5 seeds x 80 queries x budgets {0.25, 0.5, 1.0, 2.0}, guarded vs. unguarded
vs. pure-ML, same query set as Phase 5/6 (`results/adversarial_budget_sweep.parquet`,
`results/adversary_lambda_diagnosis.parquet`).

**Guard invocation rate** (vs. the original adversary's 0.5-7% even at
B=0.95):

| Budget | Guard invocation rate |
|---|---|
| 0.25 | 0% |
| 0.5 | 6.75% |
| 1.0 | 17.0% |
| 2.0 | 25.0% |

**Statistical significance** (paired Wilcoxon + Holm-Bonferroni, identical
methodology to `table1_headline_metrics.csv`):

| Budget | vs. pure_ml_astar | vs. robust_astar_no_guard |
|---|---|---|
| 0.25 | not significant (only 3/400 pairs differ at all) | not significant (0/400 differ) |
| 0.5 | p=3.0e-8, r=-0.97, **significant** | p=3.0e-8, r=-1.00, **significant** |
| 1.0 | p=7.4e-13, r=-0.99, **significant** | p=7.6e-13, r=-1.00, **significant** |
| 2.0 | p=1.8e-18, r=-0.99, **significant** | p=3.9e-18, r=-1.00, **significant** |

At B=0.25 the result is honestly still "ns" — reported as such, not
hidden: even a routing-aware adversary needs enough budget for the lure to
actually change a routing decision. At B>=0.5, the separation is
overwhelming (effect sizes near -1.0: robust_astar consistently beats both
baselines on nearly every differing pair). **This resolves Phase 5/6's
"ns" weakness** — see `results/figures/fig4_routing_aware_adversary.png`
for the visual (worst-case ratio_vs_classical reaches 2.68x classical at
B=2.0 for the unguarded variants, while guarded stays at 1.26x, under the
1.3 floor).

### 2c. Guard-floor safety check — held

Every `robust_astar` row's realized ratio was checked against `1 + alpha`
**the moment it was computed**, not after the fact
(`roar/eval/adversarial_experiment.py`'s `_check_guard_floor`, which raises
`GuardFloorViolation` immediately if ever exceeded). Across all 4,800 +
2,400 rows: **max realized ratio = 1.2988** (bound = 1.3). **No
violations.** The guard held even against an adversary with full knowledge
of the query and the ground truth — stronger evidence for the guarantee
than holding against the original, weaker, query-independent adversary.

## Task 3 — Diagnosing (not fixing) the flat-lambda result

### Pre-registered hypothesis

Stated here before running anything for this task: **the quality
trade-off in lambda should emerge when the predictor is degraded** (Phase
6's flat lambda curve was measured against a reasonably well-calibrated
real predictor; a badly degraded one should show a different shape).

### 3a. Test: lambda sweep under degradation

`lambda_base` swept 0->1 (same 6 values as Phase 6's
`ablation_lambda_sweep`) under: `NoisyPredictor` at sigma in {0.5, 1.0,
2.0} (`exp_lambda_sweep_degraded_sigma_{0.5,1.0,2.0}.yaml`, run via
`roar/eval/lambda_diagnosis_noisy.py` — reusing `roar/eval/harness.py`'s
exact, unmodified sweep mechanism, just a different predictor spec), and
under the Task 2 adversary at a fixed high budget (B=1.0,
`roar/eval/adversarial_experiment.py`'s `run_adversary_lambda_diagnosis`).
5 seeds x 80 queries each.

### 3b. Result: the hypothesis is REFUTED

Guard-clamp (`guard_invoked`) rate as a function of lambda
(`results/figures/fig5_guard_clamp_rate_vs_lambda.png`):

| Condition | Rate at λ=0.0 | Rate at λ=1.0 | Flat within condition? |
|---|---|---|---|
| Real LightGBM (Phase 6 reference) | 0.25% | 0.25% | yes |
| NoisyPredictor σ=0.5 | 2.00% | 2.00% | yes |
| NoisyPredictor σ=1.0 | 3.75% | 3.75% | yes |
| NoisyPredictor σ=2.0 | 9.00% | 8.50% | yes (minor noise) |
| Adversary B=1.0 | 16.75% | 17.00% | yes (~flat) |

Mean `competitive_ratio` is likewise flat within every condition. **The
guard-clamp rate does NOT rise as lambda increases within any single
degradation condition — the pre-registered hypothesis does not hold.**

What DOES vary with lambda, in every condition including under the
adversary: `node_expansions` drops sharply (884.7 at λ=0 -> 140.8 at λ=1
under the adversary, `results/adversary_lambda_diagnosis.parquet`) — the
same efficiency effect Phase 6 found with the real predictor, and it
persists under heavy degradation.

What DOES vary with the degradation condition (just not with lambda
within it): the guard-clamp rate climbs from 0.25% (real predictor) up to
17% (adversary) as the predictor gets progressively worse. **Predictor
quality controls how often the guard is needed; lambda does not — not
even under significant degradation.**

### Why (a more general explanation than Phase 6's)

Phase 6 speculated the flat curve was because "a well-calibrated predictor
rarely misleads" — implying a worse predictor should show a different
curve shape. This data shows that's not the mechanism. The actual
explanation is structural: `ml_estimate(n)` (`roar/routing/robust_astar.py`)
is computed **exactly** relative to whatever cost function is actually
driving the forward search — accurate, noisy, or adversarial, it doesn't
matter which. Blending it with the classical lower bound at any lambda
therefore still guides A* toward the same optimal-path-under-that-cost-model;
lambda only changes how efficiently that path is discovered
(`node_expansions`), never which path is chosen. This holds regardless of
how wrong the underlying cost model is relative to reality, which is
exactly why degrading the predictor shifts the guard-clamp rate's *level*
between conditions but never its *shape* within one. This is reported as
the actual finding, in place of the pre-registered hypothesis it refutes,
per this phase's hard rules.

## Deliverables

- New results files (none overwrite prior ones): `latency_vs_graph_size_lightgbm.parquet`,
  `adversarial_budget_sweep.parquet`, `adversary_lambda_diagnosis.parquet`,
  `lambda_sweep_degraded_sigma_{0.5,1.0,2.0}.parquet`.
- New figures/tables via `make figures` (now also runs
  `roar.eval.figures_improvement_phase`, additive to Phase 6's
  `roar.eval.figures`, which is unmodified): `fig4_routing_aware_adversary.png`,
  `fig5_guard_clamp_rate_vs_lambda.png`, `fig6_latency_lightgbm_vs_materialized.png`,
  `table3_improvement_phase_summary.csv` (plus each figure's underlying CSV).
- New tests, all passing: `test_materialized_lightgbm.py` (path identity,
  500 real queries), `test_routing_aware_adversarial.py` (7 tests on
  hand-checkable graphs).
- `make experiments` now also runs `roar.eval.scaling_lightgbm`,
  `roar.eval.adversarial_experiment`, and `roar.eval.lambda_diagnosis_noisy`
  (additive to the existing `roar.eval.harness` and `roar.eval.scaling`
  calls, which are unchanged).
