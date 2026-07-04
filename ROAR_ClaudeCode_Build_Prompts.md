# ROAR — Claude Code Build Prompts (Phase by Phase)
### Publication-quality execution plan

This document contains everything you paste into Claude Code, in order. Read "How to drive this" once, then work one phase per session.

---

## How to drive this (read once — this is the professional part)

1. **One phase = one Claude Code session.** Never paste two phase prompts at once. Let it finish, then you review before moving on.
2. **Phase 0 first.** It creates `CLAUDE.md` (persistent project memory). Every later phase relies on it, so you don't repeat context each time.
3. **Review gate after every phase.** Before you accept and move on, you personally check: do the tests pass? does the code do what the acceptance criteria say? did it invent any data? If anything smells off, tell it to fix before continuing.
4. **Commit per phase.** End each phase with a git commit. Your commit history becomes part of the reproducibility story.
5. **Never let it fabricate results or data.** If a real dataset won't download or a step is blocked, it must STOP and tell you — not fake numbers, not silently generate synthetic data and present it as real. This is the single rule that separates a paper from a fraud.
6. **You own the science, it owns the code.** Claude Code implements; you decide what's a valid experiment. Read what it writes.
7. **Keep the blueprint (`ROAR_Research_Blueprint.md`) in the repo** so the agent can reference it.

---

## PHASE 0 — Project initialization & persistent memory

> **Paste to Claude Code:**

```
You are helping me build a research-paper-quality project called ROAR (Robust
Learning-Augmented Routing on Real Road Networks). This is an empirical/systems
paper in the "algorithms with predictions" area. The headline metric is
ROBUSTNESS, not prediction accuracy.

Set up the project foundation ONLY in this session. Do not implement any
algorithms yet.

Tasks:
1. Create a Python project (Python 3.11) using a src-layout package named `roar`,
   managed with a pinned requirements.txt (or uv/poetry lockfile — your choice,
   but it MUST be fully pinned for reproducibility).
2. Create the exact folder structure below and add a Makefile with targets:
   `make setup`, `make data`, `make train`, `make experiments`, `make figures`,
   `make test`, `make api`, `make demo`.
   Structure:
     data/{raw,processed}/  roar/{graph,predictor,routing,eval,api}/
     experiments/{configs,notebooks}/  tests/  results/  demo/  paper/
3. Create CLAUDE.md at the repo root containing the project memory:
   - The one-line thesis and the "robustness not accuracy" framing.
   - Target venues: SIGSPATIAL, IEEE BigData, IEEE ITSC, ML-for-systems /
     learning-augmented workshops, IEEE Access. Positioning: empirical, not
     pure theory.
   - NON-NEGOTIABLE RULES section:
     * Never fabricate data or results. If blocked, stop and report.
     * Never random-split time series — always split by whole days/weeks.
     * Every experiment must be seeded and reproducible from a config file.
     * Every reported number must be regenerable via `make experiments`.
     * The travel-time predictor is a PLUGGABLE oracle behind an interface;
       routing code must never import a concrete predictor directly.
   - The list of components to be built in later phases (predictor interface,
     baselines, robust A*, robustness guard, experiment harness, stats, api, demo).
4. Set up pytest, ruff (lint), and a GitHub Actions workflow that runs lint+tests.
5. Add MIT LICENSE, CITATION.cff, and a README with a "Reproduce everything"
   section (placeholder commands for now).
6. Initialize git and make the first commit.

Do NOT write any routing, ML, or data-loading logic yet. Confirm the structure,
show me the CLAUDE.md content, and stop.
```

**Acceptance:** repo scaffolds, `make test` runs (even with 0 tests), CI file exists, `CLAUDE.md` reads well. Commit: `chore: project scaffold + project memory`.

---

## PHASE 1 — Data & graph pipeline (REAL data only)

> **Paste to Claude Code:**

```
Read CLAUDE.md. Phase 1: build the data and graph pipeline using REAL public data.
No synthetic data anywhere in this phase.

Datasets (both public):
- Road topology: OpenStreetMap via OSMnx for a chosen city. Start with the area
  covered by the METR-LA sensor network (Los Angeles). Make the city/bbox a config
  value so we can add a second city (PEMS-BAY / San Francisco Bay Area) later.
- Traffic speeds: the METR-LA dataset (loop-detector speeds, standard in traffic
  forecasting). Provide a documented download step in data/README.md. If the file
  cannot be fetched automatically, STOP and give me the manual download instructions
  — do not invent speeds.

Build in roar/graph/:
1. `load_graph.py` — download/cache the OSMnx drive network; store nodes/edges with
   length_m, road_class (highway tag), speed_limit (maxspeed, with sane defaults per
   class), lanes, geometry. Persist to data/processed/ (GraphML or parquet + pickle).
2. `map_sensors.py` — map METR-LA sensors to their nearest road edges. This is the
   HARD, error-prone part: handle sensors that don't match well, log match distance,
   and drop or flag bad matches. Write a short data-quality report (how many sensors
   matched, distance distribution).
3. `features.py` — build per-edge, per-time-bucket (5-min) features: historical
   mean/var speed, hour-of-day, day-of-week, holiday flag, upstream/downstream
   congestion proxy, road_class, length, lanes, speed_limit. Output tidy parquet.
4. A SQLite schema (edges, edge_time_features, sensor_readings) OR document why you
   chose parquet-only. Either is fine; justify it in code comments.

Reproducibility:
- Everything driven by experiments/configs/data.yaml (city bbox, time range, bucket
  size, split boundaries).
- `make data` runs the full pipeline end to end from raw download to processed
  features.

Tests (tests/):
- Graph loads and is connected (or report the largest strongly-connected component).
- Feature table has no leakage (a feature at time t must not use speed at time >= t).
- Sensor-mapping produces a documented match rate.

Deliverables: working `make data`, a data-quality report printed and saved to
results/data_quality.md, and passing tests. Commit when done.
```

**Acceptance:** `make data` produces real processed features from real METR-LA + real OSM; data-quality report exists; leakage test passes. This phase is where most projects secretly cheat — do not let it.

---

## PHASE 2 — Travel-time predictor (with confidence)

> **Paste to Claude Code:**

```
Read CLAUDE.md. Phase 2: the pluggable travel-time predictor.

1. Define the interface in roar/predictor/base.py:
   class TravelTimePredictor with methods:
     eta(edge_id, depart_time) -> float
     eta_with_confidence(edge_id, depart_time) -> (eta: float, sigma: float)
   Routing code will depend ONLY on this interface.

2. Implement concrete predictors:
   - LightGBMPredictor (roar/predictor/lightgbm.py): trained on Phase 1 features.
     Use quantile regression (or NGBoost) to produce a real per-edge uncertainty
     sigma, not a constant. Train/val/test split MUST be by whole days/weeks
     (per CLAUDE.md), never random.
   - OraclePredictor (perfect: returns realized travel time from held-out ground
     truth) — for the consistency experiments.
   - NoisyPredictor(sigma_level): wraps the oracle and injects controlled error —
     for the smoothness curve.
   - AdversarialPredictor: worst-case corruption within a budget — for robustness.

3. Report honest predictor quality on the temporal test set: MAE, RMSE, MAPE, and
   calibration of the uncertainty (are the quantiles well-calibrated?). Save to
   results/predictor_report.md with a calibration plot.

4. `make train` trains and serializes the LightGBM model + confidence to a versioned
   artifact in results/models/.

Tests: interface conformance for all predictors; temporal-split enforcement;
serialized model loads and predicts.

Do NOT jump to a GNN. LightGBM first — it is a strong, honest baseline and keeps the
paper's story clean. We may add a GNN later as an ablation. Commit when done.
```

**Acceptance:** honest predictor metrics on a temporal test set + calibrated uncertainty; all four predictor types conform to one interface. If MAE looks suspiciously perfect, suspect leakage — investigate before continuing.

---

## PHASE 3 — Baselines (honest, correct, tested)

> **Paste to Claude Code:**

```
Read CLAUDE.md. Phase 3: classical baselines. These MUST be correct and fair —
weak baselines invalidate the whole paper.

In roar/routing/baselines.py implement:
1. Dijkstra (time-dependent edge costs).
2. Plain A* with an admissible, consistent heuristic (great-circle / Euclidean
   lower bound on travel time using max speed limit). Prove admissibility in a
   comment.
3. Contraction Hierarchies (or clearly document why we substitute bidirectional
   Dijkstra for scale) — a strong speed baseline.
4. Pure-ML-guided A*: uses the LightGBM ETA directly in the heuristic with NO
   robustness mechanism. This is the "unsafe" baseline we will beat on robustness.

Requirements:
- All baselines share one interface: search(origin, dest, depart_time) ->
  {path, cost, node_expansions, latency_ms}.
- Correctness tests: on small hand-checkable graphs, Dijkstra and A* must return
  identical optimal costs; A* must expand <= Dijkstra nodes when heuristic is good.
- A fairness test: all methods run on the exact same graph, costs, and queries.

Add tests/ for each. Commit when done.
```

**Acceptance:** Dijkstra and A* agree on optimal cost on test graphs; baselines are genuinely competitive (a strawman baseline is worse than no baseline). Commit: `feat: routing baselines`.

---

## PHASE 4 — Robust A* + robustness guard (the core contribution)

> **Paste to Claude Code:**

```
Read CLAUDE.md. Phase 4: the core research contribution — robust learning-augmented
A* with a provable degradation floor.

In roar/routing/:
1. robust_astar.py — A* whose heuristic blends the classical admissible lower bound
   with the ML estimate via a trust parameter lambda in [0,1]:
       h(n) = classical_lb(n) + lambda * (ml_estimate(n) - classical_lb(n))
   and per-query lambda modulation from predictor confidence sigma (higher sigma ->
   lower trust). Make both fixed-lambda and confidence-modulated modes available.

2. guard.py — RobustnessGuard that clamps realized edge costs so that the final path
   cost can never exceed (1 + alpha) times the classical-optimal cost, for a
   configurable alpha. This is the robustness FLOOR. Implement it as an explicit,
   testable mechanism, and write a clear docstring stating the guarantee and its
   assumptions.

3. Add a `robustness_bound` field to the search output reporting the realized ratio
   and whether the guard was ever invoked.

CRITICAL tests (this is the heart of the paper):
- Consistency: with OraclePredictor, robust A* returns the optimal path (ratio ~1).
- Robustness: with AdversarialPredictor at maximum corruption, the realized path
  cost NEVER exceeds (1+alpha)*optimal across a large random query set. If it ever
  does, the guard is broken — the test must fail loudly.
- Monotonicity smoke test: with NoisyPredictor, increasing sigma should not decrease
  average path quality below the guard floor.

If you cannot make the guard provably hold, STOP and tell me — we will weaken the
theoretical claim and reposition, rather than ship a false guarantee. Commit when
done.
```

**Acceptance:** the robustness test passes over thousands of adversarial queries — the guard floor genuinely holds empirically. This test passing is the difference between a paper and a blog post. Commit: `feat: robust A* + robustness guard`.

---

## PHASE 5 — Experiment harness & sweeps

> **Paste to Claude Code:**

```
Read CLAUDE.md. Phase 5: the experiment harness that generates every number in the
paper.

In roar/eval/:
1. harness.py — reads an experiments/configs/*.yaml specifying: city, predictor type,
   lambda values, error levels (sigma), adversarial budgets, query set, seeds.
   Runs all methods (baselines + robust A*) on the SAME stratified query set
   (thousands of O/D/depart-time triples stratified by distance and time-of-day).
2. metrics.py — for each query, compute competitive ratio vs the oracle-optimal path,
   node expansions, latency_ms, and guard-invocation flag. Log to
   results/<run_id>.parquet with full run metadata.
3. A config that produces the three core experiments:
   - Consistency sweep (predictor quality: oracle -> real -> degraded).
   - Robustness sweep (adversarial budget 0 -> max).
   - Smoothness sweep (NoisyPredictor sigma 0 -> high) on the SAME queries.
4. Ablation configs: (a) guard on vs off, (b) fixed vs confidence-modulated lambda,
   (c) lambda sweep for the Pareto front, (d) feature ablation hook.

Requirements: fully seeded; `make experiments` regenerates all result parquets from
scratch; each run records library versions and git commit hash. No plotting here —
just clean logged data. Tests: harness runs a tiny config end-to-end deterministically
(same seed -> same numbers). Commit when done.
```

**Acceptance:** `make experiments` regenerates all raw result files deterministically; every method ran on identical queries. Commit: `feat: experiment harness + sweeps`.

---

## PHASE 6 — Analysis, statistics & figures

> **Paste to Claude Code:**

```
Read CLAUDE.md. Phase 6: turn logged results into the paper's figures and statistics.
Correctness of statistics matters as much as the code.

In roar/eval/stats.py and experiments/notebooks/:
1. Statistical validation:
   - Report mean +/- 95% bootstrap CI over queries and over >=5 seeds.
   - Paired Wilcoxon signed-rank tests (ratios are non-normal) comparing robust A*
     vs each baseline on the SAME queries; report effect sizes, not just p-values.
   - Holm-Bonferroni correction across the multiple baseline comparisons.
   State every test's assumptions in comments.
2. Figures (save to results/figures/, publication-quality, colorblind-safe):
   - Fig 1 (the money plot): competitive ratio vs prediction error — robust A* vs
     pure-ML-A* vs classical, showing robust A* never exceeds the guard floor while
     pure-ML blows up.
   - Fig 2: consistency-vs-robustness Pareto front as lambda varies.
   - Fig 3: latency vs graph size (deployability).
   - Table 1: headline metrics per city per method with CIs + significance markers.
   - Table 2: ablation results.
3. `make figures` regenerates every figure and table from the parquet logs — no
   manual steps, no hand-edited numbers.

Tests: stats functions validated on synthetic inputs with known answers (this is the
ONE place synthetic data is legitimate — for unit-testing the math, never for
results). Commit when done.
```

**Acceptance:** every figure/table regenerates from logged data via `make figures`; stats include CIs, paired tests, corrections, and effect sizes. Commit: `feat: analysis, statistics, figures`.

---

## PHASE 7 — API, demo & reproducibility packaging

> **Paste to Claude Code:**

```
Read CLAUDE.md. Phase 7: make it deployable and fully reproducible.

1. API (roar/api/app.py, FastAPI): POST /route {origin,destination,depart_time,
   lambda?} -> {path, cost, eta, node_expansions, latency_ms, robustness_bound};
   GET /predictor/health. Stateless; load graph + model once at startup. Dockerize.
2. Demo (demo/): a Streamlit or Leaflet map where I pick origin/destination and see
   the robust route, with a lambda slider that visibly trades consistency for
   robustness. Host-ready for a free tier.
3. Reproducibility packaging:
   - `docker compose up` brings up API + demo.
   - Update README "Reproduce everything": make setup -> make data -> make train ->
     make experiments -> make figures reproduces every number and figure in the paper.
   - Add results/ CSVs behind every figure, CITATION.cff, and a short screencast
     placeholder.
   - Ensure CI still passes (lint + tests).
4. Write a LIMITATIONS.md honestly listing threats to validity (sensor-mapping error,
   single-city generalization if second city not done, guard assumptions, predictor
   scope). Reviewers reward this.

Commit when done. Then summarize what a reviewer could still attack, so I know what to
harden before submission.
```

**Acceptance:** one command brings up the demo; one documented sequence reproduces the whole paper; honest limitations written. Commit: `feat: api, demo, full reproducibility`.

---

## Global research-integrity checklist (re-read before you submit)

- [ ] No synthetic data was ever passed off as real (synthetic used ONLY for unit-testing math).
- [ ] All time-series splits are by whole days/weeks — zero random splits.
- [ ] Baselines are strong and fair, not strawmen.
- [ ] The robustness guard's floor holds empirically across thousands of adversarial queries.
- [ ] Every number in the paper regenerates from `make experiments && make figures`.
- [ ] Statistics: CIs + paired non-parametric tests + multiple-comparison correction + effect sizes.
- [ ] Every claim in the paper maps to a specific figure/table/test.
- [ ] LIMITATIONS.md is honest and complete.
- [ ] Git history shows clean per-phase progress; CI is green.

If any box is unchecked, you are not ready to submit — fix it, don't hide it.
```
