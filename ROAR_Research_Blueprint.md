# ROAR — Robust Learning-Augmented Routing on Real Road Networks
### Complete Research Blueprint (Step 5)

> **One-line thesis:** A deployable shortest-path router that uses an ML travel-time predictor as *advice* but guarantees bounded worst-case degradation via a single tunable trust parameter λ — with the first reproducible empirical study of the consistency–robustness–smoothness curve on real road-network + traffic data.

> **Positioning reminder:** This is an *empirical / systems* paper in the algorithms-with-predictions area, NOT a pure-theory paper. The headline metric is **robustness**, not prediction accuracy. Target venues: SIGSPATIAL, IEEE BigData, IEEE ITSC, ML-for-systems / learning-augmented-algorithms workshops, IEEE Access.

---

## 1. Literature Review Strategy

**Three concentric rings — read in this order:**

1. **Core framework (read deeply, ~10 papers):**
   - Mitzenmacher & Vassilvitskii, "Algorithms with Predictions," CACM 2022 (the manifesto).
   - McCauley, Moseley, Niaparast, Niaparast, Singh, "Incremental Approximate SSSP with Predictions," ICALP 2025 (closest prior art — know it cold).
   - Benomar & Coester, "Learning-Augmented Priority Queues," NeurIPS 2024.
   - Lykouris & Vassilvitskii, "Competitive Caching with ML Advice," JACM 2021 (origin of consistency/robustness).
   - Purohit, Svitkina, Kumar, "Improving Online Algorithms via ML Predictions," NeurIPS 2018 (the λ-interpolation trick you will adapt).

2. **Applied routing / travel-time (read for baselines & datasets, ~10 papers):**
   - DeepTravel; geo-convolution TTE; spatio-temporal GNN TTE; RLER-TTE.
   - Contraction Hierarchies (Geisberger et al.) and time-dependent CH.
   - Stochastic / path-centric routing in uncertain road networks.

3. **Methodology / reproducibility (read for rigor, ~5 papers):**
   - The DIMACS road-network challenge and ALENEX experimental-algorithmics norms.
   - Any recent "benchmark paper" in your target venue to copy its structure.

**Search discipline:** Maintain a living table (Zotero/Notion) with columns: *Paper · Problem · Uses real road data? · Gives robustness guarantee? · Deployed? · Metric reported*. The empty cells in "real data + robustness + deployed" **are your contribution** — literally screenshot that table for the paper's related-work section.

**Trigger to pivot:** If you find a 2025–2026 paper filling all three columns for routing, switch to the "empirical benchmark for learning-augmented dynamic-graph data structures" idea instead.

---

## 2. Research Questions

- **RQ1 (Consistency):** When the travel-time predictor is accurate, does prediction-guided A* approach oracle-optimal routing quality and reduce node expansions vs classical A*?
- **RQ2 (Robustness):** Under adversarial or incident-corrupted predictions, can we *bound* worst-case path suboptimality, and does the λ knob deliver that bound in practice?
- **RQ3 (Smoothness):** How does route quality and search effort degrade as prediction error grows from 0 → high? Is degradation graceful and monotone?
- **RQ4 (Deployability):** Does the method run within interactive latency (< 200 ms/query) on a city-scale graph on commodity hardware?

## 3. Hypotheses

- **H1:** There exists a λ-parameterized router that is (1+ε)-consistent and c-robust, with ε → 0 as λ → prediction-trust and c bounded as λ → classical.
- **H2:** On real data, the empirical robustness curve dominates both pure-ML-guided A* (which has no floor) and classical A* (which ignores learnable structure) on a Pareto sense (quality vs worst-case).
- **H3:** Prediction confidence, used to modulate λ *per-query*, beats a fixed global λ.

---

## 4. System Architecture

```
                 ┌────────────────────────────────────────────┐
                 │                CLIENT / DEMO                 │
                 │   (Streamlit or Leaflet map: pick O, D)      │
                 └───────────────────────┬──────────────────────┘
                                         │ REST
                 ┌───────────────────────▼──────────────────────┐
                 │              ROUTING SERVICE (API)            │
                 │  ┌──────────────┐   ┌───────────────────────┐ │
                 │  │  Graph Store │   │  Robust A* Engine     │ │
                 │  │ (OSM, edges, │──▶│  h(n)=blend(class,ML,λ)│ │
                 │  │  time-buckets)│   │  fallback + guarantee │ │
                 │  └──────────────┘   └──────────┬────────────┘ │
                 │                                 │ query edge   │
                 │  ┌──────────────────────────────▼───────────┐ │
                 │  │  Travel-Time Predictor (LightGBM / GNN)   │ │
                 │  │  input: edge feats + time + weather       │ │
                 │  │  output: ETA + confidence σ               │ │
                 │  └───────────────────────────────────────────┘ │
                 └────────────────────────────────────────────────┘
```

**Key design choice:** the predictor is a *pluggable oracle*. Swap it (perfect oracle, real model, adversarial) without touching the routing core — this is what makes clean ablations possible.

---

## 5. UML Diagrams (describe these; draw in draw.io / PlantUML)

**Class diagram (core):**
- `Graph` (nodes, edges, `getNeighbors()`, `staticWeight(e)`)
- `TravelTimePredictor` (interface) → `LightGBMPredictor`, `OraclePredictor`, `AdversarialPredictor`, `NoisyPredictor(σ)`
- `RobustAStar` (fields: `lambda`, `predictor`, `graph`; methods: `heuristic(n, goal)`, `edgeCost(e, t)`, `search(o, d, departTime)`)
- `RobustnessGuard` (enforces admissible fallback bound; method `clamp(mlCost, classicalBound)`)
- `Evaluator` (runs experiments, logs metrics)

**Sequence diagram (one query):** Client → API → RobustAStar.search → (loop) Predictor.eta(edge,t) → RobustnessGuard.clamp → priority-queue expand → return path + diagnostics.

**Component diagram:** Client · API Gateway · Routing Engine · Predictor Service · Graph Store · Experiment Harness.

---

## 6. AI/ML Pipeline

1. **Data ingest:** OSMnx pulls city graph (e.g., Los Angeles / San Francisco to match METR-LA / PEMS-BAY sensors). Map sensor speeds → edges.
2. **Feature engineering (per edge, per 5-min bucket):** road class, length, speed limit, lanes, historical mean/var speed, hour-of-day, day-of-week, holiday flag, upstream/downstream congestion, weather (optional).
3. **Predictor:** start with **LightGBM/XGBoost** (fast, strong, honest baseline — do NOT jump to a GNN first). Optionally add a spatio-temporal GNN as a second predictor for a "does a stronger oracle help?" experiment.
4. **Confidence σ:** use quantile regression (LightGBM supports it) → gives per-edge uncertainty that feeds per-query λ.
5. **Robust routing:** heuristic `h(n) = classical_lower_bound(n) + λ·(ml_estimate(n) − classical_lower_bound(n))`, clamped so realized edge cost can never exceed a `(1+α)`-inflated classical bound → guarantees the robustness floor.
6. **Evaluation harness:** swap predictors, sweep λ and error level, log everything to CSV/Parquet.

---

## 7. Database Design

Lightweight — this is not a data-heavy app.

- **`edges`** (edge_id PK, u, v, length_m, road_class, speed_limit, lanes, geometry)
- **`edge_time_features`** (edge_id FK, time_bucket, hist_mean_speed, hist_var, congestion_idx) — composite PK (edge_id, time_bucket)
- **`sensor_readings`** (sensor_id, edge_id FK, timestamp, speed) — training source
- **`experiment_runs`** (run_id PK, predictor_type, lambda, error_level, seed, timestamp)
- **`results`** (run_id FK, query_id, path_cost, oracle_cost, ratio, node_expansions, latency_ms)

Storage: **SQLite** (single-file, deployable, reproducible) or Parquet files for the experiment logs.

---

## 8. API Design (REST)

- `POST /route` → body `{origin, destination, depart_time, lambda?}` → `{path, cost, eta, node_expansions, latency_ms, robustness_bound}`
- `GET /predictor/health` → predictor status + last-trained timestamp
- `POST /experiment` → run a batch sweep (internal/admin) → `{run_id}`
- `GET /results/{run_id}` → metrics for a run

Keep it stateless; the graph + model load once at startup.

---

## 9. Folder Structure

```
roar/
├── data/
│   ├── raw/                # OSM extracts, sensor CSVs
│   ├── processed/          # feature parquet
│   └── README.md           # exact download commands (reproducibility!)
├── roar/
│   ├── graph/              # OSM loading, edge features
│   ├── predictor/          # lightgbm.py, gnn.py, oracle.py, adversarial.py
│   ├── routing/            # astar.py, robust_astar.py, guard.py, baselines.py
│   ├── eval/               # harness.py, metrics.py, stats.py
│   └── api/                # app.py (FastAPI/Flask)
├── experiments/
│   ├── configs/            # yaml per experiment
│   └── notebooks/          # analysis, figures
├── tests/                  # pytest — test the guarantee empirically
├── results/                # csv/parquet + generated figures
├── demo/                   # streamlit or leaflet app
├── paper/                  # LaTeX (IEEE template)
├── requirements.txt
├── Makefile                # `make data`, `make train`, `make experiments`, `make figures`
└── README.md               # reproduce-everything instructions
```

---

## 10. Development Roadmap (≈ 5 months, solo)

| Phase | Weeks | Deliverable |
|---|---|---|
| 0. Lit review + gap table | 1–3 | The 3-column table; confirmed novelty; pre-registered RQs |
| 1. Data + graph pipeline | 3–5 | OSM graph + sensor-mapped features, SQLite/Parquet |
| 2. Predictor (LightGBM) | 5–7 | Trained ETA model + quantile confidence, held-out error report |
| 3. Baselines | 6–8 | Dijkstra, A*, CH, pure-ML-A* all working + tested |
| 4. Robust A* + guard | 8–11 | λ-router + robustness clamp; **empirical test that the bound holds** |
| 5. Experiments | 11–15 | Full consistency/robustness/smoothness sweeps; ablations |
| 6. Analysis + stats | 14–16 | Figures, significance tests, result tables |
| 7. Writing + demo | 15–20 | IEEE paper draft, live demo, open-source release |

Build in slack — everyone underestimates phase 5.

---

## 11. Experimental Methodology

- **Datasets:** ≥ 2 cities (generalization). METR-LA / PEMS-BAY for speeds; OSM for topology; hold out entire days/weeks for temporal generalization (never random-split time series).
- **Query sampling:** thousands of (O, D, depart-time) triples stratified by distance and time-of-day.
- **Oracle:** compute true shortest path under *realized* travel times (ground truth) for the ratio denominator.
- **Prediction-error control:** the `NoisyPredictor(σ)` and `AdversarialPredictor` let you sweep error from 0 → high *on the same queries* — this produces the smoothness curve, the paper's money figure.
- **Hardware/latency:** report on one commodity machine; fix seeds; log versions.

## 12. Ablation Studies

1. **Remove the robustness guard** → show worst-case blows up (justifies the guard).
2. **Fixed λ vs confidence-modulated λ** → justifies per-query λ (H3).
3. **LightGBM vs GNN predictor** → does a stronger oracle actually help routing, or does the guard dominate?
4. **Feature ablation** → which features drive prediction gains.
5. **λ sweep** → the consistency↔robustness Pareto front.

## 13. Statistical Validation

- Report **mean ± 95% CI** over queries and over ≥ 5 seeds.
- **Paired tests** (Wilcoxon signed-rank, since ratios aren't normal) comparing ROAR vs each baseline on the same queries; report effect sizes, not just p-values.
- **Bonferroni/Holm correction** across the multiple baseline comparisons.
- Bootstrap CIs for the competitive-ratio curves.

## 14. Result-Analysis Plan

- **Figure 1:** smoothness curve — competitive ratio vs prediction error, ROAR vs pure-ML vs classical (the killer plot).
- **Figure 2:** Pareto front — consistency (x) vs robustness (y) as λ varies.
- **Table 1:** headline metrics per city per method with CIs + significance.
- **Table 2:** ablations.
- **Figure 3:** latency vs graph size (deployability).
- Narrative: lead with robustness, show ROAR is never worse than classical yet captures most of the ML upside.

---

## 15. GitHub Repository Structure (for release)

Mirror §9, plus: MIT license, `CITATION.cff`, a `reproduce.md` with exact commands, pinned `requirements.txt`/lockfile, a `results/` folder with the CSVs behind every figure, GitHub Actions running `pytest`, and a short screencast of the demo. Reproducibility is a *reviewer trust signal* — treat it as part of the contribution.

## 16. Deployment Plan

- **API:** FastAPI + Uvicorn, Dockerized.
- **Demo:** Streamlit (fast) or a Leaflet map front-end; host on a free tier (Streamlit Cloud / HF Spaces / Render).
- **Model:** ship the trained LightGBM as a versioned artifact; load at startup.
- **Repro:** `docker compose up` should bring up API + demo. `make experiments` should regenerate every number in the paper.

---

## 17. Research Paper Outline (IEEE format)

**Title:** *ROAR: Robust Learning-Augmented Routing with Provable Degradation Bounds on Real Road Networks*

- **Abstract** (200 words): problem, the robustness-first framing, λ mechanism, key empirical result (e.g., "matches ML routing under good predictions, never worse than classical under bad ones"), open release.
- **I. Introduction:** motivation (nav systems fail unpredictably), the accuracy-vs-robustness gap, contributions bulleted (algorithm + guarantee + first real-data benchmark + open artifact).
- **II. Related Work:** algorithms-with-predictions; ML travel-time estimation; classical/CH routing; stochastic routing. End with the 3-column gap table.
- **III. Preliminaries & Problem Formulation:** graph, time-dependent costs, predictor oracle, definitions of consistency/robustness/smoothness for routing.
- **IV. Method:** the λ-blended heuristic, the robustness guard, per-query confidence modulation, and the (light) theoretical statement of the consistency/robustness bound with proof sketch in an appendix.
- **V. Experimental Setup:** datasets, predictors, baselines, metrics, hardware.
- **VI. Results:** RQ1–RQ4, the smoothness curve, Pareto front, ablations, significance.
- **VII. Deployment & Demo.**
- **VIII. Limitations & Threats to Validity** (be honest — reviewers reward this).
- **IX. Conclusion & Future Work.**
- **References** · **Appendix:** proof sketch, extra plots, reproducibility details.

---

### Final honest note
The paper lives or dies on **Figure 1 (smoothness curve)** and on the **guard actually holding empirically**. If those are clean and reproducible, you have a real contribution. If the guarantee is hand-wavy, cut the theory claim and sell it purely as the first rigorous empirical benchmark — that alone is publishable at an applied venue. Do not overclaim; reviewers punish overclaiming far more than modest scope.
