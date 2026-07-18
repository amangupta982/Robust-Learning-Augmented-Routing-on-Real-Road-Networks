# Limitations

Written honestly and specifically, not as a boilerplate disclaimer. Every
item below names the concrete mechanism, points at the code/data that
causes it, and (where relevant) the number that demonstrates it. If you're
deciding whether to trust a specific claim in the paper, check whether it
depends on one of these.

## 1. Data & scope

- **Single city, single dataset.** Every result is on the LA drive network
  with METR-LA sensor data. No second city was run. Findings about
  robustness/consistency/smoothness may be specific to LA's
  freeway-dominated topology, its sensor placement, and its 2012 traffic
  patterns -- they have not been shown to generalize to denser urban
  grids, different road conventions, or other traffic regimes.
- **The data is from 2012.** METR-LA (Li et al., DCRNN) is 14 years old at
  time of writing. Road network structure, typical congestion patterns,
  and driving behavior have likely changed since; the trained predictor
  and every downstream number reflect 2012 LA traffic, not current
  conditions.
- **Predictor coverage is ~0.2% of the graph.** Only 186 of ~90,171 edges
  (results/data_quality.md) have a real METR-LA sensor. Every other edge
  falls back to a static, speed-limit-based free-flow cost regardless of
  which predictor (oracle, LightGBM, noisy, adversarial) is in play. This
  is the single biggest limitation on how much any of this project's
  routing mechanisms can matter in practice: a predictor -- honest or
  adversarial -- can only influence a route where it actually has an
  opinion.
- **Sensor-to-edge mapping can be wrong.** `roar/graph/map_sensors.py`
  snaps each sensor to its nearest OSM edge by geographic distance; this
  run matched 207/207 sensors within 200m (mean 2.6m, max 11.9m -- see
  results/data_quality.md), but nearest-edge matching is inherently
  capable of snapping a sensor to the wrong carriageway/direction of a
  divided highway, especially near interchanges. A silently-wrong mapping
  would make that edge's "real" data actually describe a different,
  nearby edge.

## 2. The predictor

- **LightGBM is honestly mediocre, not strong.** results/predictor_report.md:
  MAE ~87s, MAPE ~38.6% on the held-out temporal test set. Consistency-sweep
  conclusions ("does a better predictor help") should be read as "does THIS
  fairly weak real predictor help," not as an upper bound on what a
  stronger model (the GNN ablation CLAUDE.md explicitly defers) would show.
- **The feature-ablation hook was exercised for exactly one feature**
  (`neighbor_congestion_proxy`, dropping it roughly doubled the competitive
  ratio -- results/figures/table2_ablations.csv) because each ablation
  point is a real LightGBM retrain (~5 min). Broader feature-importance
  claims are not established; the hook is real and reusable, but only one
  data point has been generated from it.
- **Confidence-modulated lambda showed no measurable difference from fixed
  lambda** in ablation_lambda_mode.parquet. This could mean the mechanism
  genuinely doesn't matter for this predictor, OR that the real LightGBM
  model's per-edge sigma isn't differentiated enough across edges to drive
  meaningfully different trust levels, OR the same query-set skew in
  Limitation 4 below. Not disentangled here.

## 3. The robustness guarantee itself

- **The (1+alpha) bound is relative to the classical free-flow baseline,
  NOT the hindsight-optimal path.** A route can satisfy the guard's bound
  and still be much worse than the best possible route in reality --
  robust A*'s `competitive_ratio` (vs. the true oracle-optimal cost) has
  been observed well above (1+alpha) in these experiments even while its
  `ratio_vs_classical` (the metric the guarantee is actually about) stays
  under the floor. Don't conflate the two ratios; only one of them is
  guaranteed.
- **The guard requires an independent ground-truth cost function to
  evaluate a candidate before returning it** (roar/routing/guard.py). In
  the paper's offline replay (Phases 4-6) that's `OraclePredictor` --
  legitimate because every replayed query's true outcome is already known.
  **The live API/demo (Phase 7) has no such foreknowledge** and instead
  uses the SAME LightGBM predictor as both planner and "ground truth"
  reference (see `roar/api/app.py`'s module docstring). That makes the
  demo's `robustness_bound` a **self-consistency guard** (protects against
  the search algorithm returning a policy that's bad even by the
  predictor's own model) -- it is explicitly NOT the paper's stronger
  empirical guarantee against reality. A real deployment would need a live
  ground-truth signal or an online monitoring/replanning mechanism; neither
  exists here.
- **Efficiency shortcuts are approximations, not part of the correctness
  guarantee, but they can shift which candidate the guard evaluates:**
  `bidirectional_dijkstra`'s frozen-weight approximation, and
  `RobustAStar`'s bounded-radius backward Dijkstra for `ml_estimate` (both
  documented in roar/routing/baselines.py and robust_astar.py). Neither
  affects the (1+alpha) bound itself (that's proven independent of
  heuristic/search quality), but a tighter or looser approximation could
  change which path gets proposed and hence the reported consistency
  numbers.

## 4. Experimental design

- **The query set is skewed toward single-edge queries by construction**
  (`direct_fraction: 0.85` in experiments/configs/query_set.yaml, chosen to
  bound compute time -- cross-network queries are the expensive ones due
  to RobustAStar's backward-search radius). This means most queries offer
  little genuine alternate-route opportunity, which is why the guard fires
  on only ~0.5-7% of queries across the ablations
  (results/figures/table2_ablations.csv) even under near-maximum
  adversarial corruption. The (1+alpha) guarantee still holds exactly
  (it's proven, not estimated), but the empirical PICTURE of how often
  robustness actually matters likely understates real-world risk on
  longer, more route-rich trips. A query set with a higher cross-network
  fraction would show the guard engaging more often.
- **200-400 queries per experiment, 5 seeds, not "thousands."** CLAUDE.md's
  build plan called for thousands of queries and >=5 seeds; the seed count
  was met, but the per-seed query count (80, from
  experiments/configs/query_set.yaml) was reduced from an original 200 to
  keep a full `make experiments` run to well under an hour in this
  environment. This limits statistical power (see below) and the diversity
  of the stratified sample. Raising `n_queries` is a one-line config change
  for a full-scale run.
- **Fig 3 (latency vs. graph size) uses BFS-truncated subgraphs of the real
  LA network, not independently-sourced smaller cities.** This is real
  topology, not fabricated, but a truncated subgraph's structure (e.g. edge
  density near the truncation boundary) isn't necessarily representative of
  what an actually-smaller city's road network looks like.
- **Bidirectional Dijkstra substitutes for Contraction Hierarchies**
  (documented design choice, roar/routing/baselines.py) as the "speed
  baseline." It is a real, correct speedup technique, but a production CH
  implementation would likely be faster still; Fig 3's speed comparison
  should not be read as "the fastest possible classical baseline."

## 5. Statistical power

- With 5 seeds and 80-200 queries per seed, confidence intervals on
  seed-level means (roar/eval/stats.py's `seed_level_summary`) are wide in
  several places (see results/figures/table1_headline_metrics.csv) --
  e.g. the classical baseline's own CI spans roughly 2.0-3.0. Effect sizes
  and significance markers should be read with that in mind; a "ns" result
  in Table 1 can mean "no effect" or "not enough seeds to tell," and this
  project cannot distinguish those from 5 seeds alone.
- No formal power analysis was conducted to determine how many
  seeds/queries would be needed for a target minimum detectable effect.

## 6. Deployment (Phase 7)

- **Docker packaging is untested in this environment** (no Docker
  installed here to build/run it). The `Dockerfile` and
  `docker-compose.yml` follow standard, carefully-reasoned patterns and the
  API/demo were verified directly (FastAPI TestClient, Streamlit AppTest,
  and a live uvicorn+streamlit run against real data), but `docker compose
  up` itself has not been executed end-to-end. Verify it before relying on
  it for a submission artifact.
- **The API/demo only accept real OSM node IDs**, not addresses or
  arbitrary lat/lon -- there is no geocoding layer. The demo works around
  this by only offering a curated list of real METR-LA-instrumented edge
  endpoints (the only nodes with genuine predictor coverage), not arbitrary
  user-chosen locations.
- **No live traffic feed.** The API/demo operate entirely within the METR-LA
  2012 date range (2012-03-01 to 2012-06-27); there is no mechanism to
  route against real current conditions.
