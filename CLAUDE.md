# ROAR — Project Memory

## Thesis

A deployable shortest-path router that uses an ML travel-time predictor as
*advice* but guarantees bounded worst-case degradation via a single tunable
trust parameter λ — with the first reproducible empirical study of the
consistency–robustness–smoothness curve on real road-network + traffic data.

**The headline metric is ROBUSTNESS, not prediction accuracy.** Every design
and implementation decision should be judged against: "does this make the
robustness guarantee more real, more testable, more honest?" — not "does this
make the predictor more accurate?"

## Positioning

This is an **empirical / systems** paper in the algorithms-with-predictions
area, NOT a pure-theory paper.

**Target venues:** SIGSPATIAL, IEEE BigData, IEEE ITSC, ML-for-systems /
learning-augmented-algorithms workshops, IEEE Access.

Full background, literature strategy, RQs, hypotheses, architecture, and
paper outline live in `ROAR_Research_Blueprint.md` — read it for context
before starting a new phase. The phase-by-phase execution plan lives in
`ROAR_ClaudeCode_Build_Prompts.md`.

## NON-NEGOTIABLE RULES

1. **Never fabricate data or results.** If a dataset won't download, a step
   is blocked, or a guarantee can't be made to hold — STOP and report it.
   Do not silently substitute synthetic data and present it as real. The one
   exception: synthetic inputs to unit-test statistics/math code, never to
   produce reported results.
2. **Never random-split time series.** Always split by whole days/weeks.
   Random splits on temporal data leak future information into training and
   invalidate every downstream number.
3. **Every experiment must be seeded and reproducible from a config file.**
   No hard-coded parameters inside experiment code — they live in
   `experiments/configs/*.yaml`.
4. **Every reported number must be regenerable via `make experiments`**
   (and `make figures` for plots/tables). No hand-edited numbers anywhere.
5. **The travel-time predictor is a PLUGGABLE oracle behind an interface**
   (`roar/predictor/base.py`). Routing code (`roar/routing/`) must never
   import a concrete predictor class directly — only the interface.

## Components (build order, later phases)

| Phase | Component | Location |
|---|---|---|
| 1 | Data & graph pipeline (OSM + METR-LA, real data only) | `roar/graph/` |
| 2 | Predictor interface + LightGBM/Oracle/Noisy/Adversarial predictors | `roar/predictor/` |
| 3 | Baselines: Dijkstra, A*, CH (or bidirectional Dijkstra), pure-ML-A* | `roar/routing/baselines.py` |
| 4 | Robust A* + RobustnessGuard (the core contribution) | `roar/routing/robust_astar.py`, `roar/routing/guard.py` |
| 5 | Experiment harness + sweeps (consistency/robustness/smoothness) | `roar/eval/harness.py` |
| 6 | Statistics + figures (CIs, Wilcoxon, Holm-Bonferroni) | `roar/eval/stats.py` |
| 7 | API + demo + reproducibility packaging | `roar/api/`, `demo/` |

## Working agreement

- One phase per session. Read this file first, do only that phase's tasks,
  stop at its boundary.
- Commit at the end of each phase with a descriptive message.
- If something can't be verified (guarantee doesn't hold, data doesn't
  download, split leaks), stop and say so rather than pushing forward.
