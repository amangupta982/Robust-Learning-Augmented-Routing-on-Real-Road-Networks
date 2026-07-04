# ROAR — Robust Learning-Augmented Routing on Real Road Networks

A deployable shortest-path router that uses an ML travel-time predictor as
*advice* but guarantees bounded worst-case degradation via a single tunable
trust parameter λ. First reproducible empirical study of the
consistency–robustness–smoothness curve for routing on real road-network +
traffic data.

**The headline metric is robustness, not prediction accuracy.**

See [`ROAR_Research_Blueprint.md`](ROAR_Research_Blueprint.md) for the full
research plan and [`CLAUDE.md`](CLAUDE.md) for the project's non-negotiable
rules and component roadmap.

> **Status:** project scaffold only. No routing, ML, or data-loading logic
> has been implemented yet — see the roadmap in `CLAUDE.md`.

## Project layout

```
data/{raw,processed}/       real data only — see data/README.md (added in Phase 1)
roar/graph/                 OSM + sensor loading, feature engineering
roar/predictor/             pluggable travel-time predictor interface + implementations
roar/routing/               baselines, robust A*, robustness guard
roar/eval/                  experiment harness, metrics, statistics
roar/api/                   FastAPI routing service
experiments/configs/        one YAML per experiment
experiments/notebooks/      analysis notebooks
tests/                      pytest suite, including the robustness guarantee tests
results/                    generated CSV/parquet + figures (regenerated, not hand-edited)
demo/                       interactive map demo
paper/                      LaTeX source (IEEE format)
```

## Setup

Requires Python 3.11.

```bash
make setup
```

## Reproduce everything

Once later phases land, the full pipeline from raw data to paper figures
will run end to end with:

```bash
make setup        # create venv, install pinned deps
make data          # download/cache OSM + METR-LA, build features (Phase 1)
make train         # train the travel-time predictor (Phase 2)
make experiments   # run all consistency/robustness/smoothness sweeps (Phase 5)
make figures       # regenerate every figure/table from logged results (Phase 6)
```

Every reported number in the paper must be regenerable via `make experiments`
and `make figures` — see `CLAUDE.md` for the full reproducibility rules.

## Development

```bash
make test   # run pytest
make lint   # run ruff
make api    # run the FastAPI routing service (Phase 7)
make demo   # run the interactive demo (Phase 7)
```

## License

MIT — see [`LICENSE`](LICENSE). Please cite via [`CITATION.cff`](CITATION.cff).
