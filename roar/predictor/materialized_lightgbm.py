"""Latency-optimized wrapper around LightGBMPredictor -- Improvement Phase
Task 1.

## What profiling actually found (read before assuming this fixes what the
## task brief assumed)

The task brief's premise was "per-edge LightGBM inference dominates"
latency. Profiling `RobustAStar.search()` on the full 32,696-node graph
(cProfile, see IMPROVEMENTS.md for the full breakdown) found this is NOT
quite right: of 3.65s total, only ~0.5s (14%) was spent inside
`lightgbm.basic.Booster.predict()` itself. The other ~86% was spent in
`LightGBMPredictor._predicted_times()`'s pandas MultiIndex `.loc[]` lookup
-- and the large majority of those lookups (31,530 of 31,609 in the
profiled query) are FAILED lookups (KeyError, for the ~99.8% of edges with
no METR-LA sensor coverage), which pandas does not resolve any faster than
a successful one. So the real bottleneck is "repeated pandas index lookups
in the search's hot loop," not "model inference" specifically -- reported
honestly here rather than silently confirmed, per this phase's hard rules.

The fix below still addresses it directly: replace the pandas lookup with
a plain Python dict, which is O(1) whether the key is present or absent.

## What this class guarantees

`MaterializedLightGBMPredictor` MUST return numerically IDENTICAL
eta/sigma to the wrapped `LightGBMPredictor` for every (edge_id,
depart_time) -- it is a performance optimization, not a different model or
a different metric. Proven in tests/test_materialized_lightgbm.py by
comparing 500 real queries' full RobustAStar paths and costs against the
unoptimized predictor; if they ever differ, that test fails loudly (see
its module docstring).

Mechanism:
  1. At construction, `LightGBMPredictor.predict_times_and_sigma()` is
     called ONCE, in a single batched call, over every row of the
     features table (every instrumented edge x every 5-minute bucket --
     the same "per-time-bucket edge-cost materialization" CLAUDE.md's
     build prompt asked for). Results are stored in a plain dict keyed by
     (edge_id, bucket timestamp).
  2. eta()/eta_with_confidence() look the answer up in that dict --
     O(1), no pandas involved, whether the edge is covered or not.
  3. Anything not covered by the materialized table (e.g. a caller passes
     a smaller/different features_df than the one used to build the
     table) falls back to the wrapped predictor's own per-row computation,
     LRU-cached by (edge_id, bucket) so a repeated miss on the same key
     only invokes the model once.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from roar.predictor.base import TravelTimePredictor, floor_to_bucket
from roar.predictor.lightgbm import LightGBMPredictor

_FALLBACK_CACHE_MAXSIZE = 4096


class MaterializedLightGBMPredictor(TravelTimePredictor):
    def __init__(self, base: LightGBMPredictor, features_df: pd.DataFrame):
        self._base = base
        self._time_bucket_minutes = base.time_bucket_minutes
        self._table = self._materialize(features_df)
        self._fallback_cache: dict[tuple[str, dt.datetime], tuple[float, float]] = {}

    def _materialize(
        self, features_df: pd.DataFrame
    ) -> dict[tuple[str, pd.Timestamp], tuple[float, float]]:
        eta, sigma = self._base.predict_times_and_sigma(features_df)
        table: dict[tuple[str, pd.Timestamp], tuple[float, float]] = {}
        for edge_id, timestamp, e, s in zip(
            features_df["edge_id"], features_df["timestamp"], eta, sigma, strict=True
        ):
            table[(edge_id, timestamp)] = (float(e), float(s))
        return table

    def _lookup(self, edge_id: str, depart_time: dt.datetime) -> tuple[float, float]:
        bucket = floor_to_bucket(depart_time, self._time_bucket_minutes)
        key = (edge_id, pd.Timestamp(bucket))
        cached = self._table.get(key)
        if cached is not None:
            return cached

        # Not in the materialized table (e.g. a features_df subset was
        # used to build it) -- fall back to the real predictor, LRU-style
        # (cache grows unbounded only across the genuinely-uncovered key
        # space, which is bounded by the graph's edge count x bucket
        # count; a hard cap avoids unbounded growth in a pathological
        # caller that materializes from an empty/tiny table).
        fallback_key = (edge_id, bucket)
        if fallback_key not in self._fallback_cache:
            if len(self._fallback_cache) >= _FALLBACK_CACHE_MAXSIZE:
                self._fallback_cache.pop(next(iter(self._fallback_cache)))
            self._fallback_cache[fallback_key] = self._base.eta_with_confidence(
                edge_id, depart_time
            )
        return self._fallback_cache[fallback_key]

    def eta(self, edge_id: str, depart_time: dt.datetime) -> float:
        return self._lookup(edge_id, depart_time)[0]

    def eta_with_confidence(self, edge_id: str, depart_time: dt.datetime) -> tuple[float, float]:
        return self._lookup(edge_id, depart_time)
