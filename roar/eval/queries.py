"""Stratified query-set generation for the Phase 5 experiment harness.

Every experiment sweep (consistency/robustness/smoothness/ablations) runs
every method on the EXACT SAME query set -- generated ONCE here and reused
by every experiment config `roar/eval/harness.py` runs, so "same queries
across experiments" holds by construction, not by coincidentally matching
seeds.

Scope note (inherited from roar/graph/features.py and Phase 4's test
fixtures): only ~186 of ~90171 edges are METR-LA-instrumented, so uniformly
random node pairs would rarely touch real sensor data at all. Queries are
therefore built from the endpoints of instrumented edges -- either a single
instrumented edge directly (shorter distance) or two different
instrumented edges' endpoints (longer, multi-hop) -- the same approach
validated in Phase 4's tests (tests/robust_astar_fixtures.py).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import random

import pandas as pd

from roar.graph.map_sensors import SENSOR_EDGE_MAP_PATH
from roar.routing.baselines import classical_lower_bound_heuristic
from roar.routing.graph import RoutingGraph


@dataclasses.dataclass(frozen=True)
class Query:
    query_id: int
    origin: str
    dest: str
    depart_time: dt.datetime
    distance_stratum: str
    time_stratum: str


def _instrumented_edge_endpoints() -> list[tuple[str, str]]:
    sensor_map = pd.read_parquet(SENSOR_EDGE_MAP_PATH)
    sensor_map = sensor_map[sensor_map["matched"]]
    return [(str(u), str(v)) for u, v in zip(sensor_map["u"], sensor_map["v"], strict=False)]


def _time_stratum(depart_time: dt.datetime, time_strata: dict[str, list[int]]) -> str:
    hour = depart_time.hour
    for name, (start, end) in time_strata.items():
        if start <= hour < end:
            return name
    return "other"


def generate_stratified_queries(
    graph: RoutingGraph,
    features_df: pd.DataFrame,
    cfg: dict,
    seed: int,
) -> list[Query]:
    """Builds `cfg['n_queries']` queries stratified by (distance_stratum,
    time_stratum), sampled from real instrumented-edge endpoints and real
    test-split timestamps. Fully deterministic given `seed` (same seed ->
    byte-identical query list, since sampling only ever uses this one
    `random.Random(seed)` instance, in a fixed order).
    """
    rng = random.Random(seed)
    endpoints = _instrumented_edge_endpoints()
    test_timestamps = (
        features_df.loc[features_df["split"] == "test", "timestamp"].drop_duplicates().tolist()
    )

    n_total = cfg["n_queries"]
    n_direct = round(n_total * cfg.get("direct_fraction", 0.7))
    n_cross = n_total - n_direct

    raw_pairs: list[tuple[str, str]] = [rng.choice(endpoints) for _ in range(n_direct)]
    for _ in range(n_cross):
        u, _ = rng.choice(endpoints)
        _, v = rng.choice(endpoints)
        raw_pairs.append((u, v))
    rng.shuffle(raw_pairs)

    lower_bounds = [classical_lower_bound_heuristic(graph, d)(o) for o, d in raw_pairs]
    lb_series = pd.Series(lower_bounds)
    cut_low, cut_high = lb_series.quantile([1 / 3, 2 / 3])
    short_name, medium_name, long_name = cfg.get("distance_strata", ["short", "medium", "long"])

    def distance_stratum(lb: float) -> str:
        if lb <= cut_low:
            return short_name
        if lb <= cut_high:
            return medium_name
        return long_name

    time_strata = cfg["time_strata"]
    queries = []
    for i, ((origin, dest), lb) in enumerate(zip(raw_pairs, lower_bounds, strict=True)):
        depart_time = pd.Timestamp(rng.choice(test_timestamps)).to_pydatetime()
        queries.append(
            Query(
                query_id=i,
                origin=origin,
                dest=dest,
                depart_time=depart_time,
                distance_stratum=distance_stratum(lb),
                time_stratum=_time_stratum(depart_time, time_strata),
            )
        )
    return queries
