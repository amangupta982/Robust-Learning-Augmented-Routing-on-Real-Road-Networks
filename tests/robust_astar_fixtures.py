"""Shared real-data fixtures for the Phase 4 RobustAStar tests: the real LA
graph, real METR-LA-instrumented edges, and real test-split timestamps --
so these tests exercise the actual (origin, dest, depart_time) query shape
the router will see in the real experiment harness (Phase 5), not a
hand-wavy substitute. Not a test module itself (no test_ prefix).
"""

from __future__ import annotations

import datetime as dt
import random

import pandas as pd
from roar.graph.config import load_config
from roar.graph.features import FEATURES_PATH
from roar.graph.load_graph import EDGES_PATH, GRAPH_PATH
from roar.graph.map_sensors import SENSOR_EDGE_MAP_PATH
from roar.predictor.oracle import OraclePredictor
from roar.routing.graph import RoutingGraph, load_road_graph


def real_data_available() -> bool:
    return (
        GRAPH_PATH.exists()
        and EDGES_PATH.exists()
        and FEATURES_PATH.exists()
        and SENSOR_EDGE_MAP_PATH.exists()
    )


def load_real_graph() -> RoutingGraph:
    return load_road_graph()


def load_features_df() -> pd.DataFrame:
    return pd.read_parquet(
        FEATURES_PATH, columns=["edge_id", "timestamp", "speed_mph", "length_m", "split"]
    )


def load_oracle(features_df: pd.DataFrame) -> OraclePredictor:
    cfg = load_config()
    return OraclePredictor(features_df, time_bucket_minutes=cfg["time_bucket_minutes"])


def instrumented_edge_endpoints() -> list[tuple[str, str]]:
    """(u, v) node-id pairs for every METR-LA-instrumented, successfully
    matched edge -- used to build queries guaranteed to touch real sensor
    data (only ~186/90171 edges are instrumented; see
    results/data_quality.md, so random node pairs would rarely hit one)."""
    sensor_map = pd.read_parquet(SENSOR_EDGE_MAP_PATH)
    sensor_map = sensor_map[sensor_map["matched"]]
    return [
        (str(u), str(v))
        for u, v in zip(sensor_map["u"], sensor_map["v"], strict=False)
    ]


def sample_test_split_timestamps(
    features_df: pd.DataFrame, n: int, seed: int
) -> list[dt.datetime]:
    test_ts = features_df.loc[features_df["split"] == "test", "timestamp"].drop_duplicates()
    rng = random.Random(seed)
    chosen = rng.sample(list(test_ts), min(n, len(test_ts)))
    return [pd.Timestamp(t).to_pydatetime() for t in chosen]


def direct_instrumented_queries(
    features_df: pd.DataFrame, n: int, seed: int
) -> list[tuple[str, str, dt.datetime]]:
    """Origin/dest are the (u, v) endpoints of the SAME random instrumented
    edge -- guarantees the query directly involves real sensor data."""
    endpoints = instrumented_edge_endpoints()
    rng = random.Random(seed)
    timestamps = sample_test_split_timestamps(features_df, n, seed)
    return [
        (*rng.choice(endpoints), timestamps[i % len(timestamps)]) for i in range(n)
    ]


def cross_network_queries(
    features_df: pd.DataFrame, n: int, seed: int
) -> list[tuple[str, str, dt.datetime]]:
    """Origin/dest are endpoints of two DIFFERENT random instrumented
    edges -- forces a genuine multi-hop route that may mix instrumented and
    uninstrumented edges, for longer-distance coverage."""
    endpoints = instrumented_edge_endpoints()
    rng = random.Random(seed + 1)
    timestamps = sample_test_split_timestamps(features_df, n, seed + 1)
    queries = []
    for i in range(n):
        u, _ = rng.choice(endpoints)
        _, v = rng.choice(endpoints)
        queries.append((u, v, timestamps[i % len(timestamps)]))
    return queries
