"""Build the leakage-safe, per-edge, per-5-minute-bucket feature table.

Storage choice — parquet-only, no SQLite: every consumer of this table
(predictor training in Phase 2, the experiment harness in Phase 5) reads the
whole table at once for vectorized/columnar processing; there is exactly one
writer (this script) and no concurrent-write use case that would justify
SQLite's transactional guarantees. Parquet also round-trips pandas dtypes
(including the boolean `is_holiday`/`matched` columns) without a schema
translation layer. If a future phase needs indexed point lookups by
(edge_id, timestamp), that's the trigger to add SQLite — not before.

Leakage rule (CLAUDE.md rule 2, generalized to features): every feature at
row (edge_id, t) is computed from strictly-prior timestamps only
(`expanding().shift(1)` / a one-bucket `.shift(1)` on a uniform 5-min grid).
`tests/test_features_leakage.py` recomputes several features by hand on a
small fixture and asserts equality.

Scope note: only the METR-LA-instrumented edges (~207 of them) get a feature
row. Building features for the rest of the LA drive network would require
inventing speeds for uninstrumented edges, which CLAUDE.md rule 1 forbids —
so the routing/predictor phases must treat this as the training/evaluation
scope, not the full graph (see results/data_quality.md and LIMITATIONS.md
in a later phase).

Run directly: `python -m roar.graph.features` (requires
`data/processed/sensor_edge_map.parquet` from map_sensors.py).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from roar.graph.config import REPO_ROOT, load_config
from roar.graph.download import fetch
from roar.graph.load_graph import DATA_QUALITY_REPORT, EDGES_PATH
from roar.graph.map_sensors import SENSOR_EDGE_MAP_PATH

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"


def load_speeds_long(cfg: dict, sensor_ids: list[str]) -> pd.DataFrame:
    raw_dir = REPO_ROOT / cfg["metr_la"]["raw_dir"]
    path = fetch(cfg["metr_la"]["speeds_url"], raw_dir / "vel_metr_la.csv")
    wide = pd.read_csv(path, index_col=0, parse_dates=True)
    wide.columns = wide.columns.astype(str)
    wide = wide[[c for c in wide.columns if c in sensor_ids]]
    wide.index.name = "timestamp"
    long = wide.reset_index().melt(
        id_vars="timestamp", var_name="sensor_id", value_name="speed_mph"
    )
    return long.dropna(subset=["speed_mph"])


def load_neighbor_map(cfg: dict, sensor_ids: set[str]) -> dict[str, list[str]]:
    raw_dir = REPO_ROOT / cfg["metr_la"]["raw_dir"]
    path = raw_dir / "distances_la_2012.csv"  # fetched by map_sensors.py
    dist = pd.read_csv(path, dtype={"from": str, "to": str})
    dist = dist[
        dist["from"].isin(sensor_ids) & dist["to"].isin(sensor_ids) & (dist["from"] != dist["to"])
    ]
    radius = cfg["congestion_proxy"]["neighbor_radius_m"]
    max_neighbors = cfg["congestion_proxy"]["max_neighbors"]
    dist = dist[dist["cost"] <= radius].sort_values("cost")

    neighbors: dict[str, list[str]] = {sid: [] for sid in sensor_ids}
    for sensor_id, group in dist.groupby("from"):
        neighbors[sensor_id] = group["to"].head(max_neighbors).tolist()
    return neighbors


def compute_neighbor_congestion(
    speeds_long: pd.DataFrame, sensor_speed_limit: pd.Series, neighbors: dict[str, list[str]]
) -> pd.DataFrame:
    wide_speed = speeds_long.pivot(index="timestamp", columns="sensor_id", values="speed_mph")
    congestion = 1 - wide_speed.divide(sensor_speed_limit, axis=1)
    # Uniform 5-min grid -> shifting one row = strictly the prior bucket (t-1),
    # never the current or a future bucket.
    prior_congestion = congestion.shift(1)

    proxy_cols = {}
    for sensor_id in wide_speed.columns:
        neigh = [n for n in neighbors.get(sensor_id, []) if n in prior_congestion.columns]
        if neigh:
            proxy_cols[sensor_id] = prior_congestion[neigh].mean(axis=1)
        else:
            proxy_cols[sensor_id] = pd.Series(index=prior_congestion.index, dtype=float)
    proxy_wide = pd.DataFrame(proxy_cols)
    proxy_long = proxy_wide.reset_index().melt(
        id_vars="timestamp", var_name="sensor_id", value_name="neighbor_congestion_proxy"
    )
    return proxy_long


def historical_speed_stats(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per-edge expanding mean/var of speed_mph, shifted by one row.

    `df` must already be sorted by (edge_id, timestamp). The `.shift(1)`
    after `.expanding()` is what makes this leakage-safe: the value at row i
    is a function of rows [0, i-1] only, never row i itself.
    """
    grouped = df.groupby("edge_id")["speed_mph"]
    hist_mean = grouped.transform(lambda s: s.expanding().mean().shift(1))
    hist_var = grouped.transform(lambda s: s.expanding().var().shift(1))
    return hist_mean, hist_var


def assign_split(timestamp: pd.Series, split_cfg: dict) -> pd.Series:
    date = timestamp.dt.date
    labels = pd.Series("unassigned", index=timestamp.index)
    for name in ("train", "val", "test"):
        start = dt.date.fromisoformat(split_cfg[name]["start"])
        end = dt.date.fromisoformat(split_cfg[name]["end"])
        labels[(date >= start) & (date <= end)] = name
    return labels


def collapse_to_edge_level(df: pd.DataFrame) -> pd.DataFrame:
    """A handful of edges (see results/data_quality.md) have more than one
    METR-LA sensor snap to them -- e.g. sensors on separate lanes of the same
    simplified OSM way. Routing needs exactly one cost per (edge, timestamp),
    so co-located sensors' real readings are averaged rather than treated as
    a single sensor's sequential history (which would silently interleave
    two different sensors' measurements into one "edge" time series)."""
    agg = df.groupby(["edge_id", "timestamp"]).agg(
        speed_mph=("speed_mph", "mean"),
        neighbor_congestion_proxy=("neighbor_congestion_proxy", "mean"),
        sensor_ids=("sensor_id", lambda s: ";".join(sorted(set(s)))),
        n_sensors=("sensor_id", "nunique"),
        road_class=("road_class", "first"),
        length_m=("length_m", "first"),
        lanes=("lanes", "first"),
        speed_limit_mph=("speed_limit_mph", "first"),
    )
    return agg.reset_index()


def build_features(cfg: dict) -> pd.DataFrame:
    sensor_edge_map = pd.read_parquet(SENSOR_EDGE_MAP_PATH)
    sensor_edge_map = sensor_edge_map[sensor_edge_map["matched"]].copy()
    sensor_ids = sensor_edge_map["sensor_id"].tolist()

    static_cols = ["edge_id", "road_class", "length_m", "lanes", "speed_limit_mph"]
    edges = pd.read_parquet(EDGES_PATH, columns=static_cols)
    sensor_edge_map = sensor_edge_map.merge(edges, on="edge_id", how="left")
    speed_limit_by_sensor = sensor_edge_map.set_index("sensor_id")["speed_limit_mph"]

    speeds_long = load_speeds_long(cfg, sensor_ids)
    neighbors = load_neighbor_map(cfg, set(sensor_ids))
    proxy_long = compute_neighbor_congestion(speeds_long, speed_limit_by_sensor, neighbors)

    df = speeds_long.merge(
        sensor_edge_map[["sensor_id", *static_cols[1:], "edge_id"]], on="sensor_id", how="left"
    )
    df = df.merge(proxy_long, on=["timestamp", "sensor_id"], how="left")

    df = collapse_to_edge_level(df)
    df = df.sort_values(["edge_id", "timestamp"]).reset_index(drop=True)
    df["hist_mean_speed"], df["hist_var_speed"] = historical_speed_stats(df)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    holidays = {dt.date.fromisoformat(d) for d in cfg["holidays"]}
    df["is_holiday"] = df["timestamp"].dt.date.isin(holidays)

    df["split"] = assign_split(df["timestamp"], cfg["split"])
    assert (df["split"] != "unassigned").all(), (
        "every timestamp must fall inside a configured split"
    )

    return df[[
        "edge_id", "sensor_ids", "n_sensors", "timestamp", "speed_mph",
        "hist_mean_speed", "hist_var_speed",
        "hour_of_day", "day_of_week", "is_holiday",
        "neighbor_congestion_proxy",
        "road_class", "length_m", "lanes", "speed_limit_mph",
        "split",
    ]]


def main() -> None:
    cfg = load_config()
    if not SENSOR_EDGE_MAP_PATH.exists():
        raise FileNotFoundError(
            f"{SENSOR_EDGE_MAP_PATH} not found. Run `python -m roar.graph.map_sensors` first."
        )

    df = build_features(cfg)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH)

    split_counts = df["split"].value_counts()
    with open(DATA_QUALITY_REPORT, "a") as f:
        f.write("\n## Feature table\n\n")
        f.write(f"- Rows: {len(df)}\n")
        f.write(f"- Distinct edges (instrumented only): {df['edge_id'].nunique()}\n")
        n_multi_sensor_edges = int((df.groupby("edge_id")["n_sensors"].first() > 1).sum())
        f.write(
            f"- Edges with >1 co-located sensor (readings averaged, see "
            f"collapse_to_edge_level): {n_multi_sensor_edges}\n"
        )
        f.write(
            f"- Time range: {df['timestamp'].min()} to {df['timestamp'].max()} "
            f"({cfg['time_bucket_minutes']}-minute buckets)\n"
        )
        f.write("- Split row counts (whole-day, chronological, never random):\n")
        for name in ("train", "val", "test"):
            cfg_range = cfg["split"][name]
            f.write(
                f"  - {name}: {split_counts.get(name, 0)} rows "
                f"({cfg_range['start']} to {cfg_range['end']})\n"
            )
        n_missing_proxy = int(df["neighbor_congestion_proxy"].isna().sum())
        f.write(
            f"- Rows with no in-radius neighbor sensor (neighbor_congestion_proxy is NaN): "
            f"{n_missing_proxy} ({n_missing_proxy / len(df):.1%})\n"
        )

    print(f"Saved {len(df)} feature rows to {FEATURES_PATH}")
    print(f"Split counts: {split_counts.to_dict()}")


if __name__ == "__main__":
    main()
