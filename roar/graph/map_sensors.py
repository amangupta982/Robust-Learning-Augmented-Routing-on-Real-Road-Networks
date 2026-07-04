"""Map the 207 real METR-LA sensors to their nearest OSM drive-network edge.

This is the error-prone step CLAUDE.md warns about: a nearest-edge match can
land far from the sensor's true location (bbox edge effects, OSM geometry
gaps, sensors on ramps not covered by `network_type=drive`, etc). Every
match distance is logged; matches farther than
`sensor_matching.max_match_distance_m` (experiments/configs/data.yaml) are
flagged and dropped rather than silently kept.

Run directly: `python -m roar.graph.map_sensors` (requires
`data/processed/edges.parquet` from load_graph.py to already exist).
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from roar.graph.config import REPO_ROOT, load_config
from roar.graph.download import fetch
from roar.graph.load_graph import DATA_QUALITY_REPORT, EDGES_PATH

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SENSOR_EDGE_MAP_PATH = PROCESSED_DIR / "sensor_edge_map.parquet"


def load_sensor_locations(cfg: dict) -> pd.DataFrame:
    raw_dir = REPO_ROOT / cfg["metr_la"]["raw_dir"]
    path = fetch(cfg["metr_la"]["sensor_locations_url"], raw_dir / "graph_sensor_locations.csv")
    df = pd.read_csv(path, dtype={"sensor_id": str})
    return df[["sensor_id", "latitude", "longitude"]]


def download_sensor_distances(cfg: dict):
    """Cache the real inter-sensor road-network distance matrix for later use
    by features.py (congestion proxy). Downloaded here since this module owns
    the "fetch real METR-LA sensor metadata" responsibility."""
    raw_dir = REPO_ROOT / cfg["metr_la"]["raw_dir"]
    fetch(cfg["metr_la"]["sensor_distances_url"], raw_dir / "distances_la_2012.csv")


def match_sensors_to_edges(sensors: pd.DataFrame, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    utm_crs = edges.estimate_utm_crs()
    edges_utm = edges.to_crs(utm_crs)

    points = [
        Point(lon, lat)
        for lon, lat in zip(sensors["longitude"], sensors["latitude"], strict=True)
    ]
    sensor_points = gpd.GeoDataFrame(
        sensors.copy(), geometry=points, crs="EPSG:4326"
    ).to_crs(utm_crs)

    matched = gpd.sjoin_nearest(
        sensor_points,
        edges_utm[["edge_id", "u", "v", "key", "geometry"]],
        distance_col="match_distance_m",
        how="left",
    )
    # sjoin_nearest can return >1 row per sensor on exact ties; keep the closest.
    matched = matched.sort_values("match_distance_m").drop_duplicates("sensor_id", keep="first")
    return matched.drop(columns=["index_right"], errors="ignore")


def main() -> None:
    cfg = load_config()
    if not EDGES_PATH.exists():
        raise FileNotFoundError(
            f"{EDGES_PATH} not found. Run `python -m roar.graph.load_graph` first."
        )

    sensors = load_sensor_locations(cfg)
    download_sensor_distances(cfg)
    edges = gpd.read_parquet(EDGES_PATH)

    matched = match_sensors_to_edges(sensors, edges)

    max_dist = cfg["sensor_matching"]["max_match_distance_m"]
    matched["matched"] = matched["match_distance_m"] <= max_dist

    out = matched[["sensor_id", "latitude", "longitude", "edge_id", "u", "v", "key",
                    "match_distance_m", "matched"]]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(SENSOR_EDGE_MAP_PATH)

    n_total = len(out)
    n_matched = int(out["matched"].sum())
    n_dropped = n_total - n_matched
    dist = out["match_distance_m"]

    with open(DATA_QUALITY_REPORT, "a") as f:
        f.write("\n## Sensor -> edge mapping (METR-LA)\n\n")
        f.write(f"- Total sensors: {n_total}\n")
        f.write(f"- Matched (<= {max_dist} m): {n_matched} ({n_matched / n_total:.1%})\n")
        f.write(f"- Dropped (> {max_dist} m): {n_dropped} ({n_dropped / n_total:.1%})\n")
        f.write("- Match distance (m): "
                f"mean={dist.mean():.1f}, median={dist.median():.1f}, "
                f"p90={dist.quantile(0.9):.1f}, max={dist.max():.1f}\n")

    print(f"Matched {n_matched}/{n_total} sensors (<= {max_dist} m); dropped {n_dropped}.")
    print(f"Saved sensor->edge map to {SENSOR_EDGE_MAP_PATH}")


if __name__ == "__main__":
    main()
