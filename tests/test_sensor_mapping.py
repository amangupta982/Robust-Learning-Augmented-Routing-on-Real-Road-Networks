"""Sensor -> edge mapping produces a documented match rate.

Requires the real sensor/edge mapping (`make data` through map_sensors.py) —
skipped otherwise.
"""

import pandas as pd
import pytest
from roar.graph.load_graph import DATA_QUALITY_REPORT
from roar.graph.map_sensors import SENSOR_EDGE_MAP_PATH

pytestmark = pytest.mark.skipif(
    not SENSOR_EDGE_MAP_PATH.exists(),
    reason="sensor_edge_map.parquet not built yet; run `make data` first",
)


def test_match_rate_is_documented_and_reasonable():
    df = pd.read_parquet(SENSOR_EDGE_MAP_PATH)

    assert len(df) == 207, "METR-LA ships exactly 207 sensors"
    assert df["sensor_id"].is_unique

    match_rate = df["matched"].mean()
    # Real, non-fabricated matching against a real bbox-limited OSM extract
    # should still land the overwhelming majority of sensors on a nearby
    # drive edge; a much lower rate signals a bbox/CRS bug, not a real
    # property of the data.
    assert match_rate > 0.8, f"only {match_rate:.1%} of sensors matched an edge"

    report = DATA_QUALITY_REPORT.read_text()
    assert "Sensor -> edge mapping" in report
    assert "Matched" in report
    assert "Dropped" in report


def test_dropped_sensors_are_flagged_not_silently_kept():
    df = pd.read_parquet(SENSOR_EDGE_MAP_PATH)
    dropped = df[~df["matched"]]
    if len(dropped) > 0:
        assert (dropped["match_distance_m"] > 0).all()
