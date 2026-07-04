"""Unit-tests the leakage-prevention math with tiny synthetic fixtures.

CLAUDE.md rule 1 makes exactly one exception: synthetic inputs are fine for
unit-testing statistics/math code, never for producing reported results.
That's all this file does -- it never touches real METR-LA data and no
number here is ever reported as a result.
"""

import pandas as pd
from roar.graph.features import compute_neighbor_congestion, historical_speed_stats


def test_historical_speed_stats_uses_only_strictly_prior_rows():
    timestamps = pd.date_range("2012-03-01", periods=4, freq="5min")
    df = pd.DataFrame(
        {
            "edge_id": ["e1"] * 4 + ["e2"] * 4,
            "timestamp": list(timestamps) * 2,
            "speed_mph": [60.0, 50.0, 40.0, 30.0, 10.0, 20.0, 30.0, 40.0],
        }
    ).sort_values(["edge_id", "timestamp"]).reset_index(drop=True)

    hist_mean, hist_var = historical_speed_stats(df)

    # Row 0 of each edge has no prior rows at all -> NaN, not a leaked value.
    assert hist_mean.iloc[0] != hist_mean.iloc[0]  # NaN
    # Row 1 of e1: only row 0 (60.0) precedes it.
    assert hist_mean.iloc[1] == 60.0
    # Row 2 of e1: mean of rows 0-1 (60.0, 50.0), never row 2's own 40.0.
    assert hist_mean.iloc[2] == 55.0
    # Row 3 of e1: mean of rows 0-2 (60, 50, 40) = 50.0, excluding its own 30.0.
    assert hist_mean.iloc[3] == 50.0

    # Manually recompute for e2 the same way and confirm equality -- this is
    # the "recompute independently and compare" check the leakage test needs.
    expected_e2_mean = [float("nan"), 10.0, 15.0, 20.0]
    got_e2_mean = hist_mean.iloc[4:8].reset_index(drop=True)
    for expected, got in zip(expected_e2_mean, got_e2_mean, strict=True):
        if expected != expected:  # NaN
            assert got != got
        else:
            assert got == expected

    # var: row 2 is the first with >=2 prior points (rows 0-1: 60, 50).
    assert hist_var.iloc[2] == pd.Series([60.0, 50.0]).var()


def test_neighbor_congestion_only_uses_prior_bucket():
    timestamps = pd.date_range("2012-03-01", periods=3, freq="5min")
    speeds_long = pd.DataFrame(
        {
            "timestamp": list(timestamps) * 2,
            "sensor_id": ["A"] * 3 + ["B"] * 3,
            "speed_mph": [50.0, 40.0, 30.0, 20.0, 10.0, 0.0],
        }
    )
    speed_limit = pd.Series({"A": 50.0, "B": 50.0})
    neighbors = {"A": ["B"], "B": ["A"]}

    proxy = compute_neighbor_congestion(speeds_long, speed_limit, neighbors)
    proxy_a = (
        proxy[proxy["sensor_id"] == "A"]
        .sort_values("timestamp")["neighbor_congestion_proxy"]
        .tolist()
    )

    # A's proxy at t must equal B's congestion at t-1, i.e. 1 - speed/limit
    # computed from B's PRIOR row -- never B's (or A's own) current row.
    assert proxy_a[0] != proxy_a[0]  # no prior bucket at all -> NaN
    assert proxy_a[1] == 1 - (20.0 / 50.0)  # B's row 0 (t-1 relative to row 1)
    assert proxy_a[2] == 1 - (10.0 / 50.0)  # B's row 1 (t-1 relative to row 2)


def test_neighbor_congestion_is_nan_when_no_neighbors_in_range():
    timestamps = pd.date_range("2012-03-01", periods=2, freq="5min")
    speeds_long = pd.DataFrame(
        {"timestamp": timestamps, "sensor_id": ["A", "A"], "speed_mph": [50.0, 40.0]}
    )
    speed_limit = pd.Series({"A": 50.0})
    proxy = compute_neighbor_congestion(speeds_long, speed_limit, neighbors={"A": []})
    assert proxy["neighbor_congestion_proxy"].isna().all()
