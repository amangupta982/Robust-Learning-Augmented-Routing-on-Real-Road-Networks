"""OraclePredictor: perfect hindsight travel time, from held-out ground truth.

Used for the consistency experiments (RQ1) -- when the "predictor" is
exactly right, prediction-guided routing should match true shortest paths.
This is the one predictor allowed to read the realized speed_mph for the
row's own bucket (not a lagged/shifted feature): that's the whole point of
an oracle baseline, and it is never used as a training signal.

Scope note (inherited from roar/graph/features.py): only the ~186
METR-LA-instrumented edges have a feature row, so eta() raises KeyError for
any other edge_id or for timestamps outside the dataset's date range.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from roar.graph.config import load_config
from roar.graph.features import FEATURES_PATH
from roar.predictor.base import TravelTimePredictor, floor_to_bucket, travel_time_seconds


class OraclePredictor(TravelTimePredictor):
    def __init__(self, features_df: pd.DataFrame, time_bucket_minutes: int = 5):
        self._time_bucket_minutes = time_bucket_minutes
        self._lookup = (
            features_df.set_index(["edge_id", "timestamp"])[["speed_mph", "length_m"]]
            .sort_index()
        )

    @classmethod
    def from_parquet(cls, path=FEATURES_PATH, cfg: dict | None = None) -> OraclePredictor:
        cfg = cfg or load_config()
        df = pd.read_parquet(path, columns=["edge_id", "timestamp", "speed_mph", "length_m"])
        return cls(df, time_bucket_minutes=cfg["time_bucket_minutes"])

    def _lookup_row(self, edge_id: str, depart_time: dt.datetime) -> pd.Series:
        bucket = floor_to_bucket(depart_time, self._time_bucket_minutes)
        try:
            return self._lookup.loc[(edge_id, pd.Timestamp(bucket))]
        except KeyError as exc:
            raise KeyError(
                f"No realized speed for edge_id={edge_id!r} at {bucket} -- edge is "
                "either uninstrumented or the timestamp is outside the dataset's "
                "range (see results/data_quality.md)."
            ) from exc

    def eta(self, edge_id: str, depart_time: dt.datetime) -> float:
        row = self._lookup_row(edge_id, depart_time)
        return float(travel_time_seconds(row["length_m"], row["speed_mph"]))

    def eta_with_confidence(self, edge_id: str, depart_time: dt.datetime) -> tuple[float, float]:
        return self.eta(edge_id, depart_time), 0.0
