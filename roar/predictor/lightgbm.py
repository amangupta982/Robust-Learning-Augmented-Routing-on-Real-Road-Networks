"""LightGBMPredictor: the honest, pluggable baseline travel-time predictor
(CLAUDE.md: LightGBM first, no GNN yet -- a stronger oracle can be added
later as an ablation, not a Phase 2 requirement).

Predicts `speed_mph` (not travel time directly) at several quantile levels
via LightGBM's native quantile objective, then converts to travel time
deterministically via roar.predictor.base.travel_time_seconds. The median
quantile is the point estimate `eta`; the outer quantile pair (configured as
`confidence_interval`) is inverted into a travel-time sigma via the
standard-normal quantile spacing -- a real, per-edge uncertainty instead of
a constant.

Every input feature is either a strictly-lagged statistic (`hist_mean_speed`,
`hist_var_speed`, `neighbor_congestion_proxy` -- all `.shift(1)`'d in
roar/graph/features.py) or a static edge/calendar attribute (`length_m`,
`lanes`, `speed_limit_mph`, `road_class`, `hour_of_day`, `day_of_week`,
`is_holiday`). The realized `speed_mph` for the row's own bucket is the
prediction target, never a feature -- using it as an input would leak the
answer.

Scope: eta()/eta_with_confidence() look up the precomputed feature row for
(edge_id, depart_time) from the same table used at training time and run it
through the trained boosters -- exactly mirroring how OraclePredictor looks
up the realized value. This is an offline-evaluation-scope predictor over
the Phase 1 feature table's fixed time grid, not a live feature-computation
service.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

from roar.graph.config import REPO_ROOT, load_config
from roar.predictor.base import TravelTimePredictor, floor_to_bucket, travel_time_seconds

PREDICTOR_CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "predictor.yaml"


def _feature_columns(cfg: dict) -> list[str]:
    fc = cfg["feature_columns"]
    return [*fc["numeric"], *fc["categorical"], *fc["boolean"]]


def _prepare_X(df: pd.DataFrame, cfg: dict, categories: dict[str, list]) -> pd.DataFrame:
    fc = cfg["feature_columns"]
    X = df[_feature_columns(cfg)].copy()
    for col in fc["numeric"]:
        X[col] = X[col].astype(float)
    for col in fc["categorical"]:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    for col in fc["boolean"]:
        X[col] = X[col].astype(int)
    return X


class LightGBMPredictor(TravelTimePredictor):
    def __init__(
        self,
        boosters: dict[float, lgb.Booster],
        categories: dict[str, list],
        features_lookup: pd.DataFrame,
        cfg: dict,
    ):
        self._boosters = boosters
        self._categories = categories
        self._lookup = features_lookup
        self._cfg = cfg
        lo_q, hi_q = cfg["confidence_interval"]
        self._lo_q, self._hi_q = lo_q, hi_q
        self._med_q = cfg["median_quantile"]
        self._z_span = norm.ppf(hi_q) - norm.ppf(lo_q)

    # ---- training ----

    @classmethod
    def train(cls, features_df: pd.DataFrame, cfg: dict | None = None) -> LightGBMPredictor:
        cfg = cfg or load_config(PREDICTOR_CONFIG_PATH)
        fc = cfg["feature_columns"]
        target = cfg["target"]

        train_df = features_df[features_df["split"] == "train"]
        val_df = features_df[features_df["split"] == "val"]
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(
                "LightGBMPredictor.train requires non-empty train and val splits "
                "(temporal split from experiments/configs/data.yaml) -- got "
                f"{len(train_df)} train rows, {len(val_df)} val rows."
            )

        # Category labels are a static edge/road attribute (known ahead of
        # time, like the rest of the graph structure), not a statistic
        # learned from data -- so it is fine to enumerate them from the full
        # table rather than train-only, and doing so avoids "unseen
        # category" errors at test/inference time.
        categories = {
            col: sorted(features_df[col].dropna().unique().tolist())
            for col in fc["categorical"]
        }

        X_train = _prepare_X(train_df, cfg, categories)
        X_val = _prepare_X(val_df, cfg, categories)
        y_train = train_df[target].astype(float)
        y_val = val_df[target].astype(float)

        lgb_train = lgb.Dataset(
            X_train, label=y_train, categorical_feature=fc["categorical"], free_raw_data=False
        )
        lgb_val = lgb.Dataset(
            X_val, label=y_val, reference=lgb_train, categorical_feature=fc["categorical"]
        )

        params_base = dict(cfg["lightgbm_params"])
        params_base["seed"] = cfg["seed"]
        n_rounds = cfg["num_boost_round"]

        boosters = {}
        for q in cfg["quantiles"]:
            params = {**params_base, "alpha": q}
            boosters[q] = lgb.train(
                params,
                lgb_train,
                num_boost_round=n_rounds,
                valid_sets=[lgb_val],
                callbacks=[
                    lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

        lookup_cols = _feature_columns(cfg)
        lookup = (
            features_df.set_index(["edge_id", "timestamp"])[lookup_cols].sort_index()
        )
        return cls(boosters, categories, lookup, cfg)

    # ---- persistence ----

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for q, booster in self._boosters.items():
            booster.save_model(str(model_dir / f"quantile_{q}.txt"))
        meta = {"categories": self._categories, "cfg": self._cfg}
        (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, model_dir: Path, features_lookup_df: pd.DataFrame) -> LightGBMPredictor:
        """`features_lookup_df` must contain edge_id, timestamp, and every
        configured feature column -- normally the same Phase 1 feature table
        used at training time (or a subset of its rows)."""
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "metadata.json").read_text())
        cfg = meta["cfg"]
        categories = meta["categories"]
        boosters = {
            q: lgb.Booster(model_file=str(model_dir / f"quantile_{q}.txt"))
            for q in cfg["quantiles"]
        }
        lookup_cols = _feature_columns(cfg)
        lookup = (
            features_lookup_df.set_index(["edge_id", "timestamp"])[lookup_cols].sort_index()
        )
        return cls(boosters, categories, lookup, cfg)

    # ---- batch inference / evaluation ----

    def predict_speed_quantiles(self, df: pd.DataFrame) -> dict[float, np.ndarray]:
        X = _prepare_X(df, self._cfg, self._categories)
        floor = self._cfg.get("min_speed_mph", 1.0)
        return {
            q: np.clip(booster.predict(X), floor, None) for q, booster in self._boosters.items()
        }

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Honest quality report on a held-out slice (normally the temporal
        test split): MAE/RMSE/MAPE of the median-quantile eta (travel-time
        space, the operational unit), plus quantile calibration.

        Calibration is measured in SPEED space (empirical coverage of
        speed_true <= speed_pred[q] vs. nominal q) rather than travel-time
        space: travel_time = length_m / speed is a strictly decreasing
        function of speed, so a time-space quantile q would actually
        correspond to the (1-q) speed quantile -- comparing it to nominal q
        would silently mislabel a correctly-calibrated model as inverted.
        Speed is the space LightGBM actually regressed, so that's where
        "does quantile q cover fraction q of outcomes" is a meaningful
        question."""
        target = self._cfg["target"]
        speed_true = df[target].to_numpy(dtype=float)
        length_m = df["length_m"].to_numpy(dtype=float)
        time_true = travel_time_seconds(length_m, speed_true)

        speed_pred = self.predict_speed_quantiles(df)
        time_pred = {q: travel_time_seconds(length_m, s) for q, s in speed_pred.items()}

        eta_pred = time_pred[self._med_q]
        err = eta_pred - time_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        mape = float(np.mean(np.abs(err) / np.maximum(time_true, 1e-6)) * 100)

        calibration = [
            (q, float(np.mean(speed_true <= speed_pred[q]))) for q in self._cfg["quantiles"]
        ]
        return {
            "mae_s": mae,
            "rmse_s": rmse,
            "mape_pct": mape,
            "n": len(df),
            "calibration": calibration,
        }

    # ---- single-query inference (TravelTimePredictor interface) ----

    def _predicted_times(self, edge_id: str, depart_time: dt.datetime) -> dict[float, float]:
        bucket = floor_to_bucket(depart_time, self._cfg.get("time_bucket_minutes", 5))
        try:
            row = self._lookup.loc[(edge_id, pd.Timestamp(bucket))]
        except KeyError as exc:
            raise KeyError(
                f"No feature row for edge_id={edge_id!r} at {bucket} -- edge is "
                "either uninstrumented or the timestamp is outside the dataset's "
                "range (see results/data_quality.md)."
            ) from exc
        row_df = row.to_frame().T
        speed_pred = self.predict_speed_quantiles(row_df)
        length_m = float(row["length_m"])
        return {q: float(travel_time_seconds(length_m, s[0])) for q, s in speed_pred.items()}

    def eta(self, edge_id: str, depart_time: dt.datetime) -> float:
        return self._predicted_times(edge_id, depart_time)[self._med_q]

    def eta_with_confidence(self, edge_id: str, depart_time: dt.datetime) -> tuple[float, float]:
        times = self._predicted_times(edge_id, depart_time)
        eta = times[self._med_q]
        t_hi = times[self._lo_q]  # low-speed quantile -> long travel time
        t_lo = times[self._hi_q]  # high-speed quantile -> short travel time
        sigma = max(t_hi - t_lo, 0.0) / self._z_span
        return eta, sigma
