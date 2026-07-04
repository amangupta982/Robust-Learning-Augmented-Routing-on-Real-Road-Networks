"""Serialized model loads and predicts: LightGBMPredictor.save()/.load()
round-trip must reproduce bit-identical predictions -- `make train`'s
versioned artifact in results/models/ is only useful if reloading it gives
back the exact model that produced the reported numbers.

Uses a small real slice of the Phase 1 feature table -- skipped if that
table hasn't been built yet.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest
from roar.graph.config import load_config
from roar.graph.features import FEATURES_PATH
from roar.predictor.lightgbm import PREDICTOR_CONFIG_PATH, LightGBMPredictor

pytestmark = pytest.mark.skipif(
    not FEATURES_PATH.exists(),
    reason="features.parquet not built yet; run `make data` first",
)


@pytest.fixture(scope="module")
def small_features_df() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    edges = df["edge_id"].unique()[:2]
    return df[df["edge_id"].isin(edges)].reset_index(drop=True)


@pytest.fixture(scope="module")
def fast_cfg() -> dict:
    cfg = copy.deepcopy(load_config(PREDICTOR_CONFIG_PATH))
    cfg["num_boost_round"] = 5
    cfg["early_stopping_rounds"] = 5
    return cfg


def test_save_then_load_reproduces_identical_predictions(small_features_df, fast_cfg, tmp_path):
    model = LightGBMPredictor.train(small_features_df, cfg=fast_cfg)
    model.save(tmp_path / "model_v1")

    reloaded = LightGBMPredictor.load(tmp_path / "model_v1", small_features_df)

    test_df = small_features_df[small_features_df["split"] == "test"]
    original_metrics = model.evaluate(test_df)
    reloaded_metrics = reloaded.evaluate(test_df)
    assert original_metrics == reloaded_metrics

    query_row = test_df.iloc[0]
    edge_id, depart_time = query_row["edge_id"], query_row["timestamp"].to_pydatetime()
    assert model.eta(edge_id, depart_time) == reloaded.eta(edge_id, depart_time)
    assert model.eta_with_confidence(edge_id, depart_time) == reloaded.eta_with_confidence(
        edge_id, depart_time
    )


def test_saved_artifact_has_one_file_per_quantile_plus_metadata(
    small_features_df, fast_cfg, tmp_path
):
    model = LightGBMPredictor.train(small_features_df, cfg=fast_cfg)
    model_dir = tmp_path / "model_v2"
    model.save(model_dir)

    saved_files = {p.name for p in model_dir.iterdir()}
    assert "metadata.json" in saved_files
    for q in fast_cfg["quantiles"]:
        assert f"quantile_{q}.txt" in saved_files


def test_loaded_model_works_end_to_end_on_a_fresh_predictor_instance(
    small_features_df, fast_cfg, tmp_path
):
    """Simulates the real deployment path: train once, persist, and later
    construct a brand-new LightGBMPredictor purely from disk + a features
    table, with no reference to the original training-time object."""
    model = LightGBMPredictor.train(small_features_df, cfg=fast_cfg)
    model_dir = tmp_path / "model_v3"
    model.save(model_dir)
    del model

    reloaded = LightGBMPredictor.load(model_dir, small_features_df)
    test_row = small_features_df[small_features_df["split"] == "test"].iloc[5]
    eta, sigma = reloaded.eta_with_confidence(
        test_row["edge_id"], test_row["timestamp"].to_pydatetime()
    )
    assert eta > 0
    assert sigma >= 0
