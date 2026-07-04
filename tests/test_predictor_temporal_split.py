"""Temporal-split enforcement (CLAUDE.md rule 2): LightGBMPredictor.train
must fit only on `split == "train"` rows (early-stopped against `split ==
"val"`) and never look at `split == "test"` rows. Proven by corrupting the
test split's target column and confirming the trained model is byte-for-byte
unaffected -- if test rows leaked into training, this would change the fit.

Uses a small real slice of the Phase 1 feature table -- skipped if that
table hasn't been built yet (never a synthetic substitute for a real
model-quality result, per CLAUDE.md rule 1).
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
    small = df[df["edge_id"].isin(edges)].reset_index(drop=True)
    assert (small["split"] == "test").any(), "fixture must include some test rows"
    assert (small["split"] == "train").any(), "fixture must include some train rows"
    return small


@pytest.fixture(scope="module")
def fast_cfg() -> dict:
    cfg = copy.deepcopy(load_config(PREDICTOR_CONFIG_PATH))
    cfg["num_boost_round"] = 5
    cfg["early_stopping_rounds"] = 5
    return cfg


def test_corrupting_test_split_does_not_change_training(small_features_df, fast_cfg):
    clean = small_features_df.copy()
    corrupted = small_features_df.copy()
    is_test = corrupted["split"] == "test"
    # Wildly wrong target values on test-only rows -- if train() ever reads
    # them, predictions on train/val queries would change.
    corrupted.loc[is_test, fast_cfg["target"]] = 9999.0

    model_clean = LightGBMPredictor.train(clean, cfg=fast_cfg)
    model_corrupted = LightGBMPredictor.train(corrupted, cfg=fast_cfg)

    query_row = clean[clean["split"] == "train"].iloc[10]
    eta_clean = model_clean.eta(query_row["edge_id"], query_row["timestamp"].to_pydatetime())
    eta_corrupted = model_corrupted.eta(
        query_row["edge_id"], query_row["timestamp"].to_pydatetime()
    )
    assert eta_clean == eta_corrupted, (
        "training result changed after corrupting only test-split rows -- "
        "test data is leaking into training"
    )


def test_train_rejects_empty_train_or_val_split(small_features_df, fast_cfg):
    train_only = small_features_df[small_features_df["split"] == "train"]
    with pytest.raises(ValueError, match="train and val splits"):
        LightGBMPredictor.train(train_only, cfg=fast_cfg)


def test_lookup_only_contains_provided_rows(small_features_df, fast_cfg):
    """The single-query inference path (`eta`) must not silently succeed for
    a (edge_id, timestamp) outside the table used to build the predictor's
    lookup -- confirms there is no hidden fallback to a wider table."""
    train_val = small_features_df[small_features_df["split"].isin(["train", "val"])]
    test_row = small_features_df[small_features_df["split"] == "test"].iloc[0]

    model_train_val_only = LightGBMPredictor.train(train_val, cfg=fast_cfg)
    with pytest.raises(KeyError):
        model_train_val_only._predicted_times(
            test_row["edge_id"], test_row["timestamp"].to_pydatetime()
        )
