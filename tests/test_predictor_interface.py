"""Interface conformance: every concrete predictor must satisfy the
TravelTimePredictor contract (eta -> float, eta_with_confidence -> (float,
sigma>=0)) so routing code can depend on the interface alone (CLAUDE.md
rule 5).

Uses a small real slice of the Phase 1 feature table (never synthetic --
these are reported-result predictors, not pure math helpers) -- skipped if
that table hasn't been built yet.
"""

from __future__ import annotations

import copy
import datetime as dt

import pandas as pd
import pytest
from roar.graph.features import FEATURES_PATH
from roar.predictor.adversarial import AdversarialPredictor
from roar.predictor.base import TravelTimePredictor
from roar.predictor.lightgbm import PREDICTOR_CONFIG_PATH, LightGBMPredictor
from roar.predictor.noisy import NoisyPredictor
from roar.predictor.oracle import OraclePredictor

pytestmark = pytest.mark.skipif(
    not FEATURES_PATH.exists(),
    reason="features.parquet not built yet; run `make data` first",
)


@pytest.fixture(scope="module")
def small_features_df() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    # Two instrumented edges is enough to exercise the interface without
    # paying for a full-scale LightGBM fit.
    edges = df["edge_id"].unique()[:2]
    return df[df["edge_id"].isin(edges)].reset_index(drop=True)


@pytest.fixture(scope="module")
def sample_query(small_features_df: pd.DataFrame) -> tuple[str, dt.datetime]:
    test_row = small_features_df[small_features_df["split"] == "test"].iloc[0]
    return test_row["edge_id"], test_row["timestamp"].to_pydatetime()


@pytest.fixture(scope="module")
def lightgbm_predictor(small_features_df: pd.DataFrame) -> LightGBMPredictor:
    from roar.graph.config import load_config

    cfg = copy.deepcopy(load_config(PREDICTOR_CONFIG_PATH))
    cfg["num_boost_round"] = 5
    cfg["early_stopping_rounds"] = 5
    return LightGBMPredictor.train(small_features_df, cfg=cfg)


def all_predictors(small_features_df, lightgbm_predictor):
    oracle = OraclePredictor(small_features_df)
    return {
        "oracle": oracle,
        "noisy": NoisyPredictor(oracle, sigma_level=0.2),
        "adversarial": AdversarialPredictor(oracle, budget=0.3),
        "lightgbm": lightgbm_predictor,
    }


def test_all_predictors_are_travel_time_predictors(small_features_df, lightgbm_predictor):
    for name, predictor in all_predictors(small_features_df, lightgbm_predictor).items():
        assert isinstance(predictor, TravelTimePredictor), name


def test_eta_returns_finite_positive_float(small_features_df, lightgbm_predictor, sample_query):
    edge_id, depart_time = sample_query
    for name, predictor in all_predictors(small_features_df, lightgbm_predictor).items():
        eta = predictor.eta(edge_id, depart_time)
        assert isinstance(eta, float), name
        assert eta > 0, name


def test_eta_with_confidence_matches_eta_and_has_nonnegative_sigma(
    small_features_df, lightgbm_predictor, sample_query
):
    edge_id, depart_time = sample_query
    for name, predictor in all_predictors(small_features_df, lightgbm_predictor).items():
        eta = predictor.eta(edge_id, depart_time)
        eta2, sigma = predictor.eta_with_confidence(edge_id, depart_time)
        assert isinstance(sigma, float), name
        assert sigma >= 0, name
        assert eta2 == eta, f"{name}: eta_with_confidence()[0] must match eta()"


def test_oracle_has_zero_uncertainty(small_features_df, sample_query):
    edge_id, depart_time = sample_query
    oracle = OraclePredictor(small_features_df)
    _, sigma = oracle.eta_with_confidence(edge_id, depart_time)
    assert sigma == 0.0


def test_adversarial_underestimates_within_budget(small_features_df, sample_query):
    edge_id, depart_time = sample_query
    oracle = OraclePredictor(small_features_df)
    budget = 0.3
    adversary = AdversarialPredictor(oracle, budget=budget)

    true_eta = oracle.eta(edge_id, depart_time)
    adv_eta = adversary.eta(edge_id, depart_time)
    assert adv_eta == pytest.approx(true_eta * (1 - budget))
    assert adv_eta < true_eta


def test_noisy_predictor_is_deterministic_per_query(small_features_df, sample_query):
    edge_id, depart_time = sample_query
    oracle = OraclePredictor(small_features_df)
    noisy = NoisyPredictor(oracle, sigma_level=0.2, seed=42)

    eta_a = noisy.eta(edge_id, depart_time)
    eta_b = noisy.eta(edge_id, depart_time)
    assert eta_a == eta_b, "same query must reproduce bit-identical noise"

    noisy_other_seed = NoisyPredictor(oracle, sigma_level=0.2, seed=7)
    assert noisy_other_seed.eta(edge_id, depart_time) != eta_a
