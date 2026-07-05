"""RobustAStar's two lambda modes, on the hand-built diamond graph (no real
data needed): fixed lambda is used verbatim; confidence-modulated lambda
degrades exactly as designed as predictor uncertainty (sigma) grows.
"""

from __future__ import annotations

import pytest
from roar.routing.baselines import static_free_flow_cost
from roar.routing.robust_astar import RobustAStar

from tests.routing_fixtures import DEPART_TIME, StubPredictor, diamond_graph


def test_fixed_lambda_is_used_verbatim_regardless_of_confidence():
    g = diamond_graph()
    # classical path is A-C-B-D (see diamond_graph docstring); give the
    # predictor some values with high sigma -- fixed mode must ignore it.
    predictor = StubPredictor(
        eta_by_edge={"AC": 5.0, "CB": 3.0, "BD": 2.0},
        sigma_by_edge={"AC": 50.0, "CB": 30.0, "BD": 20.0},
    )
    robust = RobustAStar(
        g, predictor, static_free_flow_cost, alpha=0.5, lambda_base=0.7, confidence_modulated=False
    )
    result = robust.search("A", "D", DEPART_TIME)
    assert result.lambda_used == 0.7


def test_confidence_modulated_lambda_equals_base_when_sigma_is_zero():
    g = diamond_graph()
    predictor = StubPredictor(
        eta_by_edge={"AC": 5.0, "CB": 3.0, "BD": 2.0},
        sigma_by_edge={"AC": 0.0, "CB": 0.0, "BD": 0.0},
    )
    robust = RobustAStar(
        g, predictor, static_free_flow_cost, alpha=0.5, lambda_base=0.8, confidence_modulated=True
    )
    result = robust.search("A", "D", DEPART_TIME)
    assert result.lambda_used == pytest.approx(0.8)


def test_confidence_modulated_lambda_shrinks_as_sigma_grows():
    g = diamond_graph()
    # mean_relative_sigma = mean(50/5, 30/3, 20/2) = mean(10, 10, 10) = 10
    # lambda_eff = lambda_base / (1 + 10) = lambda_base / 11
    predictor = StubPredictor(
        eta_by_edge={"AC": 5.0, "CB": 3.0, "BD": 2.0},
        sigma_by_edge={"AC": 50.0, "CB": 30.0, "BD": 20.0},
    )
    robust = RobustAStar(
        g, predictor, static_free_flow_cost, alpha=0.5, lambda_base=1.0, confidence_modulated=True
    )
    result = robust.search("A", "D", DEPART_TIME)
    assert result.lambda_used == pytest.approx(1.0 / 11, rel=1e-6)
    assert result.lambda_used < 1.0


def test_confidence_modulated_falls_back_to_base_when_predictor_has_no_coverage():
    g = diamond_graph()
    predictor = StubPredictor(eta_by_edge={})  # zero coverage anywhere
    robust = RobustAStar(
        g, predictor, static_free_flow_cost, alpha=0.5, lambda_base=0.6, confidence_modulated=True
    )
    result = robust.search("A", "D", DEPART_TIME)
    assert result.lambda_used == 0.6


@pytest.mark.parametrize("bad_lambda", [-0.1, 1.1])
def test_lambda_base_out_of_range_is_rejected(bad_lambda):
    g = diamond_graph()
    predictor = StubPredictor(eta_by_edge={})
    with pytest.raises(ValueError, match="lambda_base"):
        RobustAStar(g, predictor, static_free_flow_cost, alpha=0.5, lambda_base=bad_lambda)
