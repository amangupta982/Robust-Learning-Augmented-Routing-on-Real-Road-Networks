"""AdversarialPredictor: worst-case corruption within a fixed relative
budget -- used for the robustness experiments (RQ2/RQ3): the predictor is
allowed to be wrong by up to `budget` (a fraction of the true travel time),
and it spends that entire budget in the direction that damages a
predictor-trusting router the most.

Design choices, both deliberate:

1. It ALWAYS underestimates by the full budget, never overestimates.
   Overestimating a truly-fast edge only wastes some of the router's trust
   (it avoids an edge that was actually fine); underestimating a truly-slow
   edge is what lures a predictor-trusting router onto a genuinely bad edge
   -- the failure mode the ROBUSTNESS guarantee exists to bound. A budget
   that could go either way would be a weaker, non-worst-case adversary.
2. It reports zero self-doubt (sigma=0), i.e. it lies about its own
   confidence too. A worst-case adversary that corrupted eta but honestly
   flagged a wide sigma would be trivially defeated by any guard that reads
   sigma -- that's not the worst case we need to test against.
"""

from __future__ import annotations

import datetime as dt

from roar.predictor.base import TravelTimePredictor

_MIN_ETA_S = 1e-3


class AdversarialPredictor(TravelTimePredictor):
    def __init__(self, base_predictor: TravelTimePredictor, budget: float):
        if not 0 <= budget < 1:
            raise ValueError(f"budget must be in [0, 1), got {budget}")
        self._base = base_predictor
        self._budget = budget

    def eta(self, edge_id: str, depart_time: dt.datetime) -> float:
        true_eta = self._base.eta(edge_id, depart_time)
        return max(true_eta * (1 - self._budget), _MIN_ETA_S)

    def eta_with_confidence(self, edge_id: str, depart_time: dt.datetime) -> tuple[float, float]:
        return self.eta(edge_id, depart_time), 0.0
