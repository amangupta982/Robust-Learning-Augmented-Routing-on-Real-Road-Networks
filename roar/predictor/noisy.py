"""NoisyPredictor: wraps another predictor (normally the oracle) and injects
controlled, reproducible error -- the knob that produces the smoothness
curve (how routing quality degrades as prediction error grows, swept on the
SAME queries -- see ROAR_Research_Blueprint.md).
"""

from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np

from roar.predictor.base import TravelTimePredictor

# Travel time must stay strictly positive; this is just a numerical floor
# for pathological noise draws, not a physically meaningful value.
_MIN_ETA_S = 1e-3


class NoisyPredictor(TravelTimePredictor):
    """eta = true_eta + N(0, sigma_level * true_eta). `sigma_level` is a
    *relative* noise std (e.g. 0.2 = 20% of the true travel time).

    Deterministic per (edge_id, depart_time): the RNG is seeded from a hash
    of the query itself, not from call order or a shared mutable generator,
    so repeated queries -- and repeated experiment runs with the same seed
    -- reproduce bit-identical noise (CLAUDE.md rule 3)."""

    def __init__(self, base_predictor: TravelTimePredictor, sigma_level: float, seed: int = 42):
        if sigma_level < 0:
            raise ValueError(f"sigma_level must be >= 0, got {sigma_level}")
        self._base = base_predictor
        self._sigma_level = sigma_level
        self._seed = seed

    def _rng_for(self, edge_id: str, depart_time: dt.datetime) -> np.random.Generator:
        key = f"{self._seed}:{edge_id}:{depart_time.isoformat()}".encode()
        digest = hashlib.sha256(key).digest()[:8]
        return np.random.default_rng(int.from_bytes(digest, "big"))

    def eta(self, edge_id: str, depart_time: dt.datetime) -> float:
        return self.eta_with_confidence(edge_id, depart_time)[0]

    def eta_with_confidence(self, edge_id: str, depart_time: dt.datetime) -> tuple[float, float]:
        true_eta = self._base.eta(edge_id, depart_time)
        sigma = self._sigma_level * true_eta
        if sigma == 0:
            return true_eta, 0.0
        rng = self._rng_for(edge_id, depart_time)
        noisy_eta = max(true_eta + float(rng.normal(0.0, sigma)), _MIN_ETA_S)
        return noisy_eta, sigma
