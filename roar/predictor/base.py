"""Pluggable travel-time predictor interface (CLAUDE.md rule 5).

Routing code (roar/routing/) depends only on this interface -- never on a
concrete predictor class -- so RobustAStar and the baselines can swap in the
oracle, the trained LightGBM model, the noisy wrapper, or the adversary
without any code change, just a different injected instance.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod

import numpy as np

MPH_TO_MPS = 0.44704
# A predicted or realized speed of 0 mph would make travel time infinite;
# floor it at a slow-crawl equivalent so eta() always returns a finite,
# physically meaningful number (real gridlock, not division by zero).
MIN_SPEED_MPH = 1.0


def travel_time_seconds(length_m, speed_mph):
    """length_m / speed_mph -> seconds. Vectorized: accepts scalars or
    numpy arrays / pandas Series for both arguments."""
    speed = np.maximum(speed_mph, MIN_SPEED_MPH)
    return length_m / (speed * MPH_TO_MPS)


def floor_to_bucket(timestamp: dt.datetime, minutes: int) -> dt.datetime:
    """Snap a depart_time down to the dataset's fixed-width time bucket
    (e.g. 14:07 -> 14:05 for 5-minute buckets), matching how
    roar/graph/features.py discretizes real timestamps onto a uniform grid."""
    epoch_minutes = timestamp.hour * 60 + timestamp.minute
    floored_minutes = (epoch_minutes // minutes) * minutes
    return timestamp.replace(
        hour=floored_minutes // 60, minute=floored_minutes % 60, second=0, microsecond=0
    )


class TravelTimePredictor(ABC):
    """eta() and eta_with_confidence() return seconds; sigma is a
    1-standard-deviation travel-time uncertainty, also in seconds. sigma=0.0
    is a valid, meaningful answer (e.g. a perfect oracle) -- it is not a
    sentinel for "no confidence available"."""

    @abstractmethod
    def eta(self, edge_id: str, depart_time: dt.datetime) -> float: ...

    @abstractmethod
    def eta_with_confidence(
        self, edge_id: str, depart_time: dt.datetime
    ) -> tuple[float, float]: ...
