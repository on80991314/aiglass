"""Simple rule-based fall detector working on IMU samples streamed from glasses.

Input: sequence of (t, ax, ay, az, gx, gy, gz) with ax/ay/az in m/s^2.
Heuristic:
  1. Free-fall window:  |a| < 4 m/s^2 for >= ~150 ms
  2. Impact spike:      |a| > 25 m/s^2
  3. Post-impact tilt:  gravity vector deviates > 60 deg from vertical for >= 2 s
Any 1->2->3 sequence within 3 s is treated as a fall.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class ImuSample:
    t: float
    ax: float
    ay: float
    az: float
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0

    @property
    def a_mag(self) -> float:
        return math.sqrt(self.ax * self.ax + self.ay * self.ay + self.az * self.az)


class FallDetector:
    FREEFALL_THRESH = 4.0
    IMPACT_THRESH = 25.0
    TILT_DEG = 60.0
    FREEFALL_MIN_S = 0.15
    TILT_MIN_S = 2.0
    WINDOW_S = 3.0

    def __init__(self) -> None:
        self._buf: Deque[ImuSample] = deque(maxlen=512)
        self._freefall_start: Optional[float] = None
        self._impact_t: Optional[float] = None
        self._tilt_start: Optional[float] = None
        self.last_fall_t: float = 0.0

    def push(self, s: ImuSample) -> bool:
        """Return True the first tick a fall is recognised."""
        self._buf.append(s)
        mag = s.a_mag

        # Phase 1: free-fall
        if mag < self.FREEFALL_THRESH:
            if self._freefall_start is None:
                self._freefall_start = s.t
        else:
            if (self._freefall_start is not None
                    and s.t - self._freefall_start < self.FREEFALL_MIN_S):
                self._freefall_start = None

        # Phase 2: impact
        if mag > self.IMPACT_THRESH and self._freefall_start is not None \
                and s.t - self._freefall_start >= self.FREEFALL_MIN_S:
            self._impact_t = s.t

        # Phase 3: sustained tilt after impact
        if self._impact_t is not None and s.t - self._impact_t <= self.WINDOW_S:
            # gravity should be near (0,0,9.8) when upright
            tilt_cos = s.az / (mag or 1e-6)
            tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, tilt_cos))))
            if tilt_deg > self.TILT_DEG:
                if self._tilt_start is None:
                    self._tilt_start = s.t
                elif s.t - self._tilt_start >= self.TILT_MIN_S:
                    self._reset()
                    if s.t - self.last_fall_t > 10.0:
                        self.last_fall_t = s.t
                        return True
            else:
                self._tilt_start = None
        elif self._impact_t is not None and s.t - self._impact_t > self.WINDOW_S:
            self._reset()
        return False

    def _reset(self) -> None:
        self._freefall_start = None
        self._impact_t = None
        self._tilt_start = None


def now_sample(ax: float, ay: float, az: float,
               gx: float = 0.0, gy: float = 0.0, gz: float = 0.0) -> ImuSample:
    return ImuSample(time.monotonic(), ax, ay, az, gx, gy, gz)
