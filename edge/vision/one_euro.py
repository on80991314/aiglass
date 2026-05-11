"""1€ filter (Casiez et al., 2012) — adaptive low-pass filter for noisy
real-time signals like MediaPipe landmarks.

The cutoff frequency rises with the speed of the signal: slow movement
gets aggressive smoothing (kills jitter), fast movement gets light
smoothing (preserves responsiveness). One small tunable knob each:
  * min_cutoff: cutoff at zero speed. Lower = more smoothing.
  * beta:       how much speed raises the cutoff. Higher = more responsive.
"""

from __future__ import annotations

import math
from typing import Optional


class _LowPass:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.y: Optional[float] = None
        self.last_raw: Optional[float] = None

    def filter(self, x: float, alpha: Optional[float] = None) -> float:
        a = self.alpha if alpha is None else alpha
        if self.y is None:
            self.y = x
        else:
            self.y = a * x + (1.0 - a) * self.y
        self.last_raw = x
        return self.y


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    def __init__(self,
                 min_cutoff: float = 1.0,
                 beta: float = 0.007,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass(_alpha(min_cutoff, 1.0))
        self._dx = _LowPass(_alpha(d_cutoff, 1.0))
        self._t_prev: Optional[float] = None

    def filter(self, x: float, t: float) -> float:
        if self._t_prev is None:
            self._t_prev = t
            return self._x.filter(x, alpha=1.0)
        dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t
        prev = self._x.last_raw if self._x.last_raw is not None else x
        dx = (x - prev) / dt
        edx = self._dx.filter(dx, alpha=_alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.filter(x, alpha=_alpha(cutoff, dt))


class OneEuroPoint2D:
    """Convenience wrapper for an (x, y) pair sharing one filter setting."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007) -> None:
        self.fx = OneEuroFilter(min_cutoff, beta)
        self.fy = OneEuroFilter(min_cutoff, beta)

    def filter(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self.fx.filter(x, t), self.fy.filter(y, t)
