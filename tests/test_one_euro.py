"""1€ filter sanity tests."""

import math
import random

from vision.one_euro import OneEuroFilter, OneEuroPoint2D


def test_constant_signal_unchanged():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    t = 0.0
    out = []
    for _ in range(50):
        out.append(f.filter(5.0, t))
        t += 1.0 / 30.0
    assert all(abs(y - 5.0) < 1e-6 for y in out)


def test_jitter_is_smoothed():
    """Add white noise to a constant target. Output variance < input variance."""
    random.seed(0)
    raw = [5.0 + random.gauss(0, 0.5) for _ in range(200)]
    f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    t = 0.0
    smoothed = []
    for x in raw:
        smoothed.append(f.filter(x, t))
        t += 1.0 / 30.0

    def variance(xs: list[float]) -> float:
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / len(xs)

    assert variance(smoothed) < variance(raw) * 0.5


def test_point2d_independent_axes():
    # Use a high min_cutoff + beta so the filter is in "responsive" mode
    # — we just want to verify both axes work independently, not measure
    # tracking accuracy (that would belong in a tuning test).
    p = OneEuroPoint2D(min_cutoff=10.0, beta=0.5)
    t = 0.0
    last = (0.0, 0.0)
    for i in range(60):
        x = math.sin(i / 10.0)
        y = math.cos(i / 10.0)
        last = p.filter(x, y, t)
        t += 1.0 / 30.0
    # Both axes should produce real numbers in a sane range, and the
    # x and y outputs must be different from each other (decoupled axes).
    assert -1.5 < last[0] < 1.5
    assert -1.5 < last[1] < 1.5
    assert last[0] != last[1]
