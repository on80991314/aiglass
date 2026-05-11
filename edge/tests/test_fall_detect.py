"""Feed a synthetic fall profile into FallDetector and assert it trips.
Usage: python -m edge.tests.test_fall_detect
"""

import sys
import time

sys.path.insert(0, __file__.rsplit("edge", 1)[0] + "edge")

from vision.fall_detect import FallDetector, ImuSample  # noqa: E402


def synth():
    out = []
    t = 0.0
    # 1 s upright
    for _ in range(100):
        out.append(ImuSample(t, 0, 0, 9.8))
        t += 0.01
    # 300 ms free-fall
    for _ in range(30):
        out.append(ImuSample(t, 0, 0, 1.5))
        t += 0.01
    # impact spike
    out.append(ImuSample(t, 0, 0, 40.0)); t += 0.01
    # 2.5 s lying on side (gravity ~ x)
    for _ in range(250):
        out.append(ImuSample(t, 9.8, 0, 0.5))
        t += 0.01
    return out


def main():
    fd = FallDetector()
    tripped = False
    for s in synth():
        if fd.push(s):
            tripped = True
            print("FALL DETECTED at t=", s.t)
            break
    assert tripped, "expected fall detection to trigger"
    print("OK")


if __name__ == "__main__":
    main()
