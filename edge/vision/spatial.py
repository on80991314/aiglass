"""Turn detections into spoken spatial guidance."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .detector import Detection
from .labels import EN_TO_ZH, to_zh as zh_label  # noqa: F401  re-exported


def describe_position(d: Detection, frame_w: int, frame_h: int, target_zh: str) -> str:
    third = frame_w / 3
    if d.cx < third:
        horiz = "左前方"
    elif d.cx < 2 * third:
        horiz = "正前方"
    else:
        horiz = "右前方"

    ratio = d.area / float(frame_w * frame_h or 1)
    if ratio < 0.02:
        dist = "較遠"
    elif ratio < 0.10:
        dist = "中距離"
    else:
        dist = "近處"

    return f"{target_zh}在{horiz}{dist}"


def traffic_light_color(frame_bgr: np.ndarray, d: Detection) -> str:
    """Rough HSV check for red/green. Returns 'red' | 'green' | 'unknown'."""
    y1, y2 = max(0, d.y1), min(frame_bgr.shape[0], d.y2)
    x1, x2 = max(0, d.x1), min(frame_bgr.shape[1], d.x2)
    if y2 <= y1 or x2 <= x1:
        return "unknown"
    roi = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = cv2.countNonZero(cv2.inRange(hsv, (0, 120, 80), (10, 255, 255))) + \
          cv2.countNonZero(cv2.inRange(hsv, (170, 120, 80), (180, 255, 255)))
    green = cv2.countNonZero(cv2.inRange(hsv, (40, 80, 80), (85, 255, 255)))
    if green > red * 1.2 and green > 30:
        return "green"
    if red > green * 1.2 and red > 30:
        return "red"
    return "unknown"


def crosswalk_hint(frame_bgr: np.ndarray) -> Optional[str]:
    """Placeholder crosswalk detector.

    Until a custom YOLO head is trained (Phase 2), this gives a weak
    geometric hint by counting long horizontal bright bands in the lower
    half of the frame. Swap this out with a trained classifier later.
    """
    h, w = frame_bgr.shape[:2]
    lower = frame_bgr[h // 2:, :]
    gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # horizontal run length by row
    row_sums = bw.sum(axis=1)
    bright_rows = int((row_sums > 0.4 * 255 * w).sum())
    if bright_rows > 10:
        return "前方疑似斑馬線"
    return None


# COCO_ZH and zh_label are re-exported from .labels for backwards compat.
COCO_ZH = EN_TO_ZH
