"""MediaPipe Tasks Hands wrapper (new API, forward-compatible).

Auto-downloads `hand_landmarker.task` on first use. Returns normalized
[0, 1] coords in frame space.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from utils.paths import MODELS_DIR
except ImportError:
    from ..utils.paths import MODELS_DIR

from .one_euro import OneEuroPoint2D

log = logging.getLogger("hands")

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/"
              "hand_landmarker.task")
_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"


def _ensure_model() -> Path:
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("downloading hand_landmarker.task to %s ...", _MODEL_PATH)
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    log.info("downloaded (%d bytes)", _MODEL_PATH.stat().st_size)
    return _MODEL_PATH


@dataclass
class HandResult:
    tip_x: float
    tip_y: float
    palm_x: float
    palm_y: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox_norm(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class HandsDetector:
    def __init__(self, max_num_hands: int = 1,
                 min_det: float = 0.5, min_trk: float = 0.5,
                 smooth: bool = True,
                 smooth_min_cutoff: float = 1.5,
                 smooth_beta: float = 0.01) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        model_path = _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_hand_detection_confidence=min_det,
            min_hand_presence_confidence=min_det,
            min_tracking_confidence=min_trk,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms: int = 0
        # Per-attribute 1€ filters reduce sub-pixel jitter on the tip/palm
        # without lagging fast motions. Reset whenever the hand disappears
        # so the filter does not drag in stale state across re-detections.
        self._smooth = smooth
        self._smooth_kw = dict(min_cutoff=smooth_min_cutoff, beta=smooth_beta)
        self._tip_f: Optional[OneEuroPoint2D] = None
        self._palm_f: Optional[OneEuroPoint2D] = None

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.monotonic() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        res = self._detector.detect_for_video(mp_img, ts_ms)
        if not res.hand_landmarks:
            # hand lost — discard filter state so we don't snap back from
            # a stale point when the user re-enters the frame elsewhere.
            self._tip_f = None
            self._palm_f = None
            return None
        lms = res.hand_landmarks[0]
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]

        # Use the average of the index fingertip (8) and DIP joint (7) as a
        # more stable "tip" than landmark 8 alone — single-landmark jitter
        # is the dominant noise source on MediaPipe Hands.
        raw_tip_x = (float(lms[8].x) + float(lms[7].x)) * 0.5
        raw_tip_y = (float(lms[8].y) + float(lms[7].y)) * 0.5
        raw_palm_x = float(lms[9].x)
        raw_palm_y = float(lms[9].y)

        if self._smooth:
            t = ts_ms / 1000.0
            if self._tip_f is None:
                self._tip_f = OneEuroPoint2D(**self._smooth_kw)
                self._palm_f = OneEuroPoint2D(**self._smooth_kw)
            tip_x, tip_y = self._tip_f.filter(raw_tip_x, raw_tip_y, t)
            palm_x, palm_y = self._palm_f.filter(raw_palm_x, raw_palm_y, t)
        else:
            tip_x, tip_y = raw_tip_x, raw_tip_y
            palm_x, palm_y = raw_palm_x, raw_palm_y

        return HandResult(
            tip_x=tip_x, tip_y=tip_y,
            palm_x=palm_x, palm_y=palm_y,
            x1=float(min(xs)), y1=float(min(ys)),
            x2=float(max(xs)), y2=float(max(ys)),
        )

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:
            pass
