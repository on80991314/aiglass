"""YOLOv8 wrapper with optional detection stabilizer. Loads on first use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from utils.paths import resolve_model
except ImportError:
    from ..utils.paths import resolve_model


@dataclass
class Detection:
    label: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class _Track:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf_ema: float
    activated: bool
    miss: int
    age: int


class DetectionStabilizer:
    """Per-class greedy-IoU matcher + EMA smoother + hysteresis gate.

    Smooths single-frame YOLO flicker by:
      - EMA-averaging bbox and confidence when a track is re-matched, which
        reduces jitter from conf drifting around the YOLO threshold.
      - Hysteresis: a track must reach `enter_thresh` at least once to be
        emitted, and keeps emitting while its EMA conf >= `stay_thresh`.
      - Short gap-bridging: a matched track survives up to `max_miss`
        consecutive missed frames (kept at its last smoothed bbox), masking
        one-frame YOLO drops. Longer gaps drop the track -- which is
        important so that downstream occlusion detection (FSM) can still see
        "object truly gone" when the user's hand actually covers it.
    """

    def __init__(self,
                 enter_thresh: float = 0.45,
                 stay_thresh: float = 0.20,
                 bbox_alpha: float = 0.5,
                 conf_alpha: float = 0.5,
                 match_iou: float = 0.30,
                 max_miss: int = 1):
        self.enter_thresh = enter_thresh
        self.stay_thresh = stay_thresh
        self.bbox_alpha = bbox_alpha
        self.conf_alpha = conf_alpha
        self.match_iou = match_iou
        self.max_miss = max_miss
        self._tracks: List[_Track] = []

    @staticmethod
    def _iou(t: _Track, d: Detection) -> float:
        ix1 = max(t.x1, d.x1)
        iy1 = max(t.y1, d.y1)
        ix2 = min(t.x2, d.x2)
        iy2 = min(t.y2, d.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        aa = max(0.0, t.x2 - t.x1) * max(0.0, t.y2 - t.y1)
        ab = max(0.0, d.x2 - d.x1) * max(0.0, d.y2 - d.y1)
        u = aa + ab - inter
        return 0.0 if u <= 0 else inter / u

    def step(self, dets: List[Detection]) -> List[Detection]:
        unmatched = set(range(len(self._tracks)))
        matched_pairs: List[Tuple[int, Detection]] = []
        new_dets: List[Detection] = []

        # Greedy match: dets sorted by conf desc pick best-IoU same-class track.
        for d in sorted(dets, key=lambda x: x.conf, reverse=True):
            best_i = -1
            best_iou = self.match_iou
            for i in unmatched:
                t = self._tracks[i]
                if t.label != d.label:
                    continue
                iou = self._iou(t, d)
                if iou > best_iou:
                    best_iou = iou
                    best_i = i
            if best_i >= 0:
                matched_pairs.append((best_i, d))
                unmatched.discard(best_i)
            else:
                new_dets.append(d)

        # Update matched tracks with EMA on bbox + confidence.
        a_b, a_c = self.bbox_alpha, self.conf_alpha
        for i, d in matched_pairs:
            t = self._tracks[i]
            t.x1 = (1 - a_b) * t.x1 + a_b * d.x1
            t.y1 = (1 - a_b) * t.y1 + a_b * d.y1
            t.x2 = (1 - a_b) * t.x2 + a_b * d.x2
            t.y2 = (1 - a_b) * t.y2 + a_b * d.y2
            t.conf_ema = (1 - a_c) * t.conf_ema + a_c * d.conf
            t.miss = 0
            t.age += 1
            if t.conf_ema >= self.enter_thresh:
                t.activated = True

        # Age out tracks that did not get a detection this frame.
        for i in sorted(unmatched, reverse=True):
            self._tracks[i].miss += 1
            if self._tracks[i].miss > self.max_miss:
                self._tracks.pop(i)

        # Spawn tracks for detections that matched nothing.
        for d in new_dets:
            self._tracks.append(_Track(
                label=d.label,
                x1=float(d.x1), y1=float(d.y1),
                x2=float(d.x2), y2=float(d.y2),
                conf_ema=d.conf,
                activated=d.conf >= self.enter_thresh,
                miss=0, age=1,
            ))

        # Emit activated tracks that stay above the low (stay) threshold.
        out: List[Detection] = []
        for t in self._tracks:
            if t.activated and t.conf_ema >= self.stay_thresh:
                out.append(Detection(
                    label=t.label, conf=float(t.conf_ema),
                    x1=int(round(t.x1)), y1=int(round(t.y1)),
                    x2=int(round(t.x2)), y2=int(round(t.y2)),
                ))
        return out

    def reset(self) -> None:
        self._tracks.clear()


class YoloDetector:
    def __init__(self, weights: str = "yolov8s.pt", device: str = "cpu",
                 conf: float = 0.20, imgsz: int = 960,
                 stabilize: bool = True,
                 enter_thresh: float = 0.45, stay_thresh: float = 0.20,
                 max_miss: int = 1) -> None:
        self.weights = weights
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self._model = None
        self._stabilizer: Optional[DetectionStabilizer] = (
            DetectionStabilizer(enter_thresh=enter_thresh,
                                stay_thresh=stay_thresh,
                                max_miss=max_miss)
            if stabilize else None
        )

    def _ensure_loaded(self):
        if self._model is None:
            from ultralytics import YOLO  # heavy import, deferred
            # Allow bare names like "yolov8s.pt" — resolve under <repo>/models/
            self._model = YOLO(str(resolve_model(self.weights)))
        return self._model

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        model = self._ensure_loaded()
        res = model.predict(frame_bgr, device=self.device, conf=self.conf,
                            imgsz=self.imgsz, verbose=False)[0]
        names = res.names
        raw: List[Detection] = []
        if res.boxes is not None:
            for b in res.boxes:
                cls = int(b.cls[0].item())
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                raw.append(Detection(
                    label=names[cls],
                    conf=float(b.conf[0].item()),
                    x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                ))
        if self._stabilizer is None:
            return raw
        return self._stabilizer.step(raw)

    def best(self, dets: List[Detection], label_contains: str) -> Optional[Detection]:
        lc = label_contains.lower()
        cands = [d for d in dets if lc in d.label.lower()]
        if not cands:
            return None
        return max(cands, key=lambda d: d.conf)
