"""Find-and-grab state machine (6 states).

All coordinates normalized to [0, 1] in frame space (top-left origin).

States:
  WAITING_FOR_COMMAND -> SEARCHING_OBJECT  (on STT command "想要找 XX")
  SEARCHING_OBJECT    -> GUIDING_HEAD      (YOLO found the target)
  GUIDING_HEAD        -> GUIDING_HAND      (target centered for N consecutive frames)
  GUIDING_HAND        -> CONFIRM_GRAB      (hand looks like it touched the object)
  CONFIRM_GRAB        -> GRAB_SUCCESS      (user voice 「拿到了」 — confirmed)
  CONFIRM_GRAB        -> GUIDING_HAND      (user voice 「還沒拿到」 — keep guiding)
  GRAB_SUCCESS        -> WAITING_FOR_COMMAND (after 3 s)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.detector import Detection
from vision.hands import HandResult
from vision.labels import parse_find_command


class FGState(Enum):
    WAITING_FOR_COMMAND = auto()
    SEARCHING_OBJECT = auto()
    GUIDING_HEAD = auto()
    GUIDING_HAND = auto()
    CONFIRM_GRAB = auto()
    GRAB_SUCCESS = auto()


@dataclass
class FindGrabConfig:
    # Head guidance
    head_left_thresh: float = 0.4
    head_right_thresh: float = 0.6
    head_up_thresh: float = 0.4        # y_norm < this -> object is above center (look up)
    head_down_thresh: float = 0.6      # y_norm > this -> object is below center (look down)
    center_lo: float = 0.35
    center_hi: float = 0.65
    center_frames_required: int = 10

    # Hand guidance
    hand_tolerance: float = 0.08       # +/- on normalized coords
    grab_radius: float = 0.10          # distance from hand-tip to object-center
    iou_threshold: float = 0.55        # hand bbox vs object bbox (substantial overlap)
    # Number of consecutive frames that must satisfy (near_2d AND occluded)
    # before we commit to CONFIRM_GRAB. Guards against single-frame YOLO flicker
    # being misread as "hand is occluding the object".
    grab_frames_required: int = 3

    # Direct contact (does not require occlusion baseline)
    # When the fingertip is well inside the object's bbox AND the hand bbox
    # overlaps the object bbox, the hand is almost certainly touching the
    # front of the object — accept as a grab without occlusion evidence.
    direct_contact_iou: float = 0.20
    direct_contact_iou_strong: float = 0.40

    # Transition timing
    grab_success_hold_s: float = 3.0

    # Throttle per-state voice prompts so the terminal doesn't flood
    head_prompt_interval_s: float = 1.2
    hand_prompt_interval_s: float = 0.6
    search_prompt_interval_s: float = 2.0
    behind_prompt_interval_s: float = 1.0
    confirm_prompt_interval_s: float = 2.5

    # CONFIRM_GRAB
    # Auto-fall back to GUIDING_HAND if user gives no answer for this long.
    confirm_timeout_s: float = 30.0

    # Occlusion-as-depth (方案 A)
    # Object persistence: how long to keep the last-known bbox when YOLO loses it
    # but the hand bbox is covering roughly the same area (=> hand is occluding).
    occlusion_grace_s: float = 0.8
    # Baseline (clean-view) sampling of the target, used to judge occlusion.
    baseline_min_samples: int = 5
    baseline_max_samples: int = 30
    # A frame is "clean" (eligible for baseline) when hand-object IoU < this.
    hand_clean_iou: float = 0.02
    # Occlusion evidence thresholds: object considered (partially) occluded when
    # its current confidence or bbox area drops to <= ratio * baseline.
    grab_conf_drop_ratio: float = 0.7
    grab_area_drop_ratio: float = 0.7
    # Minimum IoU(last-obj, hand) required to believe hand is covering the object
    # when it vanishes from YOLO (used with occlusion_grace_s). Too-low values
    # cause a single-frame YOLO flicker with edge hand overlap to be misread as
    # "hand is covering the object".
    occlusion_overlap_iou: float = 0.30


@dataclass
class _Timers:
    last_head_prompt: float = 0.0
    last_hand_prompt: float = 0.0
    last_search_prompt: float = 0.0
    last_behind_prompt: float = 0.0
    last_confirm_prompt: float = 0.0
    confirm_started: float = 0.0
    grab_success_start: float = 0.0


def _bbox_iou(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class FindGrabFSM:
    def __init__(self,
                 cfg: Optional[FindGrabConfig] = None,
                 emit: Optional[Callable[[str], None]] = None) -> None:
        self.cfg = cfg or FindGrabConfig()
        self._emit = emit or self._default_emit
        self.state: FGState = FGState.WAITING_FOR_COMMAND
        self.target_zh: Optional[str] = None
        self.target_en: Optional[str] = None
        self.center_streak: int = 0
        self.t = _Timers()
        # Occlusion-as-depth state (reset when entering GUIDING_HAND)
        self._last_obj: Optional[Detection] = None
        self._last_obj_t: float = 0.0
        self._baseline_conf: float = 0.0      # running mean of clean-view conf
        self._baseline_area: float = 0.0      # running mean of clean-view normalized area
        self._baseline_n: int = 0
        self._baseline_announced: bool = False
        self._grab_streak: int = 0            # consecutive (near_2d AND occluded) frames
        # CONFIRM_GRAB: latest known relative position of hand vs object
        self._last_dx: float = 0.0
        self._last_dy: float = 0.0
        # Subtitle (set from any thread; consumed by drawer)
        self._subtitle_lock = threading.Lock()
        self._subtitle_text: str = ""
        self._subtitle_until: float = 0.0

    # ---------------- subtitle (thread-safe) ----------------
    def set_subtitle(self, text: str, duration_s: float = 2.0) -> None:
        with self._subtitle_lock:
            self._subtitle_text = text or ""
            self._subtitle_until = time.monotonic() + duration_s

    def get_subtitle(self) -> str:
        with self._subtitle_lock:
            if time.monotonic() > self._subtitle_until:
                return ""
            return self._subtitle_text

    # ---------------- entry points ----------------
    def on_speech(self, text: str) -> None:
        """Legacy fast path: regex-only. Use set_target() from intent router instead."""
        parsed = parse_find_command(text)
        if not parsed:
            self._emit(f"[STT] 未辨識出尋物指令: {text!r}")
            return
        zh, en = parsed
        self.set_target(zh, en)

    def set_target(self, zh: str, en: str) -> None:
        """Called by IntentRouter+LabelNormalizer after natural-language parse.

        Allows mid-flight target switching from any state except GRAB_SUCCESS.
        E.g., user says 「找杯子」 then partway through 「改找電腦」.
        """
        if self.state == FGState.GRAB_SUCCESS:
            return
        switching = (self.target_en is not None
                     and self.target_en.lower() != en.lower())
        self.target_zh, self.target_en = zh, en
        self.center_streak = 0
        self._reset_grab_baseline()
        # Drop straight back to SEARCHING_OBJECT for the new target.
        self.state = FGState.SEARCHING_OBJECT
        self.t.last_search_prompt = 0.0
        if switching:
            self._emit(f"[FSM] 切換目標 -> {zh} ({en})，重新搜尋")
        else:
            self._emit(f"[FSM] 收到指令，開始尋找 {zh} ({en})")

    def cancel(self) -> None:
        """User asked to abort current search."""
        if self.state == FGState.WAITING_FOR_COMMAND:
            return
        self._emit("[FSM] 收到取消指令")
        self._reset()

    def hotword_reset(self) -> None:
        """Hotword interrupt — full reset, regardless of state."""
        self._emit("[FSM] HOTWORD — 強制重置")
        self._reset()

    def confirm_grab(self, success: bool) -> None:
        """User voice response to the 「是否拿到了？」 prompt."""
        if self.state != FGState.CONFIRM_GRAB:
            return
        if success:
            self._emit("[FSM] 使用者確認已拿到")
            self._goto(FGState.GRAB_SUCCESS)
        else:
            self._emit("[FSM] 使用者表示尚未拿到，回到手部導引")
            self._goto(FGState.GUIDING_HAND)

    def step(self, frame_size: tuple[int, int],
             dets: List[Detection],
             hand: Optional[HandResult]) -> None:
        """Drive the FSM one video frame. frame_size = (w, h)."""
        w, h = frame_size
        if w <= 0 or h <= 0:
            return

        if self.state == FGState.WAITING_FOR_COMMAND:
            return
        if self.state == FGState.GRAB_SUCCESS:
            if time.monotonic() - self.t.grab_success_start >= self.cfg.grab_success_hold_s:
                self._reset()
            return

        obj = self._find_target(dets)
        if self.state == FGState.SEARCHING_OBJECT:
            self._tick_search(obj)
        elif self.state == FGState.GUIDING_HEAD:
            self._tick_head(obj, w, h)
        elif self.state == FGState.GUIDING_HAND:
            self._tick_hand(obj, hand, w, h)
        elif self.state == FGState.CONFIRM_GRAB:
            self._tick_confirm(obj, hand, w, h)

    # ---------------- per-state ticks ----------------
    def _tick_search(self, obj: Optional[Detection]) -> None:
        now = time.monotonic()
        if obj is None:
            if now - self.t.last_search_prompt >= self.cfg.search_prompt_interval_s:
                self.t.last_search_prompt = now
                self._emit(f"[FSM/SEARCH] 正在尋找 {self.target_zh}...")
            return
        self._emit(f"[FSM/SEARCH] 已發現 {self.target_zh}，進入頭部導引")
        self._goto(FGState.GUIDING_HEAD)

    def _tick_head(self, obj: Optional[Detection], w: int, h: int) -> None:
        if obj is None:
            # lost the object; drop back to searching
            self.center_streak = 0
            self._emit(f"[FSM/HEAD] 失去 {self.target_zh} 視線，重新搜尋")
            self._goto(FGState.SEARCHING_OBJECT)
            return

        x_norm = obj.cx / float(w)
        y_norm = obj.cy / float(h)

        if self.cfg.center_lo <= x_norm <= self.cfg.center_hi \
                and self.cfg.center_lo <= y_norm <= self.cfg.center_hi:
            self.center_streak += 1
        else:
            self.center_streak = 0

        now = time.monotonic()
        if now - self.t.last_head_prompt >= self.cfg.head_prompt_interval_s:
            self.t.last_head_prompt = now
            parts: list[str] = []
            if x_norm < self.cfg.head_left_thresh:
                parts.append("左邊")
            elif x_norm > self.cfg.head_right_thresh:
                parts.append("右邊")
            if y_norm < self.cfg.head_up_thresh:
                parts.append("上方")
            elif y_norm > self.cfg.head_down_thresh:
                parts.append("下方")
            if not parts:
                self._emit(f"[FSM/HEAD] {self.target_zh} 就在你正前方")
            else:
                where = "".join(parts)
                self._emit(f"[FSM/HEAD] {self.target_zh} 在你的{where}，請朝該方向轉動頭部")

        if self.center_streak >= self.cfg.center_frames_required:
            self._emit(f"[FSM/HEAD] 已穩定置中 {self.center_streak} 幀，進入手部導引")
            self._goto(FGState.GUIDING_HAND)

    def _tick_hand(self, obj: Optional[Detection],
                   hand: Optional[HandResult], w: int, h: int) -> None:
        now = time.monotonic()

        # ---- 1) maintain last-known object + clean-view baseline ----
        fresh = obj is not None
        if fresh:
            self._last_obj = obj
            self._last_obj_t = now
            hand_overlap = (
                _bbox_iou(self._obj_bbox_norm(obj, w, h), hand.bbox_norm)
                if hand is not None else 0.0
            )
            if hand_overlap < self.cfg.hand_clean_iou \
                    and self._baseline_n < self.cfg.baseline_max_samples:
                area_norm = obj.area / float(w * h)
                n = self._baseline_n
                self._baseline_conf = (self._baseline_conf * n + obj.conf) / (n + 1)
                self._baseline_area = (self._baseline_area * n + area_norm) / (n + 1)
                self._baseline_n = n + 1
                if (not self._baseline_announced
                        and self._baseline_n >= self.cfg.baseline_min_samples):
                    self._baseline_announced = True
                    self._emit(
                        f"[FSM/HAND] baseline ready "
                        f"(conf={self._baseline_conf:.2f}, "
                        f"area={self._baseline_area:.4f}, n={self._baseline_n})"
                    )

        # ---- 2) resolve effective object + occlusion evidence ----
        eff_obj: Optional[Detection] = obj
        fully_occluded = False
        if not fresh:
            # YOLO sees nothing this frame. If hand is covering the last-known
            # bbox and we're still within grace, treat it as full occlusion.
            if (self._last_obj is not None
                    and hand is not None
                    and now - self._last_obj_t < self.cfg.occlusion_grace_s):
                last_norm = self._obj_bbox_norm(self._last_obj, w, h)
                if _bbox_iou(last_norm, hand.bbox_norm) >= self.cfg.occlusion_overlap_iou:
                    eff_obj = self._last_obj
                    fully_occluded = True
            if eff_obj is None:
                # genuine loss of sight — go back to head guidance
                self._emit(f"[FSM/HAND] 失去 {self.target_zh}，退回頭部導引")
                self.center_streak = 0
                self._goto(FGState.GUIDING_HEAD)
                return

        # ---- 3) hand missing ----
        if hand is None:
            if now - self.t.last_hand_prompt >= self.cfg.hand_prompt_interval_s:
                self.t.last_hand_prompt = now
                self._emit("[FSM/HAND] 找不到你的手，請把手伸進畫面")
            return

        # ---- 4) 2D proximity ----
        # Use fingertip for the contact test (it's what actually touches),
        # but use palm for direction prompts ("向左/右/上/下移") because
        # the user thinks of "their hand" as the palm, not the fingertip.
        ox = eff_obj.cx / float(w)
        oy = eff_obj.cy / float(h)
        tx, ty = hand.tip_x, hand.tip_y           # for contact / dist
        px, py = hand.palm_x, hand.palm_y         # for direction guidance
        dx_tip = ox - tx
        dy_tip = oy - ty
        dx_palm = ox - px
        dy_palm = oy - py
        dist = (dx_tip * dx_tip + dy_tip * dy_tip) ** 0.5
        # Cache palm-based dx/dy for CONFIRM_GRAB direction prompts.
        self._last_dx = dx_palm
        self._last_dy = dy_palm
        obj_bbox = self._obj_bbox_norm(eff_obj, w, h)
        iou = _bbox_iou(obj_bbox, hand.bbox_norm)
        # Strict "near" test: fingertip inside object bbox, OR very close to center,
        # OR substantial hand-bbox overlap (e.g. hand wrapping around the object).
        # Note: 2D-near alone is NOT sufficient — a hand behind the object also
        # projects as near. We combine this with occlusion evidence below.
        tip_inside = (obj_bbox[0] <= tx <= obj_bbox[2]
                      and obj_bbox[1] <= ty <= obj_bbox[3])
        near_2d = (dist < self.cfg.grab_radius
                   or tip_inside
                   or iou >= self.cfg.iou_threshold)

        # ---- 4b) direct contact (no baseline required) ----
        # When the fingertip is inside the object bbox AND the hand bbox
        # overlaps the object bbox enough, the hand is touching the front
        # of the object. Strong overlap alone also counts (hand wrapped).
        # Without this path, "touching from the front" was being misread
        # as "hand is behind the object" because conf/area drop hadn't
        # crossed the occlusion ratio.
        direct_contact = (
            (tip_inside and iou >= self.cfg.direct_contact_iou)
            or iou >= self.cfg.direct_contact_iou_strong
        )

        # ---- 5) partial-occlusion evidence (only if baseline is ready) ----
        baseline_ready = self._baseline_n >= self.cfg.baseline_min_samples
        partially_occluded = False
        if fresh and baseline_ready:
            area_norm = obj.area / float(w * h)
            if obj.conf < self.cfg.grab_conf_drop_ratio * self._baseline_conf:
                partially_occluded = True
            if area_norm < self.cfg.grab_area_drop_ratio * self._baseline_area:
                partially_occluded = True
        occluded = fully_occluded or partially_occluded

        # ---- 6) grab decision (streak-gated) ----
        # Accept either the original (near_2d AND occluded) path or the new
        # direct_contact path. Both are streak-gated against YOLO flicker.
        if direct_contact or (near_2d and occluded):
            self._grab_streak += 1
        else:
            self._grab_streak = 0

        if self._grab_streak >= self.cfg.grab_frames_required:
            via = "direct" if direct_contact else (
                "full-occlusion" if fully_occluded else "partial-occlusion"
            )
            self._emit(
                f"[FSM/HAND] 似乎已接觸 {self.target_zh}！ "
                f"(dist={dist:.3f}, IoU={iou:.2f}, streak={self._grab_streak}, "
                f"via={via})"
            )
            self._goto(FGState.CONFIRM_GRAB)
            return

        # 2D vicinity but object still fully visible AND not direct contact
        # => hand is BEHIND the object. Only trust this once we have a baseline.
        if near_2d and baseline_ready and not occluded and not direct_contact:
            if now - self.t.last_behind_prompt >= self.cfg.behind_prompt_interval_s:
                self.t.last_behind_prompt = now
                self._emit(
                    f"[FSM/HAND] 你的手在 {self.target_zh} 後方，請把手往前伸再靠過去 "
                    f"| conf={eff_obj.conf:.2f}/{self._baseline_conf:.2f} "
                    f"area={(eff_obj.area/float(w*h)):.4f}/{self._baseline_area:.4f}"
                )
            return

        # ---- 7) direction prompts (existing behaviour) ----
        if now - self.t.last_hand_prompt < self.cfg.hand_prompt_interval_s:
            return
        self.t.last_hand_prompt = now

        # Use palm-based dx/dy here too — feels more natural to the user
        # ("hand should move right" maps to palm moving right, not fingertip).
        tol = self.cfg.hand_tolerance
        msgs: list[str] = []
        if dx_palm > tol:
            msgs.append("手向右移")
        elif dx_palm < -tol:
            msgs.append("手向左移")
        if dy_palm > tol:
            msgs.append("手向下移")
        elif dy_palm < -tol:
            msgs.append("手向上移")
        if not msgs:
            msgs.append("位置已接近，慢慢靠過去")
        self._emit(f"[FSM/HAND] {'，'.join(msgs)} | d={dist:.3f} IoU={iou:.2f}")

    def _tick_confirm(self, obj: Optional[Detection],
                      hand: Optional[HandResult], w: int, h: int) -> None:
        """Wait for user voice confirmation. Keep refining direction hint."""
        now = time.monotonic()

        # Refresh dx/dy from palm (not fingertip) so the prompt
        # 「物品在你手的左/右/上/下方」 matches user intuition.
        if obj is not None and hand is not None:
            ox = obj.cx / float(w)
            oy = obj.cy / float(h)
            self._last_dx = ox - hand.palm_x
            self._last_dy = oy - hand.palm_y

        # Auto-timeout back to GUIDING_HAND so we don't get stuck forever.
        if now - self.t.confirm_started >= self.cfg.confirm_timeout_s:
            self._emit("[FSM/CONFIRM] 等待回應逾時，回到手部導引")
            self._goto(FGState.GUIDING_HAND)
            return

        if now - self.t.last_confirm_prompt < self.cfg.confirm_prompt_interval_s:
            return
        self.t.last_confirm_prompt = now

        side = self._direction_text(self._last_dx, self._last_dy)
        if side:
            self._emit(
                f"[FSM/CONFIRM] 請問是否拿到了？沒有的話，{self.target_zh} 在你手的{side}"
            )
        else:
            self._emit("[FSM/CONFIRM] 請問是否拿到了？")

    @staticmethod
    def _direction_text(dx: float, dy: float, tol: float = 0.04) -> str:
        """Return 「左/右/上/下/左上…」 describing where object is relative to hand.

        dx>0 => object is to the RIGHT of the hand. dy>0 => object is BELOW.
        """
        h_part = ""
        v_part = ""
        if dx > tol:
            h_part = "右邊"
        elif dx < -tol:
            h_part = "左邊"
        if dy > tol:
            v_part = "下方"
        elif dy < -tol:
            v_part = "上方"
        if h_part and v_part:
            # 「左上方」「右下方」 — drop 「邊」 from horizontal half
            return h_part.replace("邊", "") + v_part
        return h_part or v_part

    @staticmethod
    def _obj_bbox_norm(d: Detection, w: int, h: int) -> tuple[float, float, float, float]:
        return (d.x1 / float(w), d.y1 / float(h), d.x2 / float(w), d.y2 / float(h))

    # ---------------- helpers ----------------
    def _find_target(self, dets: List[Detection]) -> Optional[Detection]:
        if not self.target_en:
            return None
        cands = [d for d in dets if d.label.lower() == self.target_en.lower()]
        if not cands:
            return None
        return max(cands, key=lambda d: d.conf)

    def _goto(self, s: FGState) -> None:
        if s == self.state:
            return
        self.state = s
        if s == FGState.GUIDING_HAND:
            # fresh baseline per grab attempt so occlusion evidence is meaningful
            self._reset_grab_baseline()
        elif s == FGState.CONFIRM_GRAB:
            now = time.monotonic()
            self.t.confirm_started = now
            self.t.last_confirm_prompt = 0.0  # prompt immediately on entry
        elif s == FGState.GRAB_SUCCESS:
            self.t.grab_success_start = time.monotonic()
            self._emit(f"[FSM] 進入 GRAB_SUCCESS — 已成功接觸 {self.target_zh}！")
        elif s == FGState.WAITING_FOR_COMMAND:
            self._emit("[FSM] 已重置，等待下一個指令")

    def _reset_grab_baseline(self) -> None:
        self._last_obj = None
        self._last_obj_t = 0.0
        self._baseline_conf = 0.0
        self._baseline_area = 0.0
        self._baseline_n = 0
        self._baseline_announced = False
        self._grab_streak = 0

    def _reset(self) -> None:
        self.state = FGState.WAITING_FOR_COMMAND
        self.target_zh = None
        self.target_en = None
        self.center_streak = 0
        self.t = _Timers()
        self._reset_grab_baseline()
        self._emit("[FSM] WAITING_FOR_COMMAND — 請說「想要找 XX」")

    @staticmethod
    def _default_emit(msg: str) -> None:
        print(msg, flush=True)
