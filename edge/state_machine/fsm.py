"""Top-level finite state machine for the glasses.

State transitions (Phase 3+ target):

  IDLE ──voice:nav──────► NAV
  IDLE ──voice:find─────► FIND
  IDLE ──voice:tr───────► TRANSLATE
  IDLE ──voice:cross────► CROSS_STREET    ← aiglass3 過馬路
  IDLE ──voice:tl───────► TRAFFIC_LIGHT   ← aiglass3 紅綠燈
  NAV  ──voice:cross────► CROSS_STREET    ← 可從導航中切換
  *    ──imu:fall────────► FALL_ALERT
  *    ──voice:cancel────► IDLE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .events import Event, Intent, State

log = logging.getLogger("fsm")


@dataclass
class Context:
    target_object: Optional[str] = None
    nav_destination: Optional[str] = None
    translate_lang: str = "zh-TW"
    extras: dict[str, Any] = field(default_factory=dict)


class GlassesFSM:
    def __init__(self, say: Callable[[str], Any]) -> None:
        self.state = State.IDLE
        self.ctx = Context()
        self.say = say  # callable: str -> None (queues TTS)

    # ------------- event entry points -------------
    def on_intent(self, intent: Intent, payload: dict[str, Any]) -> None:
        log.info("intent=%s payload=%s (state=%s)", intent.name, payload, self.state.name)

        # ── 全域取消 ─────────────────────────────────────────────────────────
        if intent == Intent.CANCEL:
            self._goto(State.IDLE, "已取消")
            return

        # ── 導航 ─────────────────────────────────────────────────────────────
        if intent == Intent.NAV_TO:
            self.ctx.nav_destination = payload.get("destination", "")
            self._goto(State.NAV, f"開始導航到{self.ctx.nav_destination}")
            return

        if intent == Intent.START_BLINDPATH_NAV:
            self._goto(State.NAV, "開始盲道導航")
            return

        if intent == Intent.STOP_NAV:
            if self.state == State.NAV:
                self._goto(State.IDLE, "已停止導航")
            return

        # ── 找物品 ───────────────────────────────────────────────────────────
        if intent == Intent.FIND_OBJECT:
            self.ctx.target_object = payload.get("object", "")
            self._goto(State.FIND, f"正在尋找{self.ctx.target_object}")
            return

        # ── 翻譯 ─────────────────────────────────────────────────────────────
        if intent == Intent.TRANSLATE_TO:
            self.ctx.translate_lang = payload.get("lang", "zh-TW")
            self._goto(State.TRANSLATE, f"開始翻譯模式，輸出語言{self.ctx.translate_lang}")
            return

        # ── 過馬路（aiglass3 整合）────────────────────────────────────────────
        if intent == Intent.START_CROSSING:
            self._goto(State.CROSS_STREET, "過馬路模式已啟動，正在尋找斑馬線")
            return

        if intent == Intent.STOP_CROSSING:
            if self.state == State.CROSS_STREET:
                self._goto(State.IDLE, "過馬路已結束")
            return

        # ── 紅綠燈偵測（aiglass3 整合）───────────────────────────────────────
        if intent == Intent.START_TRAFFIC_LIGHT:
            self._goto(State.TRAFFIC_LIGHT, "已啟動紅綠燈偵測")
            return

        if intent == Intent.STOP_TRAFFIC_LIGHT:
            if self.state == State.TRAFFIC_LIGHT:
                self._goto(State.IDLE, "已停止紅綠燈偵測")
            return

        # ── 視覺問答（aiglass3 整合，由上層 App 實際呼叫 Gemini/VLM）─────────
        if intent == Intent.VISUAL_QUERY:
            # 不切換狀態，交由 App 層觸發一次性的 VLM 描述
            log.info("VISUAL_QUERY: 交由 App 層處理")
            return

    def on_fall(self) -> None:
        if self.state == State.FALL_ALERT:
            return
        log.warning("fall detected")
        self._goto(State.FALL_ALERT, "偵測到跌倒，正在通知緊急聯絡人")

    def on_fall_cleared(self) -> None:
        if self.state == State.FALL_ALERT:
            self._goto(State.IDLE, "已取消跌倒警示")

    # ------------- helpers -------------
    def _goto(self, new_state: State, announce: str = "") -> None:
        if new_state != self.state:
            log.info("state %s -> %s", self.state.name, new_state.name)
            self.state = new_state
        if announce:
            self.say(announce)
