"""Top-level finite state machine for the glasses.

State transitions (Phase 3 target):

  IDLE ──voice:nav────► NAV
  IDLE ──voice:find───► FIND
  IDLE ──voice:tr────-► TRANSLATE
  *    ──imu:fall────► FALL_ALERT
  *    ──voice:cancel► IDLE
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
        if intent == Intent.CANCEL:
            self._goto(State.IDLE, "已取消")
            return
        if intent == Intent.NAV_TO:
            self.ctx.nav_destination = payload.get("destination", "")
            self._goto(State.NAV, f"開始導航到{self.ctx.nav_destination}")
            return
        if intent == Intent.FIND_OBJECT:
            self.ctx.target_object = payload.get("object", "")
            self._goto(State.FIND, f"正在尋找{self.ctx.target_object}")
            return
        if intent == Intent.TRANSLATE_TO:
            self.ctx.translate_lang = payload.get("lang", "zh-TW")
            self._goto(State.TRANSLATE, f"開始翻譯模式，輸出語言{self.ctx.translate_lang}")
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
