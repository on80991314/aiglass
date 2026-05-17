from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class State(Enum):
    IDLE = auto()
    NAV = auto()            # walking directions + crosswalk assist
    FIND = auto()           # locate a named object
    TRANSLATE = auto()      # ambient or dialog translation
    FALL_ALERT = auto()     # confirmed fall, notifying contact
    CROSS_STREET = auto()   # 過馬路模式（斑馬線對齊 → 紅綠燈判定 → 通行引導）
    TRAFFIC_LIGHT = auto()  # 單純紅綠燈偵測模式（不走完整過馬路流程）


class Intent(Enum):
    NONE = auto()
    NAV_TO = auto()
    FIND_OBJECT = auto()
    TRANSLATE_TO = auto()
    CANCEL = auto()
    CHAT = auto()
    # ── aiglass3 整合新增 ──────────────────────────────────────────────────
    START_CROSSING = auto()       # 開始過馬路（對齊斑馬線 → 等綠燈 → 通行）
    STOP_CROSSING = auto()        # 結束過馬路
    START_BLINDPATH_NAV = auto()  # 開始盲道導航
    STOP_NAV = auto()             # 停止導航
    START_TRAFFIC_LIGHT = auto()  # 啟動紅綠燈偵測
    STOP_TRAFFIC_LIGHT = auto()   # 停止紅綠燈偵測
    VISUAL_QUERY = auto()         # 幫我看看這是什麼


@dataclass
class Event:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
