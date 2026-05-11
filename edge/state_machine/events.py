from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class State(Enum):
    IDLE = auto()
    NAV = auto()          # walking directions + crosswalk assist
    FIND = auto()         # locate a named object
    TRANSLATE = auto()    # ambient or dialog translation
    FALL_ALERT = auto()   # confirmed fall, notifying contact


class Intent(Enum):
    NONE = auto()
    NAV_TO = auto()
    FIND_OBJECT = auto()
    TRANSLATE_TO = auto()
    CANCEL = auto()
    CHAT = auto()


@dataclass
class Event:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
