"""Speech / event throttling — avoids the glasses talking over itself."""

from __future__ import annotations

import time


class SpeechThrottle:
    """Two-tier gate:
       * any-message gap  (min_gap)     -> minimum silence between sentences
       * same-message gap (repeat_gap)  -> suppresses repeating the same line
    """

    def __init__(self, min_gap: float = 3.0, repeat_gap: float = 7.0) -> None:
        self.min_gap = min_gap
        self.repeat_gap = repeat_gap
        self._last_any: float = 0.0
        self._last_text: dict[str, float] = {}

    def allow(self, text: str) -> bool:
        now = time.monotonic()
        if now - self._last_any < self.min_gap:
            return False
        prev = self._last_text.get(text, 0.0)
        if now - prev < self.repeat_gap:
            return False
        self._last_text[text] = now
        self._last_any = now
        # light prune
        if len(self._last_text) > 64:
            cutoff = now - self.repeat_gap * 2
            self._last_text = {k: v for k, v in self._last_text.items() if v > cutoff}
        return True

    def mark_spoken(self, text: str) -> None:
        now = time.monotonic()
        self._last_any = now
        self._last_text[text] = now
