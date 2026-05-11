"""Coordination object so the microphone listener can be temporarily
suspended while the speaker is active.

Without this, voice cues like 「向左」 are picked up by the microphone,
re-transcribed by Whisper, and fed back into the FSM as if the user had
said them — a self-wakeup loop that confuses the state machine.

Usage:
    gate = MicGate()
    cues = VoiceCues(gate=gate)        # cues call gate.suspend(seconds)
    mic = MicListener(..., gate=gate)  # mic checks gate.is_suspended()
"""

from __future__ import annotations

import threading
import time


class MicGate:
    def __init__(self) -> None:
        self._suspend_until: float = 0.0
        # Small extra buffer after audio nominally ends, to cover speaker
        # decay / room reverb. 200 ms is enough for indoor playback.
        self.tail_padding_s: float = 0.20
        self._lock = threading.Lock()

    def suspend(self, duration_s: float) -> None:
        """Mute the microphone for at least `duration_s` more seconds."""
        if duration_s <= 0:
            return
        with self._lock:
            target = time.monotonic() + duration_s + self.tail_padding_s
            if target > self._suspend_until:
                self._suspend_until = target

    def resume(self) -> None:
        with self._lock:
            self._suspend_until = 0.0

    def is_suspended(self) -> bool:
        return time.monotonic() < self._suspend_until

    def remaining_s(self) -> float:
        return max(0.0, self._suspend_until - time.monotonic())
