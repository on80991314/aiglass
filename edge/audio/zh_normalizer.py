"""Chinese text normalizer — collapses simplified -> traditional.

Whisper / Groq Llama / Gemini all sometimes emit simplified characters even
when prompted in Mandarin Taiwan. Downstream code (labels.py, intent regex,
TTS) is built around traditional characters, so EVERY string that came from
a remote model passes through `to_traditional()` before being acted on.

If `opencc` is missing we log an ERROR (not silent) so the user notices —
the pipeline still runs but simplified will leak through.
    pip install opencc-python-reimplemented
"""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("zh_normalizer")

_cc = None
_tried_load = False


def _ensure_cc():
    global _cc, _tried_load
    if _tried_load:
        return _cc
    _tried_load = True
    try:
        from opencc import OpenCC
        _cc = OpenCC("s2t")
        log.info("OpenCC s2t loaded — simplified->traditional normalisation active")
    except Exception as e:
        log.error("opencc NOT installed (%s) — simplified Chinese will leak through. "
                  "Run:  pip install opencc-python-reimplemented", e)
        _cc = None
    return _cc


# Eager-load on import so the warning shows up at startup, not on the first
# utterance. Cheap (one tiny dictionary).
_ensure_cc()


def to_traditional(text: Optional[str]) -> str:
    """Best-effort simplified -> traditional conversion. Returns text
    unchanged if OpenCC isn't installed or conversion fails. Idempotent —
    safe to apply multiple times."""
    if not text:
        return text or ""
    cc = _ensure_cc()
    if cc is None:
        return text
    try:
        return cc.convert(text)
    except Exception:
        return text


def normalize_obj(obj: Any) -> Any:
    """Recursively walk dict/list/tuple and convert every string value to
    traditional Chinese. Used to scrub LLM JSON responses in one shot."""
    if isinstance(obj, str):
        return to_traditional(obj)
    if isinstance(obj, dict):
        return {k: normalize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(normalize_obj(v) for v in obj)
    return obj
