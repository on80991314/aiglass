"""Chinese-capable text renderer for OpenCV frames.

cv2.putText cannot render CJK glyphs (shows '?' or boxes). This module
draws via PIL using a system CJK font, then returns the BGR image.

Falls back gracefully to cv2.putText if no CJK font is found.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("text_renderer")


_CANDIDATE_FONTS = [
    r"C:\Windows\Fonts\msyh.ttc",      # Microsoft YaHei (zh-CN)
    r"C:\Windows\Fonts\msjh.ttc",      # Microsoft JhengHei (zh-TW)
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


_pil_font_cache: dict[int, "any"] = {}
_pil_ok: Optional[bool] = None
_font_path: Optional[str] = None


def _resolve_font() -> Optional[str]:
    global _font_path
    if _font_path is not None:
        return _font_path
    env = os.environ.get("CJK_FONT")
    if env and Path(env).exists():
        _font_path = env
        return _font_path
    for f in _CANDIDATE_FONTS:
        if Path(f).exists():
            _font_path = f
            return _font_path
    return None


def _get_font(size: int):
    global _pil_ok
    if _pil_ok is False:
        return None
    try:
        from PIL import ImageFont  # noqa: F401
    except ImportError:
        log.warning("Pillow not installed; CJK text will fall back to cv2.putText")
        _pil_ok = False
        return None
    _pil_ok = True
    if size in _pil_font_cache:
        return _pil_font_cache[size]
    path = _resolve_font()
    if path is None:
        log.warning("no CJK font found on this system; CJK text will fall back to cv2.putText")
        return None
    from PIL import ImageFont
    font = ImageFont.truetype(path, size)
    _pil_font_cache[size] = font
    return font


def put_text(frame_bgr: np.ndarray,
             text: str,
             org: tuple[int, int],
             font_size: int = 18,
             color_bgr: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw `text` at `org` (top-left). Returns the (possibly new) frame."""
    if not text:
        return frame_bgr
    font = _get_font(font_size)
    if font is None:
        cv2.putText(frame_bgr, text, (org[0], org[1] + font_size),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 32.0,
                    color_bgr, 1, cv2.LINE_AA)
        return frame_bgr
    from PIL import Image, ImageDraw
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(org, text, font=font, fill=color_rgb)
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return out


def measure(text: str, font_size: int = 18) -> tuple[int, int]:
    font = _get_font(font_size)
    if font is None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_size / 32.0, 1)
        return tw, th
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]
