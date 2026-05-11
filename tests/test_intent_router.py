"""Regex-only intent router tests (LLM disabled)."""

from audio.intent_router import IntentRouter


def _r() -> IntentRouter:
    return IntentRouter(use_llm=False)


def test_hotword():
    assert _r().route("停下").kind == "HOTWORD"
    assert _r().route("不要說了").kind == "HOTWORD"


def test_cancel():
    intent = _r().route("算了不找了")
    assert intent.kind == "CANCEL"


def test_found_vs_not_yet():
    """Critical: 「還沒拿到」 contains 「拿到」, must NOT match FOUND."""
    assert _r().route("拿到了").kind == "FOUND"
    assert _r().route("還沒拿到").kind == "NOT_YET"
    assert _r().route("沒有拿到").kind == "NOT_YET"


def test_find_with_label():
    intent = _r().route("我想要找杯子")
    assert intent.kind == "FIND"
    assert intent.target_zh == "杯子"
    assert intent.target_en == "cup"


def test_unknown():
    assert _r().route("今天天氣不錯").kind == "NONE"
    assert _r().route("").kind == "NONE"
