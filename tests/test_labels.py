"""Pure-function tests for vision/labels.py."""

from vision.labels import to_en, to_zh, parse_find_command


def test_to_en_exact():
    assert to_en("杯子") == "cup"
    assert to_en("筆電") == "laptop"
    assert to_en("鑰匙") == "keys"


def test_to_en_containment():
    assert to_en("我的杯子") == "cup"
    assert to_en("那支電話") == "cell phone"


def test_to_en_unknown():
    # No known Chinese substring -> None (use a non-noun phrase to avoid
    # hitting any containment match like 「船」 or 「人」).
    assert to_en("阿哈哈") is None
    assert to_en("") is None


def test_to_zh_roundtrip():
    assert to_zh("cup") == "杯子"
    assert to_zh("laptop") == "筆電"
    # unknown English label falls back to itself
    assert to_zh("xyz") == "xyz"


def test_parse_find_command():
    assert parse_find_command("我想要找杯子") == ("杯子", "cup")
    assert parse_find_command("幫我找筆電") == ("筆電", "laptop")
    assert parse_find_command("幫我找我的杯子") == ("杯子", "cup")
    assert parse_find_command("今天天氣不錯") is None
