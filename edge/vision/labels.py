"""Single source of truth for object label translation.

Three things live here:
  * CN_TO_EN  — Chinese (zh-TW / zh-CN) noun -> YOLO English class name.
                Includes COCO classes plus extra everyday objects you may
                later train custom YOLO heads for (鑰匙/紅牛/可樂...).
  * EN_TO_ZH  — English class name -> primary Chinese display name.
                Used by the OpenCV overlay and the FSM voice prompts.
  * parse_find_command(text) — regex extractor for 「想要找 XX」 style
                Mandarin commands. Returns (zh, en) when matched.
"""

from __future__ import annotations

import re
from typing import Optional


# ============================================================
# CN -> EN  (used to translate STT result into YOLO label)
# ============================================================
# Tip: when adding entries, prefer to add several common synonyms that
# point at the same English label. Both 「杯子」 and 「馬克杯」 -> "cup".

CN_TO_EN: dict[str, str] = {
    # ---- COCO base classes ----
    # Both traditional (zh-TW) and simplified (zh-CN) variants are listed
    # for words that differ. Whisper sometimes outputs simplified even when
    # the speaker uses Mandarin from Taiwan; OpenCC normalisation in
    # audio/zh_normalizer.py is the primary defence, this is a safety net.
    "人": "person", "行人": "person",
    "腳踏車": "bicycle", "脚踏车": "bicycle", "自行車": "bicycle", "自行车": "bicycle",
    "機車": "motorcycle", "机车": "motorcycle", "摩托車": "motorcycle", "摩托车": "motorcycle",
    "汽車": "car", "汽车": "car", "車": "car", "车": "car", "車子": "car", "车子": "car",
    "公車": "bus", "公车": "bus", "巴士": "bus",
    "貨車": "truck", "货车": "truck", "卡車": "truck", "卡车": "truck",
    "飛機": "airplane", "飞机": "airplane", "客機": "airplane", "客机": "airplane",
    "船": "boat",
    "紅綠燈": "traffic light", "红绿灯": "traffic light",
    "停車號誌": "stop sign", "停车号志": "stop sign",
    "消防栓": "fire hydrant",
    "停車計時器": "parking meter", "停车计时器": "parking meter",
    "長椅": "bench", "长椅": "bench",
    "鳥": "bird", "鸟": "bird",
    "貓": "cat", "猫": "cat",
    "狗": "dog",
    "馬": "horse", "马": "horse",
    "羊": "sheep",
    "牛": "cow",
    "大象": "elephant",
    "熊": "bear",
    "斑馬": "zebra", "斑马": "zebra",
    "長頸鹿": "giraffe", "长颈鹿": "giraffe",
    "背包": "backpack",
    "手提包": "handbag", "包包": "handbag",
    "行李箱": "suitcase",
    "雨傘": "umbrella", "雨伞": "umbrella", "傘": "umbrella", "伞": "umbrella",
    "領帶": "tie", "领带": "tie",
    "飛盤": "frisbee", "飞盘": "frisbee",
    "滑雪板": "skis",
    "滑板": "skateboard",
    "衝浪板": "surfboard", "冲浪板": "surfboard",
    "球拍": "tennis racket", "網球拍": "tennis racket", "网球拍": "tennis racket",
    "水瓶": "bottle", "瓶子": "bottle", "礦泉水": "bottle", "矿泉水": "bottle",
    "紅酒杯": "wine glass", "红酒杯": "wine glass", "酒杯": "wine glass",
    "杯子": "cup", "馬克杯": "cup", "马克杯": "cup",
    "叉子": "fork",
    "刀子": "knife", "刀": "knife",
    "湯匙": "spoon", "汤匙": "spoon", "匙": "spoon",
    "碗": "bowl",
    "香蕉": "banana",
    "蘋果": "apple", "苹果": "apple",
    "三明治": "sandwich",
    "橘子": "orange", "柳橙": "orange",
    "青花菜": "broccoli", "花椰菜": "broccoli",
    "紅蘿蔔": "carrot", "红萝卜": "carrot", "胡蘿蔔": "carrot", "胡萝卜": "carrot",
    "熱狗": "hot dog", "热狗": "hot dog",
    "披薩": "pizza", "披萨": "pizza",
    "甜甜圈": "donut",
    "蛋糕": "cake",
    "椅子": "chair",
    "沙發": "couch", "沙发": "couch",
    "盆栽": "potted plant", "植物": "potted plant",
    "床": "bed",
    "餐桌": "dining table", "桌子": "dining table", "桌": "dining table",
    "馬桶": "toilet", "马桶": "toilet",
    "電視": "tv", "电视": "tv", "電視機": "tv", "电视机": "tv",
    "筆電": "laptop", "笔电": "laptop",
    "筆記型電腦": "laptop", "笔记型电脑": "laptop", "笔记本电脑": "laptop",
    "電腦": "laptop", "电脑": "laptop",
    "滑鼠": "mouse", "鼠標": "mouse", "鼠标": "mouse",
    "遙控器": "remote", "遥控器": "remote",
    "鍵盤": "keyboard", "键盘": "keyboard",
    "手機": "cell phone", "手机": "cell phone",
    "電話": "cell phone", "电话": "cell phone",
    "行動電話": "cell phone", "行动电话": "cell phone",
    "微波爐": "microwave", "微波炉": "microwave",
    "烤箱": "oven",
    "烤麵包機": "toaster", "烤面包机": "toaster",
    "水槽": "sink",
    "冰箱": "refrigerator",
    "書": "book", "书": "book", "書本": "book", "书本": "book",
    "時鐘": "clock", "时钟": "clock", "鐘": "clock", "钟": "clock",
    "花瓶": "vase",
    "剪刀": "scissors",
    "泰迪熊": "teddy bear", "熊娃娃": "teddy bear",
    "吹風機": "hair drier", "吹风机": "hair drier",
    "牙刷": "toothbrush",

    # ---- Extra everyday objects (require a custom-trained head) ----
    "鑰匙": "keys", "钥匙": "keys",
    "錢包": "wallet", "钱包": "wallet",
    "紅牛": "red_bull", "红牛": "red_bull",
    "可樂": "coke", "可乐": "coke",
    "雪碧": "sprite",
    "眼鏡": "glasses", "眼镜": "glasses",
    "口罩": "mask",
    "悠遊卡": "card", "悠游卡": "card",
    "信用卡": "card",
}


# ============================================================
# EN -> ZH  (primary Chinese display name)
# ============================================================
# Auto-built from CN_TO_EN by picking the FIRST Chinese word seen for
# each English label. Override per-label below if the auto pick isn't
# the most natural one for spoken prompts.

def _build_en_to_zh() -> dict[str, str]:
    seen: dict[str, str] = {}
    for zh, en in CN_TO_EN.items():
        seen.setdefault(en.lower(), zh)
    # Manual overrides for prompt clarity:
    seen["car"] = "汽車"
    seen["bottle"] = "水瓶"
    seen["cup"] = "杯子"
    seen["laptop"] = "筆電"
    seen["cell phone"] = "手機"
    seen["dining table"] = "桌子"
    seen["potted plant"] = "盆栽"
    seen["traffic light"] = "紅綠燈"
    seen["tv"] = "電視"
    return seen


EN_TO_ZH: dict[str, str] = _build_en_to_zh()


def to_en(zh: str) -> Optional[str]:
    """Best-effort Chinese -> English. Tries exact, then containment."""
    q = (zh or "").strip()
    if not q:
        return None
    if q in CN_TO_EN:
        return CN_TO_EN[q]
    for k, v in CN_TO_EN.items():
        if k in q:
            return v
    low = q.lower()
    if low in EN_TO_ZH:
        return low
    return None


def to_zh(en: str) -> str:
    """English label -> primary Chinese display name (falls back to input)."""
    if not en:
        return en
    return EN_TO_ZH.get(en.lower(), en)


# ============================================================
# Backwards-compat aliases (existing imports keep working)
# ============================================================
ZH_TO_COCO = CN_TO_EN
COCO_ZH = EN_TO_ZH


# ============================================================
# Mandarin "find X" command extractor
# ============================================================
_STRIP_PREFIXES = [
    "我的", "那個", "那个", "那支", "那隻", "那只",
    "一個", "一个", "一支", "一隻", "一只", "這個", "这个",
]

# Both 「尋找」 and 「找」 should match. Simplified 「帮我找」 also accepted as a
# safety net even though OpenCC normalisation should turn it into 「幫我找」
# upstream.
_PATTERNS = [
    re.compile(r"(?:想要?|要)?尋?找[一個個个支隻只]?(.+?)(?:[，。！？]|$)"),
    re.compile(r"[幫帮]我[尋寻]?找[一個個个支隻只]?(.+?)(?:[，。！？]|$)"),
    re.compile(r"[尋寻]?找(.+?)(?:[，。！？]|$)"),
]


def parse_find_command(text: str) -> tuple[str, str] | None:
    """Return (target_zh, target_en) if the utterance matches a find-command."""
    if not text:
        return None
    for pat in _PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        raw = m.group(1).strip()
        for pref in _STRIP_PREFIXES:
            if raw.startswith(pref):
                raw = raw[len(pref):].strip()
        if not raw:
            continue
        en = to_en(raw)
        if en:
            return raw, en
    return None
