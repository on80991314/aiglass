"""
STT 診斷腳本 — 執行方式：
    python diag_stt.py
 
會依序測試三件事：
  Step 1: 麥克風能不能錄音、VAD 有沒有觸發
  Step 2: 把錄到的 WAV 存到 diag_output.wav，讓你自己聽確認
  Step 3: 把那段 WAV 送 Groq，看 raw 回傳
 
每一步都會印出詳細 log，對照後就能知道問題在哪一層。
"""
 
import io
import logging
import os
import time
import wave
 
import numpy as np
 
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("diag")
 
# ─── 設定 ──────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
RECORD_S      = 5          # 錄幾秒
ENERGY_THRESH = 500        # 你 mic.py 裡的預設值\
api_key = os.getenv("GROQ_API_KEY")
OUTPUT_WAV    = "diag_output.wav"
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Step 1 & 2 — 錄音 + 能量分析
# ══════════════════════════════════════════════════════════════════════════════
def record_and_analyze() -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        log.error("sounddevice 未安裝：pip install sounddevice")
        raise
 
    log.info("═══ Step 1: 錄音 %ds，請開始說話… ═══", RECORD_S)
    audio = sd.rec(int(RECORD_S * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    pcm = audio[:, 0]
 
    # 能量分析 — 每 0.5s 一段
    chunk = SAMPLE_RATE // 2
    log.info("── 能量分析（每 0.5s）──")
    any_above = False
    for i in range(0, len(pcm), chunk):
        seg = pcm[i:i+chunk]
        energy = int(np.abs(seg).mean())
        bar = "█" * (energy // 50)
        flag = " ← VAD 觸發" if energy > ENERGY_THRESH else ""
        log.info("  %.1fs–%.1fs  energy=%5d  %s%s",
                 i/SAMPLE_RATE, (i+chunk)/SAMPLE_RATE, energy, bar, flag)
        if energy > ENERGY_THRESH:
            any_above = True
 
    if not any_above:
        log.warning(
            "⚠️  整段錄音能量都低於門檻 %d！\n"
            "   → 可能原因：麥克風沒選到、音量太低、或門檻太高。\n"
            "   → 建議：把 energy_thresh 調低（試試 200），或換用 webrtcvad。",
            ENERGY_THRESH,
        )
    else:
        log.info("✅ VAD 有觸發，麥克風收音正常。")
 
    # 存 WAV 讓你自己聽
    with wave.open(OUTPUT_WAV, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    log.info("錄音已存到 %s，請自行播放確認聲音是否清晰。", OUTPUT_WAV)
 
    return pcm
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — 送 Groq，印出 raw 結果
# ══════════════════════════════════════════════════════════════════════════════
def test_groq(pcm: np.ndarray):
    log.info("═══ Step 3: 送 Groq Whisper ═══")
 
    if not GROQ_API_KEY:
        log.error("GROQ_API_KEY 未設定！請先 export GROQ_API_KEY=...")
        return
 
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.astype(np.int16).tobytes())
    wav_bytes = buf.getvalue()
 
    from openai import OpenAI
    client = OpenAI(api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1")
 
    # ── 測試 A：有 prompt ────────────────────────────────────────────────────
    log.info("── 測試 A：有 prompt（你目前修正後的版本）")
    try:
        resp_a = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=("utt.wav", wav_bytes, "audio/wav"),
            language="zh",
            response_format="json",
            prompt="以下是台灣繁體中文的日常對話，請以繁體中文輸出，不要使用拼音或其他語言。",
            temperature=0.0,
        )
        log.info("  結果 A: %r", resp_a.text)
    except Exception as e:
        log.error("  錯誤 A: %s", e)
 
    # ── 測試 B：無 prompt（原始版本）────────────────────────────────────────
    log.info("── 測試 B：無 prompt（原始版本）")
    try:
        resp_b = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=("utt.wav", wav_bytes, "audio/wav"),
            language="zh",
            response_format="json",
        )
        log.info("  結果 B: %r", resp_b.text)
    except Exception as e:
        log.error("  錯誤 B: %s", e)
 
    # ── 測試 C：換 whisper-large-v3（非 turbo）───────────────────────────────
    log.info("── 測試 C：whisper-large-v3（非 turbo）+ prompt")
    try:
        resp_c = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=("utt.wav", wav_bytes, "audio/wav"),
            language="zh",
            response_format="json",
            prompt="以下是台灣繁體中文的日常對話，請以繁體中文輸出，不要使用拼音或其他語言。",
            temperature=0.0,
        )
        log.info("  結果 C: %r", resp_c.text)
    except Exception as e:
        log.error("  錯誤 C: %s", e)
 
    # ── 測試 D：直接讀 diag_output.wav 檔案送出（排除 in-memory 問題）────────
    log.info("── 測試 D：從存檔的 WAV 送出（排除 in-memory bytes 問題）")
    try:
        with open(OUTPUT_WAV, "rb") as f:
            resp_d = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=("utt.wav", f, "audio/wav"),
                language="zh",
                response_format="json",
                prompt="以下是台灣繁體中文的日常對話，請以繁體中文輸出，不要使用拼音或其他語言。",
                temperature=0.0,
            )
        log.info("  結果 D: %r", resp_d.text)
    except Exception as e:
        log.error("  錯誤 D: %s", e)
 
 
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("GROQ_API_KEY: %s", "已設定" if GROQ_API_KEY else "❌ 未設定")
    pcm = record_and_analyze()
    test_groq(pcm)
    log.info("═══ 診斷完成 ═══")
    log.info("請把上面所有輸出貼給 Claude，他會告訴你下一步怎麼修。")