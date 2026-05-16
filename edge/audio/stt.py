import os
import io
import wave
import logging
import numpy as np
from groq import Groq
from audio.zh_normalizer import to_traditional

log = logging.getLogger("stt")

class WhisperSTT:
    def __init__(self, model_size_or_path: str = "base", language: str = "zh", device: str = "cpu"):
        self.language = language
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            log.error("⚠️ 找不到 GROQ_API_KEY，請確認 .env 檔案是否設定正確！")
            
        self.client = Groq(api_key=self.api_key)
        log.info("🚀 成功載入 Groq 雲端語音辨識 (抗噪放大版)")

    def transcribe_pcm(self, pcm16_mono: np.ndarray, sample_rate: int = 16000) -> str:
        if self.client is None:
            return ""

        try:
            # 1. 將音訊轉為浮點數，準備進行數位訊號處理
            audio_float = pcm16_mono.astype(np.float32)

            # 2. 消除直流偏移 (DC Offset) - 解決 ESP32 常見的硬體電流聲
            audio_float = audio_float - np.mean(audio_float)

            # 3. 強制放大音量 (Gain) - 放大 5 倍！
            audio_float = audio_float * 5.0

            # 4. 防止音量過大破音 (Clipping)，並轉回 16-bit PCM
            audio_float = np.clip(audio_float, -32768, 32767)
            audio_data = audio_float.astype(np.int16)

            # 🎯 5. [超級除錯神器] 將這段處理過的聲音存成 WAV 檔！
            # 這樣你就可以在資料夾裡點開來，親耳聽聽看眼鏡到底錄到了什麼
            with wave.open("debug_mic.wav", 'wb') as debug_wav:
                debug_wav.setnchannels(1)
                debug_wav.setsampwidth(2)
                debug_wav.setframerate(16000)
                debug_wav.writeframes(audio_data.tobytes())

            # 6. 在記憶體中建立傳給 Groq 的檔案
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data.tobytes())
            
            wav_io.seek(0)
            wav_io.name = "audio.wav"

            # 7. 呼叫 Groq API (換一個自然一點的 Prompt，避免它又當成對白唸出來)
            transcription = self.client.audio.transcriptions.create(
                file=(wav_io.name, wav_io.read()),
                model="whisper-large-v3",
                prompt="以下是一段使用者對智慧眼鏡下達的中文語音指令：",
                response_format="json",
                language="zh",
                temperature=0.0
            )
            
            text = transcription.text.strip()
            
            # 8. 無情過濾掉所有的 Whisper 經典幻覺
            hallucinations = [
                "喔", "好", "謝謝", "別忘了", "我們下次見", 
                "謝謝觀看", "謝謝觀看,下次見!", "謝謝觀看，下次見！",
                "這就是我所謂的雜音。", "請忽略背景雜音"
            ]
            if any(h in text for h in hallucinations) or len(text) <= 1:
                return ""
                
            return to_traditional(text)
            
        except Exception as e:
            log.error(f"Groq 辨識失敗: {e}")
            return ""

    def _ensure_loaded(self):
        pass