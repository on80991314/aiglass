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
        log.info("🚀 成功載入 Groq 雲端語音辨識 (動態增益優化版)")

    def transcribe_pcm(self, pcm16_mono: np.ndarray, sample_rate: int = 16000, is_hardware_mic: bool = False) -> str:
        """辨識 PCM 語音資料。
        
        Args:
            pcm16_mono: 單聲道 16-bit PCM 陣列。
            sample_rate: 音訊採樣率。
            is_hardware_mic: 若為 True 代表音訊來自 ESP32 眼鏡端，開啟消除直流偏移與 5 倍放大；
                             若為 False 代表來自 PC 本地麥克風測試，保持正常增益防止破音。
        """
        if self.client is None or len(pcm16_mono) == 0:
            return ""

        try:
            # 1. 將音訊轉為浮點數，準備進行數位訊號處理
            audio_float = pcm16_mono.astype(np.float32)

            # 2. 依據來源動態決定增益，解決 PC 測試時破音（Clipping）導致 API 無法辨識的問題
            if is_hardware_mic:
                # 消除直流偏移 (DC Offset) - 解決 ESP32 常見的硬體電流聲
                audio_float = audio_float - np.mean(audio_float)
                # 強制放大音量 (Gain) - 眼鏡端微型麥克風放大 5 倍
                audio_float = audio_float * 5.0
            else:
                # PC 麥克風音量正常，不進行放大，保持 1.0 增益
                audio_float = audio_float * 1.0

            # 4. 防止音量過大破音 (Clipping)，並轉回 16-bit PCM
            audio_float = np.clip(audio_float, -32768, 32767)
            audio_data = audio_float.astype(np.int16)

            # 🎯 5. [超級除錯神器] 將這段處理過的聲音存成 WAV 檔！
            with wave.open("debug_mic.wav", 'wb') as debug_wav:
                debug_wav.setnchannels(1)
                debug_wav.setsampwidth(2)
                debug_wav.setframerate(sample_rate)
                debug_wav.writeframes(audio_data.tobytes())

            # 6. 在記憶體中建立傳給 Groq 的檔案，並正確取得 bytes 資料
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            wav_bytes = wav_io.getvalue()

            # 7. 呼叫 Groq API 
            # 修正：file 參數必須使用正確的三元組元組 (filename, bytes, content_type) 傳入，確保 SDK 能正確解析
            transcription = self.client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes, "audio/wav"),
                model="whisper-large-v3",
                prompt="以下是一段使用者對智慧眼鏡下達的中文語音指令：",
                response_format="json",
                language="zh",
                temperature=0.0
            )
            
            text = transcription.text.strip()
            
            # 8. 修正 Whisper 經典幻覺過濾機制
            # 修正：將 原本的 `in` 模糊包含判斷，改為 `set` 的完全相等判斷。
            # 避免當識別結果包含「好」或「喔」等常用字時，導致正常指令被誤殺清空。
            hallucinations = {
                "喔", "好", "謝謝", "別忘了", "我們下次見", 
                "謝謝觀看", "謝謝觀看,下次見!", "謝謝觀看，下次見！",
                "這就是我所謂的雜音。", "請忽略背景雜音"
            }
            if text in hallucinations or len(text) <= 1:
                return ""
                
            return to_traditional(text)
            
        except Exception as e:
            log.error(f"Groq 辨識失敗: {e}")
            return ""

    def _ensure_loaded(self):
        pass