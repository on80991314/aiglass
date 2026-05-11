"""Dump TTS output to a wav so you can verify it before streaming to hardware.
Usage: python -m edge.tests.test_tts "你好"
"""

import asyncio
import sys
import wave

sys.path.insert(0, __file__.rsplit("edge", 1)[0] + "edge")

from audio.tts import TTS  # noqa: E402


async def main(text: str):
    tts = TTS(backend="pyttsx3")
    pcm = await tts.synthesize(text)
    with wave.open("tts_out.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(tts.sample_rate)
        w.writeframes(pcm.tobytes())
    print(f"wrote tts_out.wav ({len(pcm)} samples @ {tts.sample_rate} Hz)")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "測試，我是智慧眼鏡"))
