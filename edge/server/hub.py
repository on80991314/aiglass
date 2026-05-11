"""Multi-channel WebSocket hub (Phase 3+).

Protocol on a single WS connection per device:

  BINARY messages start with 1 tag byte:
    0x01  JPEG video frame
    0x02  raw 16-bit LE mono PCM audio chunk from mic, prefix 4B LE sample_rate
    0x03  (server -> device) 16-bit LE mono PCM TTS chunk, prefix 4B LE sample_rate
    0xFF  backward-compat: raw JPEG, no tag byte (what Phase 1 streamer emits)

  TEXT messages are JSON:
    {"type":"imu","t":..,"ax":..,"ay":..,"az":..,"gx":..,"gy":..,"gz":..}
    {"type":"event","name":"fall","t":..}
    (server -> device) {"type":"cmd","name":"set_mode","mode":"NAV"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import numpy as np
import websockets

log = logging.getLogger("hub")

TAG_JPEG = 0x01
TAG_AUDIO_UP = 0x02
TAG_AUDIO_DOWN = 0x03


FrameCb = Callable[[str, np.ndarray], Awaitable[None]]
AudioCb = Callable[[str, np.ndarray, int], Awaitable[None]]
JsonCb = Callable[[str, dict], Awaitable[None]]


@dataclass
class DeviceSession:
    peer: str
    ws: object  # websockets connection (ServerConnection in v13+, WebSocketServerProtocol in older)
    meta: dict = field(default_factory=dict)


class Hub:
    def __init__(self,
                 on_frame: Optional[FrameCb] = None,
                 on_audio: Optional[AudioCb] = None,
                 on_json: Optional[JsonCb] = None) -> None:
        self.on_frame = on_frame
        self.on_audio = on_audio
        self.on_json = on_json
        self.sessions: dict[str, DeviceSession] = {}

    # ------------- outbound -------------
    async def broadcast_json(self, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False)
        await asyncio.gather(
            *(s.ws.send(data) for s in self.sessions.values()),
            return_exceptions=True,
        )

    async def send_tts_pcm(self, peer: str, pcm16: np.ndarray, sample_rate: int = 16000) -> None:
        s = self.sessions.get(peer)
        if s is None:
            log.warning("send_tts_pcm: no session %s", peer)
            return
        header = bytes([TAG_AUDIO_DOWN]) + struct.pack("<I", sample_rate)
        # ship in ~200ms slices so the MCU can start playback early
        chunk = sample_rate // 5
        for i in range(0, len(pcm16), chunk):
            slice_ = pcm16[i:i + chunk].astype(np.int16).tobytes()
            await s.ws.send(header + slice_)

    async def send_cmd(self, peer: str, name: str, **params) -> None:
        s = self.sessions.get(peer)
        if s is None:
            return
        await s.ws.send(json.dumps({"type": "cmd", "name": name, **params}, ensure_ascii=False))

    # ------------- server -------------
    async def serve(self, host: str, port: int) -> None:
        async def _handler(ws) -> None:
            await self._accept(ws)

        srv = await websockets.serve(
            _handler, host=host, port=port,
            max_size=8 * 1024 * 1024,
            ping_interval=20, ping_timeout=20,
        )
        log.info("hub listening on ws://%s:%d", host, port)
        await srv.wait_closed()

    async def _accept(self, ws) -> None:
        peer = f"{ws.remote_address[0]}:{ws.remote_address[1]}"
        self.sessions[peer] = DeviceSession(peer=peer, ws=ws)
        log.info("device connected: %s", peer)
        try:
            async for msg in ws:
                try:
                    await self._dispatch(peer, msg)
                except Exception as e:
                    log.exception("dispatch error: %s", e)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.sessions.pop(peer, None)
            log.info("device disconnected: %s", peer)

    async def _dispatch(self, peer: str, msg) -> None:
        if isinstance(msg, str):
            try:
                obj = json.loads(msg)
            except json.JSONDecodeError:
                log.warning("bad json from %s: %s", peer, msg[:120])
                return
            if self.on_json:
                await self.on_json(peer, obj)
            return

        data = bytes(msg)
        if not data:
            return

        # Backward-compat: Phase-1 firmware sends raw JPEG (no tag byte).
        if data[:2] == b"\xff\xd8":
            await self._dispatch_jpeg(peer, data)
            return

        tag = data[0]
        body = data[1:]
        if tag == TAG_JPEG:
            await self._dispatch_jpeg(peer, body)
        elif tag == TAG_AUDIO_UP:
            if len(body) < 4:
                return
            sr = struct.unpack("<I", body[:4])[0]
            pcm = np.frombuffer(body[4:], dtype=np.int16)
            if self.on_audio:
                await self.on_audio(peer, pcm, sr)
        else:
            log.debug("unknown binary tag 0x%02x from %s", tag, peer)

    async def _dispatch_jpeg(self, peer: str, jpeg: bytes) -> None:
        if not self.on_frame:
            return
        import cv2
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        await self.on_frame(peer, frame)
