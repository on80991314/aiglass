"""
AI Smart Glasses - Phase 1
Edge-side WebSocket server that receives JPEG frames from the ESP32-S3
and displays them with OpenCV to verify the link.

Run:
    pip install -r ../requirements.txt
    python ws_server.py --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Optional

import cv2
import numpy as np
import websockets


log = logging.getLogger("edge.ws")


def _ws_path(ws) -> str:
    # websockets >=13 moved path to ws.request.path; older versions had ws.path.
    req = getattr(ws, "request", None)
    if req is not None and hasattr(req, "path"):
        return req.path
    return getattr(ws, "path", "/")


# optional YOLO overlay — enabled via --yolo
_yolo_detector = None

def _yolo_enable(weights: str, device: str, conf: float) -> None:
    global _yolo_detector
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from vision.detector import YoloDetector  # noqa: E402
    _yolo_detector = YoloDetector(weights=weights, device=device, conf=conf)
    log.info("YOLO enabled: %s on %s (conf>=%.2f)", weights, device, conf)

def _yolo_draw(frame: np.ndarray) -> np.ndarray:
    if _yolo_detector is None:
        return frame
    try:
        dets = _yolo_detector.infer(frame)
    except Exception as e:
        log.warning("yolo infer failed: %s", e)
        return frame
    for d in dets:
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{d.label} {d.conf:.2f}",
                    (d.x1, max(18, d.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


class LatestFrame:
    """Single-slot frame buffer: producer overwrites, consumer reads newest."""

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._client: Optional[str] = None
        self._ts: float = 0.0
        self._lock = asyncio.Lock()

    async def set(self, frame: np.ndarray, client: str) -> None:
        async with self._lock:
            self._frame = frame
            self._client = client
            self._ts = time.monotonic()

    def snapshot(self) -> tuple[Optional[np.ndarray], Optional[str], float]:
        return self._frame, self._client, self._ts


def decode_jpeg(buf: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


async def handle_client(ws, latest: LatestFrame) -> None:
    peer = f"{ws.remote_address[0]}:{ws.remote_address[1]}"
    log.info("client connected: %s  path=%s", peer, _ws_path(ws))

    frames = 0
    bytes_rx = 0
    window_start = time.monotonic()

    try:
        async for msg in ws:
            if not isinstance(msg, (bytes, bytearray)):
                # Phase 1: ignore text messages; later phases will use JSON control.
                continue

            frame = decode_jpeg(bytes(msg))
            if frame is None:
                log.warning("bad JPEG from %s (%d bytes)", peer, len(msg))
                continue

            await latest.set(frame, peer)
            frames += 1
            bytes_rx += len(msg)

            now = time.monotonic()
            if now - window_start >= 2.0:
                fps = frames / (now - window_start)
                kbps = (bytes_rx * 8 / 1024) / (now - window_start)
                log.info("[%s] %.1f fps, %.0f kbps, %dx%d",
                         peer, fps, kbps, frame.shape[1], frame.shape[0])
                frames = 0
                bytes_rx = 0
                window_start = now
    except websockets.ConnectionClosed:
        pass
    finally:
        log.info("client disconnected: %s", peer)


async def display_loop(latest: LatestFrame, window: str, stop_event: asyncio.Event) -> None:
    """Runs on the main thread via asyncio; cv2 GUI must not move across threads on Windows."""
    last_ts = 0.0
    have_frame = False
    placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for ESP32-S3...", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(window, placeholder)

    while not stop_event.is_set():
        frame, client, ts = latest.snapshot()
        if frame is not None and ts != last_ts:
            last_ts = ts
            have_frame = True
            overlay = frame.copy()
            overlay = _yolo_draw(overlay)
            label = f"{client}  {overlay.shape[1]}x{overlay.shape[0]}"
            cv2.putText(overlay, label, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(window, overlay)
        elif not have_frame:
            cv2.imshow(window, placeholder)
        # else: no new frame this tick but we already have one on screen — leave it alone

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            stop_event.set()
            break
        await asyncio.sleep(0.01)

    cv2.destroyAllWindows()


async def main_async(host: str, port: int) -> None:
    latest = LatestFrame()
    stop_event = asyncio.Event()

    async def _handler(ws) -> None:
        await handle_client(ws, latest)

    server = await websockets.serve(
        _handler,
        host=host,
        port=port,
        max_size=8 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20,
    )
    log.info("WebSocket server listening on ws://%s:%d", host, port)

    display_task = asyncio.create_task(display_loop(latest, "AI Smart Glasses / Phase 1", stop_event))

    try:
        await stop_event.wait()
    finally:
        server.close()
        await server.wait_closed()
        display_task.cancel()
        try:
            await display_task
        except asyncio.CancelledError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge WebSocket image-stream receiver")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--yolo", action="store_true",
                        help="Overlay YOLOv8 detections (downloads yolov8n.pt on first run)")
    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument("--yolo-device", default="cpu")
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s  %(levelname)-5s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.yolo:
        _yolo_enable(args.yolo_weights, args.yolo_device, args.yolo_conf)

    try:
        asyncio.run(main_async(args.host, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
