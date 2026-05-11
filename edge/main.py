"""Single launcher for all edge-side apps.

Modes:
  find_grab  — 5-state object find-and-grab with mic STT, LLM intent,
               voice cues, MediaPipe hands, optional ESP32 video stream.
               (formerly edge/find_grab_main.py)

  full       — full multi-mode glasses orchestrator: NAV / FIND / TRANSLATE
               / FALL_ALERT, expects an ESP32-S3 connected over the WS hub
               and reads everything from .env via utils.config.
               (formerly edge/main.py)

Usage:
  python edge/main.py find_grab --mic --stt-backend groq --llm-intent --voice-cues
  python edge/main.py full
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sibling packages importable when run as `python edge/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.dotenv import load as load_dotenv  # noqa: E402
from utils.logging_setup import setup as setup_logging  # noqa: E402

log = logging.getLogger("main")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edge", description="AI Smart Glasses edge launcher")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="mode", required=True)

    # ---- find_grab ----
    from apps import find_grab as app_find_grab
    fg = sub.add_parser("find_grab", help="Object find-and-grab pipeline")
    app_find_grab.add_args(fg)
    fg.set_defaults(_run=app_find_grab.run)

    # ---- full ----
    from apps import full as app_full
    fl = sub.add_parser("full", help="Full multi-mode orchestrator (NAV/FIND/TRANSLATE/FALL)")
    app_full.add_args(fl)
    fl.set_defaults(_run=app_full.run)

    return p


def main() -> None:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    args._run(args)


if __name__ == "__main__":
    main()
