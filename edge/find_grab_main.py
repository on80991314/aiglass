"""Backwards-compat shim.

The find-grab app moved to `apps/find_grab.py` and is now invoked via
`python edge/main.py find_grab ...`. This file forwards old commands so
existing scripts keep working.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.dotenv import load as load_dotenv  # noqa: E402
from utils.logging_setup import setup as setup_logging  # noqa: E402


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--log-level", default="INFO")
    from apps import find_grab as app_find_grab
    app_find_grab.add_args(p)
    args = p.parse_args()
    setup_logging(args.log_level)
    logging.getLogger("find_grab").info(
        "(deprecated entry point) prefer:  python edge/main.py find_grab ..."
    )
    app_find_grab.run(args)


if __name__ == "__main__":
    main()
