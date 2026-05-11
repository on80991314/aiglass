"""Tiny .env loader so we don't pull in python-dotenv just for this.

Reads <repo>/.env and inserts each KEY=VALUE pair into os.environ
(without overwriting variables already set in the shell environment).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .paths import REPO_DIR

log = logging.getLogger("dotenv")


def load(env_path: Path | None = None) -> int:
    """Returns the number of variables loaded (or 0 if file missing)."""
    p = env_path or (REPO_DIR / ".env")
    if not p.exists():
        return 0
    n = 0
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
                n += 1
    except Exception as e:
        log.warning(".env load failed (%s): %s", p, e)
        return 0
    return n
