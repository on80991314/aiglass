"""Project path constants.

One source of truth so YOLO weights, MediaPipe models, voice cues etc.
can be found regardless of the cwd that launched python.
"""

from __future__ import annotations

from pathlib import Path

# This file lives at  <repo>/edge/utils/paths.py
EDGE_DIR: Path = Path(__file__).resolve().parent.parent
REPO_DIR: Path = EDGE_DIR.parent
MODELS_DIR: Path = REPO_DIR / "models"
VOICE_DIRS: list[Path] = [
    REPO_DIR / "voice",
    REPO_DIR / "OpenAIglasses_for_Navigation-main" / "voice",
    REPO_DIR / "OpenAIglasses_for_Navigation-main" / "music",
]


def resolve_model(name_or_path: str | Path) -> Path:
    """Resolve a model file. Absolute paths are returned as-is; bare
    filenames are looked up under <repo>/models/.

    Returns the resolved Path even if the file does not exist yet — many
    loaders (ultralytics) auto-download missing weights.
    """
    p = Path(name_or_path)
    if p.is_absolute() or p.parent != Path("."):
        return p
    return MODELS_DIR / p.name
