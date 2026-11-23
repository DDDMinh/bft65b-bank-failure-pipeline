"""Project path helpers.

All relative config paths are resolved against the repository root, not the
current working directory. This keeps CLI commands reproducible from any shell.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def resolve_path(path_like: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else root / path


def ensure_directory(path_like: str | Path) -> Path:
    """Create and return a directory path."""
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path
