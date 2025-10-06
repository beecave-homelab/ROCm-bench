"""Utility helper functions."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)
