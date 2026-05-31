"""Helpers for storing project-local paths without leaking absolute locations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def resolve_stored_path(path_value: str | Path, *, base_dir: str | Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(base_dir).resolve() / candidate).resolve()


def serialize_relative_path(path_value: str | Path, *, base_dir: str | Path) -> str:
    resolved_path = Path(path_value).resolve()
    resolved_base_dir = Path(base_dir).resolve()
    try:
        relative_path = Path(os.path.relpath(resolved_path, resolved_base_dir))
    except ValueError:
        return resolved_path.as_posix()
    return relative_path.as_posix()


def serialize_relative_paths(
    path_values: Mapping[str, str | Path],
    *,
    base_dir: str | Path,
) -> dict[str, str]:
    return {
        key: serialize_relative_path(value, base_dir=base_dir)
        for key, value in path_values.items()
    }