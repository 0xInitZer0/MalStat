"""Append extracted rows into the project CSV/Parquet datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.dataset.dataset_manager import DatasetManager


def append_dataset_rows(
    feature_row: dict[str, Any],
    metadata_row: dict[str, Any],
    feature_fieldnames: list[str],
    project_root: str | Path,
) -> tuple[str, Path, Path, Path, Path]:
    manager = DatasetManager(project_root)
    result = manager.add_rows(
        feature_row=feature_row,
        metadata_row=metadata_row,
        feature_fieldnames=feature_fieldnames,
    )
    return (
        result["sample_id"],
        result["features_csv"],
        result["features_parquet"],
        result["metadata_csv"],
        result["metadata_parquet"],
    )