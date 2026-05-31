"""Load leakage-aware PE-only training datasets and profile manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.feature_augmentation import add_derived_features, extend_feature_columns
from src.core.path_utils import resolve_stored_path
from src.core.extraction_support import METADATA_FIELDNAMES
from src.dataset.dataset_manager import FEATURE_BASE_FIELDS, load_dataset_frame


NON_INFORMATIVE_GROUP_VALUES = frozenset({
    "",
    "0",
    "unknown",
    "__unknown__",
    "__unknown_family__",
    "none",
    "null",
    "nan",
    "n/a",
    "na",
})


@dataclass(frozen=True)
class TrainingProfileResolution:
    profile_slug: str
    dataset_root: Path
    features_csv: Path
    features_parquet: Path
    metadata_csv: Path
    metadata_parquet: Path
    manifest_path: Path | None = None
    profile_definition: dict[str, Any] | None = None
    recommended_default_profile: str | None = None


@dataclass
class LoadedTrainingDataset:
    resolution: TrainingProfileResolution
    feature_df: pd.DataFrame
    feature_header: list[str]
    metadata_df: pd.DataFrame
    metadata_header: list[str]
    joined_df: pd.DataFrame
    feature_columns: list[str]
    target: pd.Series
    groups: pd.Series


def _normalize_text_series(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=str)
    return frame[field].astype(str).fillna("").str.strip()


def _is_informative_group_value(value: str) -> bool:
    return value.strip().casefold() not in NON_INFORMATIVE_GROUP_VALUES


def build_group_labels(frame: pd.DataFrame) -> pd.Series:
    source = _normalize_text_series(frame, "source")
    source_family = _normalize_text_series(frame, "source_family")
    sha256 = _normalize_text_series(frame, "sha256")

    labels: list[str] = []
    for index, (source_value, family_value, sha256_value) in enumerate(zip(source, source_family, sha256)):
        normalized_source = source_value if _is_informative_group_value(source_value) else ""
        normalized_family = family_value if _is_informative_group_value(family_value) else ""
        if normalized_family:
            labels.append(f"{normalized_source or '__unknown_source__'}::{normalized_family}")
        elif normalized_source:
            labels.append(f"{normalized_source}::sha256::{sha256_value or index}")
        else:
            labels.append(f"sha256::{sha256_value or index}")
    return pd.Series(labels, index=frame.index, dtype=str)


def _validate_loaded_tables(metadata_df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    if len(metadata_df) != len(feature_df):
        raise ValueError(
            "Training dataset is out of sync: metadata/features row counts do not match. "
            "Rebuild data/clear_training before training."
        )

    metadata_ids = set(_normalize_text_series(metadata_df, "sample_id"))
    feature_ids = set(_normalize_text_series(feature_df, "sample_id"))
    metadata_sha256 = set(_normalize_text_series(metadata_df, "sha256").str.casefold())
    feature_sha256 = set(_normalize_text_series(feature_df, "sha256").str.casefold())

    if metadata_ids != feature_ids:
        raise ValueError("Training dataset is out of sync: sample_id sets do not match.")
    if metadata_sha256 != feature_sha256:
        raise ValueError("Training dataset is out of sync: sha256 sets do not match.")


def resolve_training_profile(
    dataset_root: str | Path,
    profile: str | None = None,
) -> TrainingProfileResolution:
    root = Path(dataset_root).resolve()
    manifest_path = root / "model_profiles.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        profiles = manifest.get("profiles", {})
        selected_profile = profile or manifest.get("recommended_default_profile") or "pure_static"
        if selected_profile not in profiles:
            raise ValueError(
                f"Unknown training profile '{selected_profile}'. Available profiles: {', '.join(sorted(profiles))}"
            )
        profile_definition = profiles[selected_profile]
        profile_paths = profile_definition.get("paths", {})
        return TrainingProfileResolution(
            profile_slug=selected_profile,
            dataset_root=root,
            features_csv=resolve_stored_path(profile_paths["features_csv"], base_dir=manifest_path.parent),
            features_parquet=resolve_stored_path(profile_paths["features_parquet"], base_dir=manifest_path.parent),
            metadata_csv=resolve_stored_path(profile_paths["metadata_csv"], base_dir=manifest_path.parent),
            metadata_parquet=resolve_stored_path(profile_paths["metadata_parquet"], base_dir=manifest_path.parent),
            manifest_path=manifest_path,
            profile_definition=profile_definition,
            recommended_default_profile=manifest.get("recommended_default_profile"),
        )

    if profile is not None:
        raise ValueError(
            f"Dataset root {root} does not contain model_profiles.json; cannot resolve explicit profile '{profile}'."
        )

    return TrainingProfileResolution(
        profile_slug="canonical",
        dataset_root=root,
        features_csv=root / "features" / "clear_training_features.csv",
        features_parquet=root / "features" / "clear_training_features.parquet",
        metadata_csv=root / "metadata" / "clear_training_metadata.csv",
        metadata_parquet=root / "metadata" / "clear_training_metadata.parquet",
        manifest_path=None,
        profile_definition=None,
        recommended_default_profile=None,
    )


def load_training_dataset(
    dataset_root: str | Path,
    profile: str | None = None,
) -> LoadedTrainingDataset:
    resolution = resolve_training_profile(dataset_root, profile=profile)

    feature_df, feature_header = load_dataset_frame(
        resolution.features_csv,
        resolution.features_parquet,
        FEATURE_BASE_FIELDS,
        drop_deprecated_features=True,
    )
    feature_df = add_derived_features(feature_df)
    feature_header = extend_feature_columns(feature_header)
    metadata_df, metadata_header = load_dataset_frame(
        resolution.metadata_csv,
        resolution.metadata_parquet,
        METADATA_FIELDNAMES,
    )

    _validate_loaded_tables(metadata_df, feature_df)

    merge_keys = [field for field in FEATURE_BASE_FIELDS if field in metadata_df.columns and field in feature_df.columns]
    joined_df = metadata_df.merge(
        feature_df,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )
    if len(joined_df) != len(metadata_df):
        raise ValueError("Failed to join training metadata/features one-to-one.")

    feature_columns = [column for column in feature_header if column not in FEATURE_BASE_FIELDS]
    if not feature_columns:
        raise ValueError("Training dataset has no usable feature columns.")

    label_values = _normalize_text_series(joined_df, "label")
    invalid_labels = sorted({value for value in label_values.unique().tolist() if value not in {"0", "1"}})
    if invalid_labels:
        raise ValueError(f"Training dataset contains non-binary labels: {invalid_labels}")

    target = label_values.astype(int)
    groups = build_group_labels(joined_df)

    return LoadedTrainingDataset(
        resolution=resolution,
        feature_df=feature_df.reset_index(drop=True),
        feature_header=feature_header,
        metadata_df=metadata_df.reset_index(drop=True),
        metadata_header=metadata_header,
        joined_df=joined_df.reset_index(drop=True),
        feature_columns=feature_columns,
        target=target.reset_index(drop=True),
        groups=groups.reset_index(drop=True),
    )