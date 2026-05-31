"""Helpers for exporting a filtered training subset from aggregate dataset stores."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.core.path_utils import serialize_relative_paths
from src.dataset.dataset_manager import (
    FEATURE_BASE_FIELDS,
    DatasetPaths,
    dataset_paths,
    load_dataset_frame,
    write_dataset_files,
)
from src.core.extraction_support import METADATA_FIELDNAMES


@dataclass(frozen=True)
class TrainingSubsetPaths:
    output_root: Path
    features_csv: Path
    features_parquet: Path
    metadata_csv: Path
    metadata_parquet: Path
    summary_json: Path
    task_definition_json: Path
    profiles_root: Path
    model_profiles_json: Path


@dataclass(frozen=True)
class TrainingSubsetSummary:
    source_rows: int
    kept_rows: int
    excluded_rows: int
    excluded_low_confidence: int
    excluded_invalid_pe: int
    excluded_non_pe: int
    excluded_unlabeled: int
    label_distribution: dict[str, int]
    pe_kind_distribution: dict[str, int]


@dataclass(frozen=True)
class TrainingTaskDefinition:
    task_slug: str
    problem_type: str
    scope: str
    classifier_input_contract: str
    positive_class: str
    negative_class: str
    included_pe_kinds: list[str]
    excluded_inputs: list[str]
    filter_rules: dict[str, object]
    non_pe_policy: dict[str, str]
    recommended_training_profiles: list[str]


def training_subset_paths(output_root: str | Path) -> TrainingSubsetPaths:
    root = Path(output_root).resolve()
    return TrainingSubsetPaths(
        output_root=root,
        features_csv=root / "features" / "clear_training_features.csv",
        features_parquet=root / "features" / "clear_training_features.parquet",
        metadata_csv=root / "metadata" / "clear_training_metadata.csv",
        metadata_parquet=root / "metadata" / "clear_training_metadata.parquet",
        summary_json=root / "clear_training_summary.json",
        task_definition_json=root / "task_definition.json",
        profiles_root=root / "profiles",
        model_profiles_json=root / "model_profiles.json",
    )


def _profile_output_paths(profiles_root: Path, profile_slug: str) -> dict[str, Path]:
    profile_root = profiles_root / profile_slug
    return {
        "output_root": profile_root,
        "features_csv": profile_root / "features" / f"{profile_slug}_features.csv",
        "features_parquet": profile_root / "features" / f"{profile_slug}_features.parquet",
        "metadata_csv": profile_root / "metadata" / f"{profile_slug}_metadata.csv",
        "metadata_parquet": profile_root / "metadata" / f"{profile_slug}_metadata.parquet",
        "profile_definition_json": profile_root / "profile_definition.json",
    }


def _normalize_series(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=str)
    return frame[field].astype(str).fillna("").str.strip()


def _build_clean_training_mask(metadata_df: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    label_confidence = _normalize_series(metadata_df, "label_confidence").str.casefold()
    is_pe_valid = _normalize_series(metadata_df, "is_pe_valid")
    pe_kind = _normalize_series(metadata_df, "pe_kind").str.casefold()
    label = _normalize_series(metadata_df, "label")

    excluded_low_confidence = label_confidence.eq("low")
    excluded_invalid_pe = is_pe_valid.eq("0")
    excluded_non_pe = pe_kind.eq("non_pe")
    excluded_unlabeled = ~label.isin({"0", "1"})

    keep_mask = ~(excluded_low_confidence | excluded_invalid_pe | excluded_non_pe | excluded_unlabeled)
    counts = {
        "excluded_low_confidence": int(excluded_low_confidence.sum()),
        "excluded_invalid_pe": int(excluded_invalid_pe.sum()),
        "excluded_non_pe": int(excluded_non_pe.sum()),
        "excluded_unlabeled": int(excluded_unlabeled.sum()),
    }
    return keep_mask, counts


def _validate_source_tables(metadata_df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    if len(metadata_df) != len(feature_df):
        raise ValueError(
            "Aggregate dataset is out of sync: metadata/features row counts do not match. "
            "Run scripts/check_dataset.py before exporting a training subset."
        )

    metadata_ids = _normalize_series(metadata_df, "sample_id")
    feature_ids = _normalize_series(feature_df, "sample_id")
    metadata_sha256 = _normalize_series(metadata_df, "sha256").str.casefold()
    feature_sha256 = _normalize_series(feature_df, "sha256").str.casefold()

    if set(metadata_ids) != set(feature_ids):
        raise ValueError("Aggregate dataset is out of sync: sample_id sets do not match.")
    if set(metadata_sha256) != set(feature_sha256):
        raise ValueError("Aggregate dataset is out of sync: sha256 sets do not match.")


def _ordered_feature_subset(feature_df: pd.DataFrame, metadata_subset: pd.DataFrame) -> pd.DataFrame:
    selected_sample_ids = _normalize_series(metadata_subset, "sample_id").tolist()
    selected_sha256 = _normalize_series(metadata_subset, "sha256").str.casefold().tolist()

    feature_subset = feature_df.copy()
    feature_subset = feature_subset.assign(
        _sample_id=_normalize_series(feature_subset, "sample_id"),
        _sha256=_normalize_series(feature_subset, "sha256").str.casefold(),
    )

    sample_order = {value: index for index, value in enumerate(selected_sample_ids) if value}
    sha256_order = {value: index for index, value in enumerate(selected_sha256) if value}

    mask = feature_subset["_sample_id"].isin(sample_order) | feature_subset["_sha256"].isin(sha256_order)
    feature_subset = feature_subset.loc[mask].copy()
    if feature_subset.empty and not metadata_subset.empty:
        raise ValueError("Failed to locate matching feature rows for the filtered metadata subset.")

    feature_subset["_order"] = feature_subset["_sample_id"].map(sample_order)
    missing_order = feature_subset["_order"].isna()
    if missing_order.any():
        feature_subset.loc[missing_order, "_order"] = feature_subset.loc[missing_order, "_sha256"].map(sha256_order)

    if feature_subset["_order"].isna().any():
        raise ValueError("Could not order filtered feature rows against metadata subset.")

    feature_subset = feature_subset.sort_values("_order", kind="stable")
    feature_subset = feature_subset.drop(columns=["_sample_id", "_sha256", "_order"])
    feature_subset = feature_subset.reset_index(drop=True)
    return feature_subset


def _vt_feature_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column.startswith("vt_")]


def _serialize_profile_paths(paths: dict[str, Path], *, base_dir: Path) -> dict[str, str]:
    return serialize_relative_paths(
        {key: value for key, value in paths.items() if key != "output_root"},
        base_dir=base_dir,
    )


def _build_training_profiles(
    *,
    output_paths: TrainingSubsetPaths,
    metadata_subset: pd.DataFrame,
    metadata_header: list[str],
    feature_subset: pd.DataFrame,
    feature_header: list[str],
) -> dict[str, dict[str, object]]:
    vt_columns = _vt_feature_columns(list(feature_subset.columns))
    profiles: dict[str, dict[str, object]] = {}

    pure_static_paths = _profile_output_paths(output_paths.profiles_root, "pure_static")
    pure_static_feature_df = feature_subset.drop(columns=vt_columns, errors="ignore")
    pure_static_feature_header = [column for column in feature_header if column not in set(vt_columns)]
    pure_static_metadata_df = metadata_subset.copy()
    if "av_detection_count" in pure_static_metadata_df.columns:
        pure_static_metadata_df["av_detection_count"] = ""

    write_dataset_files(
        pure_static_paths["metadata_csv"],
        pure_static_paths["metadata_parquet"],
        pure_static_metadata_df,
        metadata_header,
    )
    write_dataset_files(
        pure_static_paths["features_csv"],
        pure_static_paths["features_parquet"],
        pure_static_feature_df,
        pure_static_feature_header,
        drop_deprecated_features=True,
    )
    pure_static_definition = {
        "profile_slug": "pure_static",
        "intended_use": "first_baseline",
        "description": "Pure static PE-only training profile with VT-derived signals removed.",
        "vt_features_included": False,
        "dropped_feature_columns": vt_columns,
        "scrubbed_metadata_fields": ["av_detection_count"],
        "feature_count": len(pure_static_feature_df.columns),
        "row_count": len(pure_static_feature_df),
        "paths": _serialize_profile_paths(pure_static_paths, base_dir=output_paths.output_root),
    }
    pure_static_paths["profile_definition_json"].parent.mkdir(parents=True, exist_ok=True)
    pure_static_paths["profile_definition_json"].write_text(
        json.dumps(
            {
                **pure_static_definition,
                "paths": _serialize_profile_paths(
                    pure_static_paths,
                    base_dir=pure_static_paths["profile_definition_json"].parent,
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    profiles["pure_static"] = pure_static_definition

    augmented_paths = _profile_output_paths(output_paths.profiles_root, "augmented_triage")
    write_dataset_files(
        augmented_paths["metadata_csv"],
        augmented_paths["metadata_parquet"],
        metadata_subset,
        metadata_header,
    )
    write_dataset_files(
        augmented_paths["features_csv"],
        augmented_paths["features_parquet"],
        feature_subset,
        feature_header,
        drop_deprecated_features=True,
    )
    augmented_definition = {
        "profile_slug": "augmented_triage",
        "intended_use": "operational_triage",
        "description": "PE-only training profile that keeps VT-derived features for operational use.",
        "vt_features_included": True,
        "dropped_feature_columns": [],
        "scrubbed_metadata_fields": [],
        "feature_count": len(feature_subset.columns),
        "row_count": len(feature_subset),
        "paths": _serialize_profile_paths(augmented_paths, base_dir=output_paths.output_root),
    }
    augmented_paths["profile_definition_json"].parent.mkdir(parents=True, exist_ok=True)
    augmented_paths["profile_definition_json"].write_text(
        json.dumps(
            {
                **augmented_definition,
                "paths": _serialize_profile_paths(
                    augmented_paths,
                    base_dir=augmented_paths["profile_definition_json"].parent,
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    profiles["augmented_triage"] = augmented_definition

    output_paths.model_profiles_json.parent.mkdir(parents=True, exist_ok=True)
    output_paths.model_profiles_json.write_text(
        json.dumps(
            {
                "recommended_default_profile": "pure_static",
                "profiles": profiles,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return profiles


def export_clean_training_subset(
    project_root: str | Path,
    output_root: str | Path,
) -> tuple[TrainingSubsetPaths, TrainingSubsetSummary, TrainingTaskDefinition, dict[str, dict[str, object]]]:
    source_paths: DatasetPaths = dataset_paths(project_root)
    output_paths = training_subset_paths(output_root)

    metadata_df, metadata_header = load_dataset_frame(
        source_paths.metadata_csv,
        source_paths.metadata_parquet,
        METADATA_FIELDNAMES,
    )
    feature_df, feature_header = load_dataset_frame(
        source_paths.features_csv,
        source_paths.features_parquet,
        FEATURE_BASE_FIELDS,
        drop_deprecated_features=True,
    )

    _validate_source_tables(metadata_df, feature_df)

    keep_mask, excluded_counts = _build_clean_training_mask(metadata_df)
    metadata_subset = metadata_df.loc[keep_mask].copy().reset_index(drop=True)
    feature_subset = _ordered_feature_subset(feature_df, metadata_subset)

    if len(metadata_subset) != len(feature_subset):
        raise ValueError("Filtered subset is out of sync: metadata/features row counts do not match.")

    metadata_subset_ids = _normalize_series(metadata_subset, "sample_id").tolist()
    feature_subset_ids = _normalize_series(feature_subset, "sample_id").tolist()
    metadata_subset_sha256 = _normalize_series(metadata_subset, "sha256").str.casefold().tolist()
    feature_subset_sha256 = _normalize_series(feature_subset, "sha256").str.casefold().tolist()
    if metadata_subset_ids != feature_subset_ids:
        raise ValueError("Filtered subset is out of sync: sample_id order does not match.")
    if metadata_subset_sha256 != feature_subset_sha256:
        raise ValueError("Filtered subset is out of sync: sha256 order does not match.")

    write_dataset_files(output_paths.metadata_csv, output_paths.metadata_parquet, metadata_subset, metadata_header)
    write_dataset_files(
        output_paths.features_csv,
        output_paths.features_parquet,
        feature_subset,
        feature_header,
        drop_deprecated_features=True,
    )

    summary = TrainingSubsetSummary(
        source_rows=len(metadata_df),
        kept_rows=len(metadata_subset),
        excluded_rows=len(metadata_df) - len(metadata_subset),
        excluded_low_confidence=excluded_counts["excluded_low_confidence"],
        excluded_invalid_pe=excluded_counts["excluded_invalid_pe"],
        excluded_non_pe=excluded_counts["excluded_non_pe"],
        excluded_unlabeled=excluded_counts["excluded_unlabeled"],
        label_distribution={
            key: int(value)
            for key, value in _normalize_series(metadata_subset, "label").value_counts().sort_index().items()
        },
        pe_kind_distribution={
            key: int(value)
            for key, value in _normalize_series(metadata_subset, "pe_kind").value_counts().sort_index().items()
        },
    )

    task_definition = TrainingTaskDefinition(
        task_slug="pe_only_malicious_classifier",
        problem_type="binary_classification",
        scope="valid_pe_only",
        classifier_input_contract=(
            "Score only files that pass PE validation. Non-PE inputs are out of scope for this model."
        ),
        positive_class="malicious_pe",
        negative_class="benign_pe",
        included_pe_kinds=["exe", "dll", "driver", "other_pe"],
        excluded_inputs=["non_pe", "label_confidence=low", "unlabeled_rows"],
        filter_rules={
            "label_in": ["0", "1"],
            "label_confidence_not_in": ["low"],
            "is_pe_valid_equals": "1",
            "pe_kind_not_in": ["non_pe"],
        },
        non_pe_policy={
            "decision": "out_of_scope",
            "future_extension": "dedicated_pre_classifier",
        },
        recommended_training_profiles=["pure_static", "augmented_triage"],
    )

    profiles = _build_training_profiles(
        output_paths=output_paths,
        metadata_subset=metadata_subset,
        metadata_header=metadata_header,
        feature_subset=feature_subset,
        feature_header=feature_header,
    )

    output_paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_paths.summary_json.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_paths.task_definition_json.write_text(
        json.dumps(asdict(task_definition), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_paths, summary, task_definition, profiles