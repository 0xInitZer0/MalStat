"""Shared management helpers for aggregate dataset CSV/Parquet stores."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.core.extraction_support import (
    DEPRECATED_FEATURE_ALIASES,
    METADATA_FIELDNAMES,
    filter_deprecated_feature_aliases,
    infer_label_from_source,
    normalize_row_value,
    unique_preserve_order,
)


COMMON_DATASET_FIELDS = frozenset({"sample_id", "sha256", "label"})
FEATURE_BASE_FIELDS = ["sample_id", "sha256", "label"]
DEFAULT_LIST_COLUMNS = [
    "sample_id",
    "sha256",
    "original_name",
    "file_type",
    "pe_kind",
    "label",
    "label_confidence",
    "source",
    "source_family",
    "av_detection_count",
]


@dataclass(frozen=True)
class DatasetPaths:
    features_csv: Path
    features_parquet: Path
    metadata_csv: Path
    metadata_parquet: Path


@dataclass(frozen=True)
class DatasetSelection:
    select_all: bool = False
    sample_ids: tuple[str, ...] = ()
    sha256s: tuple[str, ...] = ()
    row_numbers: tuple[int, ...] = ()
    match_filters: tuple[tuple[str, str], ...] = ()

    def is_empty(self) -> bool:
        return not any([self.select_all, self.sample_ids, self.sha256s, self.row_numbers, self.match_filters])


@dataclass
class DatasetTables:
    metadata_df: pd.DataFrame
    metadata_header: list[str]
    feature_df: pd.DataFrame
    feature_header: list[str]


def dataset_paths(project_root: str | Path) -> DatasetPaths:
    root = Path(project_root).resolve()
    return DatasetPaths(
        features_csv=root / "data" / "features" / "features_example.csv",
        features_parquet=root / "data" / "features" / "features_example.parquet",
        metadata_csv=root / "data" / "metadata" / "samples_metadata.csv",
        metadata_parquet=root / "data" / "metadata" / "samples_metadata.parquet",
    )


def read_header(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), [])


def parse_key_value_pairs(raw_pairs: Iterable[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for raw_pair in raw_pairs:
        if "=" not in raw_pair:
            raise ValueError(f"Invalid field=value pair: {raw_pair}")
        field, value = raw_pair.split("=", 1)
        field = field.strip()
        if not field:
            raise ValueError(f"Invalid field=value pair: {raw_pair}")
        pairs[field] = value.strip()
    return pairs


def load_dataset_frame(
    csv_path: Path,
    parquet_path: Path,
    default_columns: list[str] | None = None,
    *,
    drop_deprecated_features: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    columns = read_header(csv_path)
    if drop_deprecated_features:
        columns = filter_deprecated_feature_aliases(columns)

    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[]).fillna("")
        if not columns:
            columns = list(df.columns)
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path).fillna("")
        df = df.astype(str)
        columns = list(df.columns)
    else:
        columns = list(default_columns or [])
        df = pd.DataFrame(columns=columns)

    if drop_deprecated_features:
        columns = filter_deprecated_feature_aliases(columns)
        df = df.reindex(columns=columns, fill_value="")

    if default_columns:
        defaults = list(default_columns)
        if drop_deprecated_features:
            defaults = filter_deprecated_feature_aliases(defaults)
        columns = unique_preserve_order(list(columns) + defaults)
    if columns:
        df = df.reindex(columns=columns, fill_value="")
    return df.fillna("").astype(str), columns


def write_dataset_files(
    csv_path: Path,
    parquet_path: Path,
    df: pd.DataFrame,
    header: list[str],
    *,
    drop_deprecated_features: bool = False,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_header = list(header)
    if drop_deprecated_features:
        ordered_header = filter_deprecated_feature_aliases(ordered_header)
    if not ordered_header:
        ordered_header = list(df.columns)
    if drop_deprecated_features:
        ordered_header = filter_deprecated_feature_aliases(ordered_header)

    out_df = df.reindex(columns=ordered_header, fill_value="").fillna("").astype(str)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_header)
        writer.writeheader()
        for record in out_df.to_dict(orient="records"):
            writer.writerow({field: record.get(field, "") for field in ordered_header})

    out_df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")


def read_single_row_csv(path: str | Path) -> tuple[list[str], dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
        if row is None:
            raise ValueError(f"CSV file has no data rows: {csv_path}")
        return list(reader.fieldnames or []), row


def apply_append_overrides(
    metadata_row: dict[str, Any],
    feature_row: dict[str, Any],
    *,
    label: str | None = None,
    label_confidence: str | None = None,
    download_date: str | None = None,
    source: str | None = None,
    source_family: str | None = None,
    av_detection_count: str | int | None = None,
    allow_missing_label: bool = False,
    shared_updates: dict[str, str] | None = None,
    metadata_updates: dict[str, str] | None = None,
    feature_updates: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_out = dict(metadata_row)
    feature_out = dict(feature_row)

    for field, value in (shared_updates or {}).items():
        if field not in COMMON_DATASET_FIELDS:
            raise ValueError(f"Field '{field}' is not shared across metadata/features. Use --set-meta or --set-feature.")
        metadata_out[field] = value
        feature_out[field] = value

    for field, value in (metadata_updates or {}).items():
        metadata_out[field] = value
    for field, value in (feature_updates or {}).items():
        if field in DEPRECATED_FEATURE_ALIASES:
            raise ValueError(f"Field '{field}' is a deprecated feature alias.")
        feature_out[field] = value

    if source is not None:
        metadata_out["source"] = source
    if source_family is not None:
        metadata_out["source_family"] = source_family
    if av_detection_count is not None:
        metadata_out["av_detection_count"] = av_detection_count

    resolved_label = label if label is not None else str(metadata_out.get("label", ""))
    if resolved_label == "":
        inferred_label = infer_label_from_source(
            metadata_out.get("source"),
            metadata_out.get("source_family"),
        )
        if inferred_label is not None:
            resolved_label = str(inferred_label)

    if resolved_label == "" and not allow_missing_label:
        raise ValueError(
            "Label is required for dataset append. Pass --label or use a source/source_family that implies benign/malware."
        )

    metadata_out["label"] = resolved_label
    feature_out["label"] = resolved_label
    metadata_out["label_confidence"] = (
        label_confidence
        or str(metadata_out.get("label_confidence", ""))
        or ("manual" if resolved_label != "" else "")
    )
    metadata_out["download_date"] = (
        download_date
        or str(metadata_out.get("download_date", ""))
        or date.today().isoformat()
    )
    return metadata_out, feature_out


class DatasetManager:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.paths = dataset_paths(self.project_root)

    def load_tables(self) -> DatasetTables:
        metadata_df, metadata_header = load_dataset_frame(
            self.paths.metadata_csv,
            self.paths.metadata_parquet,
            METADATA_FIELDNAMES,
        )
        feature_df, feature_header = load_dataset_frame(
            self.paths.features_csv,
            self.paths.features_parquet,
            FEATURE_BASE_FIELDS,
            drop_deprecated_features=True,
        )
        return DatasetTables(
            metadata_df=metadata_df,
            metadata_header=unique_preserve_order(list(metadata_header) + METADATA_FIELDNAMES),
            feature_df=feature_df,
            feature_header=unique_preserve_order(filter_deprecated_feature_aliases(list(feature_header) + FEATURE_BASE_FIELDS)),
        )

    def list_records(
        self,
        selection: DatasetSelection | None = None,
        *,
        columns: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        tables = self.load_tables()
        view = self._build_record_view(tables)
        effective_selection = selection or DatasetSelection(select_all=True)
        if effective_selection.is_empty():
            effective_selection = DatasetSelection(select_all=True)
        mask = self._build_record_view_mask(view, effective_selection)
        selected = view.loc[mask].copy() if not mask.empty else view.iloc[0:0].copy()

        requested_columns = columns or [field for field in DEFAULT_LIST_COLUMNS if field in selected.columns]
        if requested_columns:
            missing = [field for field in requested_columns if field not in selected.columns]
            if missing:
                raise ValueError(f"Unknown dataset columns: {', '.join(missing)}")
            selected = selected.reindex(columns=requested_columns)
        if limit is not None and limit >= 0:
            selected = selected.head(limit)
        return selected.fillna("").astype(str).to_dict(orient="records")

    def add_rows(
        self,
        *,
        feature_row: dict[str, Any],
        metadata_row: dict[str, Any],
        feature_fieldnames: list[str],
        allow_duplicates: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        tables = self.load_tables()
        metadata_out = self._normalize_row(metadata_row)
        feature_out = self._normalize_row(feature_row)

        sha256 = str(metadata_out.get("sha256") or feature_out.get("sha256") or "").strip()
        if not sha256:
            raise ValueError("Both dataset rows require a sha256 value.")
        metadata_out["sha256"] = sha256
        feature_out["sha256"] = sha256

        if not allow_duplicates:
            existing_sha256s = {
                value.strip().casefold()
                for value in tables.metadata_df.get("sha256", pd.Series(dtype=str)).astype(str)
                if value.strip()
            }
            if sha256.casefold() in existing_sha256s:
                raise ValueError(f"Duplicate sha256 already exists in dataset: {sha256}")

        sample_id = str(metadata_out.get("sample_id") or feature_out.get("sample_id") or "").strip()
        if not sample_id:
            sample_id = str(self._next_sample_id(tables.metadata_df))
        metadata_out["sample_id"] = sample_id
        feature_out["sample_id"] = sample_id

        metadata_header = unique_preserve_order(list(tables.metadata_header) + METADATA_FIELDNAMES)
        feature_header = unique_preserve_order(
            filter_deprecated_feature_aliases(list(tables.feature_header) + list(feature_fieldnames) + list(feature_out.keys()))
        )

        metadata_appended = pd.concat(
            [tables.metadata_df, pd.DataFrame([{field: str(metadata_out.get(field, "")) for field in metadata_header}])],
            ignore_index=True,
            sort=False,
        )
        feature_appended = pd.concat(
            [tables.feature_df, pd.DataFrame([{field: str(feature_out.get(field, "")) for field in feature_header}])],
            ignore_index=True,
            sort=False,
        )

        if not dry_run:
            write_dataset_files(
                self.paths.metadata_csv,
                self.paths.metadata_parquet,
                metadata_appended,
                metadata_header,
            )
            write_dataset_files(
                self.paths.features_csv,
                self.paths.features_parquet,
                feature_appended,
                feature_header,
                drop_deprecated_features=True,
            )

        return {
            "sample_id": sample_id,
            "sha256": sha256,
            "metadata_added": 1,
            "features_added": 1,
            "features_csv": self.paths.features_csv,
            "features_parquet": self.paths.features_parquet,
            "metadata_csv": self.paths.metadata_csv,
            "metadata_parquet": self.paths.metadata_parquet,
        }

    def backfill_feature_rows(
        self,
        feature_rows: Iterable[dict[str, Any]],
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        tables = self.load_tables()
        feature_out = tables.feature_df.copy()
        feature_header = list(tables.feature_header)

        if "sha256" not in feature_out.columns:
            feature_out["sha256"] = ""
            feature_header.append("sha256")

        normalized_sha256 = feature_out["sha256"].astype(str).str.strip().str.casefold()
        processed_files = 0
        matched_files = 0
        unmatched_files = 0
        updated_rows = 0
        added_feature_columns: list[str] = []

        for raw_feature_row in feature_rows:
            processed_files += 1
            normalized_row = self._normalize_row(raw_feature_row)
            sha256 = str(normalized_row.get("sha256", "")).strip().casefold()
            if not sha256:
                raise ValueError("Backfill feature rows require sha256 values.")

            feature_updates = {
                field: value
                for field, value in normalized_row.items()
                if field not in COMMON_DATASET_FIELDS
            }
            mask = normalized_sha256.eq(sha256)
            matched_count = int(mask.sum()) if not mask.empty else 0
            if matched_count == 0:
                unmatched_files += 1
                continue

            matched_files += 1
            for field in feature_updates:
                if field not in feature_out.columns:
                    feature_out[field] = ""
                    feature_header.append(field)
                    added_feature_columns.append(field)

            for field, value in feature_updates.items():
                feature_out.loc[mask, field] = value
            updated_rows += matched_count

        feature_header = unique_preserve_order(filter_deprecated_feature_aliases(feature_header))
        if not dry_run:
            write_dataset_files(
                self.paths.features_csv,
                self.paths.features_parquet,
                feature_out,
                feature_header,
                drop_deprecated_features=True,
            )

        return {
            "processed_files": processed_files,
            "matched_files": matched_files,
            "unmatched_files": unmatched_files,
            "updated_rows": updated_rows,
            "added_feature_columns": unique_preserve_order(added_feature_columns),
        }

    def edit_records(
        self,
        selection: DatasetSelection,
        *,
        shared_updates: dict[str, str] | None = None,
        metadata_updates: dict[str, str] | None = None,
        feature_updates: dict[str, str] | None = None,
        metadata_only: bool = False,
        features_only: bool = False,
        dry_run: bool = False,
    ) -> dict[str, int]:
        if metadata_only and features_only:
            raise ValueError("Use either metadata_only or features_only, not both.")

        shared_updates = dict(shared_updates or {})
        metadata_updates = dict(metadata_updates or {})
        feature_updates = dict(feature_updates or {})
        if not any([shared_updates, metadata_updates, feature_updates]):
            raise ValueError("Provide at least one update field.")

        self._validate_update_fields(shared_updates, metadata_updates, feature_updates)
        tables = self.load_tables()
        metadata_mask, feature_mask = self._resolve_row_masks(tables, selection)
        if metadata_only:
            feature_mask = pd.Series([False] * len(tables.feature_df), index=tables.feature_df.index)
        if features_only:
            metadata_mask = pd.Series([False] * len(tables.metadata_df), index=tables.metadata_df.index)

        metadata_matched = int(metadata_mask.sum()) if not metadata_mask.empty else 0
        features_matched = int(feature_mask.sum()) if not feature_mask.empty else 0
        if dry_run:
            return {
                "metadata_matched": metadata_matched,
                "features_matched": features_matched,
            }

        metadata_out = tables.metadata_df.copy()
        feature_out = tables.feature_df.copy()

        metadata_header = list(tables.metadata_header)
        for field in unique_preserve_order(list(shared_updates) + list(metadata_updates)):
            if field not in metadata_out.columns:
                metadata_out[field] = ""
            if field not in metadata_header:
                metadata_header.append(field)

        feature_header = list(tables.feature_header)
        for field in unique_preserve_order(list(shared_updates) + list(feature_updates)):
            if field in DEPRECATED_FEATURE_ALIASES:
                continue
            if field not in feature_out.columns:
                feature_out[field] = ""
            if field not in feature_header:
                feature_header.append(field)

        if metadata_matched:
            for field, value in {**shared_updates, **metadata_updates}.items():
                metadata_out.loc[metadata_mask, field] = value
        if features_matched:
            for field, value in {**shared_updates, **feature_updates}.items():
                feature_out.loc[feature_mask, field] = value

        if not features_only:
            write_dataset_files(
                self.paths.metadata_csv,
                self.paths.metadata_parquet,
                metadata_out,
                metadata_header,
            )
        if not metadata_only:
            write_dataset_files(
                self.paths.features_csv,
                self.paths.features_parquet,
                feature_out,
                feature_header,
                drop_deprecated_features=True,
            )

        return {
            "metadata_matched": metadata_matched,
            "features_matched": features_matched,
        }

    def delete_records(
        self,
        selection: DatasetSelection,
        *,
        metadata_only: bool = False,
        features_only: bool = False,
        dry_run: bool = False,
    ) -> dict[str, int]:
        if metadata_only and features_only:
            raise ValueError("Use either metadata_only or features_only, not both.")

        tables = self.load_tables()
        metadata_mask, feature_mask = self._resolve_row_masks(tables, selection)
        if metadata_only:
            feature_mask = pd.Series([False] * len(tables.feature_df), index=tables.feature_df.index)
        if features_only:
            metadata_mask = pd.Series([False] * len(tables.metadata_df), index=tables.metadata_df.index)

        metadata_removed = int(metadata_mask.sum()) if not metadata_mask.empty else 0
        features_removed = int(feature_mask.sum()) if not feature_mask.empty else 0
        if dry_run:
            return {
                "metadata_removed": metadata_removed,
                "features_removed": features_removed,
                "metadata_remaining": len(tables.metadata_df) - metadata_removed,
                "features_remaining": len(tables.feature_df) - features_removed,
            }

        if not features_only:
            metadata_out = tables.metadata_df.loc[~metadata_mask].copy() if not metadata_mask.empty else tables.metadata_df.copy()
            write_dataset_files(
                self.paths.metadata_csv,
                self.paths.metadata_parquet,
                metadata_out,
                tables.metadata_header,
            )
        if not metadata_only:
            feature_out = tables.feature_df.loc[~feature_mask].copy() if not feature_mask.empty else tables.feature_df.copy()
            write_dataset_files(
                self.paths.features_csv,
                self.paths.features_parquet,
                feature_out,
                tables.feature_header,
                drop_deprecated_features=True,
            )

        return {
            "metadata_removed": metadata_removed,
            "features_removed": features_removed,
            "metadata_remaining": len(tables.metadata_df) - metadata_removed,
            "features_remaining": len(tables.feature_df) - features_removed,
        }

    def _build_record_view(self, tables: DatasetTables) -> pd.DataFrame:
        merged = tables.metadata_df.merge(
            tables.feature_df,
            on=["sample_id", "sha256"],
            how="outer",
            suffixes=("_metadata", "_feature"),
            sort=False,
        ).fillna("")

        view = pd.DataFrame(index=merged.index)
        view["row_number"] = [index + 1 for index in range(len(merged))]
        view["sample_id"] = merged.get("sample_id", pd.Series([""] * len(merged), index=merged.index)).astype(str)
        view["sha256"] = merged.get("sha256", pd.Series([""] * len(merged), index=merged.index)).astype(str)
        view["label"] = self._coalesce_series(
            merged.get("label_metadata"),
            merged.get("label_feature"),
            merged.index,
        )

        for field in tables.metadata_header:
            if field in {"sample_id", "sha256", "label"}:
                continue
            if field in merged.columns:
                view[field] = merged[field].astype(str)

        for field in tables.feature_header:
            if field in {"sample_id", "sha256", "label"}:
                continue
            if field in merged.columns and field not in view.columns:
                view[field] = merged[field].astype(str)

        return view.fillna("").astype(str)

    def _build_record_view_mask(self, view: pd.DataFrame, selection: DatasetSelection) -> pd.Series:
        if view.empty:
            return pd.Series(dtype=bool)
        if selection.select_all:
            return pd.Series([True] * len(view), index=view.index)

        masks: list[pd.Series] = []
        if selection.sample_ids:
            masks.append(view["sample_id"].astype(str).isin([str(value) for value in selection.sample_ids]))
        if selection.sha256s:
            normalized = {value.strip().casefold() for value in selection.sha256s}
            masks.append(view["sha256"].astype(str).str.strip().str.casefold().isin(normalized))
        if selection.row_numbers:
            valid_numbers = {value for value in selection.row_numbers if int(value) >= 1}
            if not valid_numbers:
                raise ValueError("row_numbers must be 1-based positive integers.")
            masks.append(view["row_number"].astype(int).isin(valid_numbers))
        if selection.match_filters:
            match_mask = pd.Series([True] * len(view), index=view.index)
            for field, expected in selection.match_filters:
                if field not in view.columns:
                    raise ValueError(f"Dataset does not contain field '{field}'.")
                match_mask &= view[field].astype(str) == expected
            masks.append(match_mask)

        if not masks:
            raise ValueError("Provide at least one selector or set select_all=True.")

        combined = pd.Series([False] * len(view), index=view.index)
        for mask in masks:
            combined |= mask
        return combined

    def _resolve_row_masks(self, tables: DatasetTables, selection: DatasetSelection) -> tuple[pd.Series, pd.Series]:
        view = self._build_record_view(tables)
        mask = self._build_record_view_mask(view, selection)

        metadata_mask = pd.Series([False] * len(tables.metadata_df), index=tables.metadata_df.index)
        feature_mask = pd.Series([False] * len(tables.feature_df), index=tables.feature_df.index)
        if view.empty or mask.empty:
            return metadata_mask, feature_mask

        if selection.select_all:
            return (
                pd.Series([True] * len(tables.metadata_df), index=tables.metadata_df.index),
                pd.Series([True] * len(tables.feature_df), index=tables.feature_df.index),
            )

        selected_sample_ids = {
            value
            for value in view.loc[mask, "sample_id"].astype(str)
            if value != ""
        }
        selected_sha256s = {
            value.strip().casefold()
            for value in view.loc[mask, "sha256"].astype(str)
            if value.strip()
        }

        if selected_sample_ids and "sample_id" in tables.metadata_df.columns:
            metadata_mask |= tables.metadata_df["sample_id"].astype(str).isin(selected_sample_ids)
        if selected_sha256s and "sha256" in tables.metadata_df.columns:
            metadata_mask |= tables.metadata_df["sha256"].astype(str).str.strip().str.casefold().isin(selected_sha256s)
        if selected_sample_ids and "sample_id" in tables.feature_df.columns:
            feature_mask |= tables.feature_df["sample_id"].astype(str).isin(selected_sample_ids)
        if selected_sha256s and "sha256" in tables.feature_df.columns:
            feature_mask |= tables.feature_df["sha256"].astype(str).str.strip().str.casefold().isin(selected_sha256s)

        return metadata_mask, feature_mask

    def _next_sample_id(self, metadata_df: pd.DataFrame) -> int:
        if metadata_df.empty or "sample_id" not in metadata_df.columns:
            return 1
        max_sample_id = 0
        for raw_value in metadata_df["sample_id"].astype(str):
            value = raw_value.strip()
            if not value:
                continue
            try:
                max_sample_id = max(max_sample_id, int(value))
            except ValueError:
                continue
        return max_sample_id + 1

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in row.items():
            if key in DEPRECATED_FEATURE_ALIASES:
                continue
            normalized[key] = "" if value is None else str(normalize_row_value(value))
        return normalized

    def _validate_update_fields(
        self,
        shared_updates: dict[str, str],
        metadata_updates: dict[str, str],
        feature_updates: dict[str, str],
    ) -> None:
        invalid_shared = [field for field in shared_updates if field not in COMMON_DATASET_FIELDS]
        if invalid_shared:
            raise ValueError(
                "Shared updates support only sample_id, sha256, and label. Use metadata/feature-specific updates for other fields."
            )

        allowed_metadata = set(METADATA_FIELDNAMES)
        invalid_metadata = [field for field in metadata_updates if field not in allowed_metadata]
        if invalid_metadata:
            raise ValueError(f"Unknown metadata field(s): {', '.join(invalid_metadata)}")

        invalid_feature = [field for field in feature_updates if field in DEPRECATED_FEATURE_ALIASES]
        if invalid_feature:
            raise ValueError(f"Deprecated feature alias field(s): {', '.join(invalid_feature)}")

    def _coalesce_series(
        self,
        preferred: pd.Series | None,
        fallback: pd.Series | None,
        index: pd.Index,
    ) -> pd.Series:
        if preferred is None:
            preferred_series = pd.Series([""] * len(index), index=index)
        else:
            preferred_series = preferred.fillna("").astype(str)
        if fallback is None:
            fallback_series = pd.Series([""] * len(index), index=index)
        else:
            fallback_series = fallback.fillna("").astype(str)
        return preferred_series.where(preferred_series != "", fallback_series)