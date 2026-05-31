"""Clear aggregate dataset files fully or selectively."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.extraction_support import METADATA_FIELDNAMES, unique_preserve_order


@dataclass(frozen=True)
class DatasetPaths:
    features_csv: Path
    features_parquet: Path
    metadata_csv: Path
    metadata_parquet: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clear aggregate dataset files completely or selectively by sample_id, sha256, or metadata row number."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root path.")
    parser.add_argument("--all", action="store_true", help="Remove all rows from the selected dataset files but keep headers/schema.")
    parser.add_argument("--sample-id", nargs="+", help="One or more sample_id values to remove.")
    parser.add_argument("--sha256", nargs="+", help="One or more SHA-256 values to remove.")
    parser.add_argument("--row-number", nargs="+", type=int, help="One or more 1-based metadata row numbers to remove.")
    parser.add_argument("--match", action="append", default=[], help="Additional metadata selector in the form field=value. Can be repeated.")
    parser.add_argument("--features-only", action="store_true", help="Only rewrite features_example.csv/parquet.")
    parser.add_argument("--metadata-only", action="store_true", help="Only rewrite samples_metadata.csv/parquet.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be removed without rewriting files.")
    return parser.parse_args()


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


def load_dataset_frame(csv_path: Path, parquet_path: Path, default_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    columns = read_header(csv_path)
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

    if default_columns:
        columns = unique_preserve_order(list(columns) + list(default_columns))
    if columns:
        df = df.reindex(columns=columns, fill_value="")
    return df, columns


def parse_match_filters(raw_filters: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for raw_filter in raw_filters:
        if "=" not in raw_filter:
            raise ValueError(f"Invalid --match value: {raw_filter}. Expected field=value.")
        field, value = raw_filter.split("=", 1)
        field = field.strip()
        if not field:
            raise ValueError(f"Invalid --match value: {raw_filter}. Field name is empty.")
        filters[field] = value.strip()
    return filters


def build_selection_mask(
    df: pd.DataFrame,
    *,
    clear_all: bool,
    sample_ids: list[str] | None = None,
    sha256s: list[str] | None = None,
    row_numbers: list[int] | None = None,
    match_filters: dict[str, str] | None = None,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    if clear_all:
        return pd.Series([True] * len(df), index=df.index)

    mask = pd.Series([False] * len(df), index=df.index)
    if sample_ids:
        if "sample_id" not in df.columns:
            raise ValueError("Dataset does not contain sample_id column.")
        mask |= df["sample_id"].astype(str).isin([str(value) for value in sample_ids])
    if sha256s:
        if "sha256" not in df.columns:
            raise ValueError("Dataset does not contain sha256 column.")
        normalized = {value.strip().casefold() for value in sha256s}
        mask |= df["sha256"].astype(str).str.strip().str.casefold().isin(normalized)
    if row_numbers:
        valid_numbers = {value for value in row_numbers if value >= 1}
        if not valid_numbers:
            raise ValueError("--row-number values must be 1-based positive integers.")
        metadata_positions = pd.Series(range(1, len(df) + 1), index=df.index)
        mask |= metadata_positions.isin(valid_numbers)
    for field, expected in (match_filters or {}).items():
        if field not in df.columns:
            raise ValueError(f"Dataset does not contain field '{field}'.")
        mask |= df[field].astype(str) == expected

    return mask


def write_dataset_files(csv_path: Path, parquet_path: Path, df: pd.DataFrame, header: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(header) if header else list(df.columns)
    out_df = df.reindex(columns=ordered, fill_value="") if ordered else df.copy()

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for record in out_df.to_dict(orient="records"):
            writer.writerow({field: record.get(field, "") for field in ordered})

    out_df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")


def clear_dataset(
    *,
    project_root: str | Path,
    clear_all: bool = False,
    sample_ids: list[str] | None = None,
    sha256s: list[str] | None = None,
    row_numbers: list[int] | None = None,
    match_filters: dict[str, str] | None = None,
    features_only: bool = False,
    metadata_only: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    if features_only and metadata_only:
        raise ValueError("Use either --features-only or --metadata-only, not both.")
    if not clear_all and not any([sample_ids, sha256s, row_numbers, match_filters]):
        raise ValueError("Provide at least one selector or pass --all.")

    paths = dataset_paths(project_root)
    metadata_df, metadata_header = load_dataset_frame(paths.metadata_csv, paths.metadata_parquet, METADATA_FIELDNAMES)
    feature_df, feature_header = load_dataset_frame(paths.features_csv, paths.features_parquet, ["sample_id", "sha256", "label"])

    if metadata_only:
        metadata_mask = build_selection_mask(
            metadata_df,
            clear_all=clear_all,
            sample_ids=sample_ids,
            sha256s=sha256s,
            row_numbers=row_numbers,
            match_filters=match_filters,
        )
        feature_mask = pd.Series([False] * len(feature_df), index=feature_df.index)
    elif features_only:
        feature_mask = build_selection_mask(
            feature_df,
            clear_all=clear_all,
            sample_ids=sample_ids,
            sha256s=sha256s,
            row_numbers=row_numbers,
            match_filters=match_filters,
        )
        metadata_mask = pd.Series([False] * len(metadata_df), index=metadata_df.index)
    else:
        metadata_mask = build_selection_mask(
            metadata_df,
            clear_all=clear_all,
            sample_ids=sample_ids,
            sha256s=sha256s,
            row_numbers=row_numbers,
            match_filters=match_filters,
        )
        selected_sample_ids = set(metadata_df.loc[metadata_mask, "sample_id"].astype(str)) if "sample_id" in metadata_df.columns else set()
        selected_sha256s = set(metadata_df.loc[metadata_mask, "sha256"].astype(str).str.strip().str.casefold()) if "sha256" in metadata_df.columns else set()
        feature_mask = pd.Series([False] * len(feature_df), index=feature_df.index)
        if not feature_df.empty:
            if selected_sample_ids and "sample_id" in feature_df.columns:
                feature_mask |= feature_df["sample_id"].astype(str).isin(selected_sample_ids)
            if selected_sha256s and "sha256" in feature_df.columns:
                feature_mask |= feature_df["sha256"].astype(str).str.strip().str.casefold().isin(selected_sha256s)

    metadata_removed = int(metadata_mask.sum()) if not metadata_mask.empty else 0
    features_removed = int(feature_mask.sum()) if not feature_mask.empty else 0

    if dry_run:
        return {
            "metadata_removed": metadata_removed,
            "features_removed": features_removed,
            "metadata_remaining": len(metadata_df) - metadata_removed,
            "features_remaining": len(feature_df) - features_removed,
        }

    if not features_only:
        metadata_out = metadata_df.loc[~metadata_mask].copy() if not metadata_mask.empty else metadata_df.copy()
        write_dataset_files(paths.metadata_csv, paths.metadata_parquet, metadata_out, metadata_header)
    if not metadata_only:
        feature_out = feature_df.loc[~feature_mask].copy() if not feature_mask.empty else feature_df.copy()
        write_dataset_files(paths.features_csv, paths.features_parquet, feature_out, feature_header)

    return {
        "metadata_removed": metadata_removed,
        "features_removed": features_removed,
        "metadata_remaining": len(metadata_df) - metadata_removed,
        "features_remaining": len(feature_df) - features_removed,
    }


def main() -> int:
    args = parse_args()
    stats = clear_dataset(
        project_root=args.project_root,
        clear_all=args.all,
        sample_ids=args.sample_id,
        sha256s=args.sha256,
        row_numbers=args.row_number,
        match_filters=parse_match_filters(args.match),
        features_only=args.features_only,
        metadata_only=args.metadata_only,
        dry_run=args.dry_run,
    )

    print(f"metadata_removed={stats['metadata_removed']}")
    print(f"features_removed={stats['features_removed']}")
    print(f"metadata_remaining={stats['metadata_remaining']}")
    print(f"features_remaining={stats['features_remaining']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())