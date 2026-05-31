"""Manage aggregate dataset records in CSV and Parquet stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    DatasetManager,
    DatasetSelection,
    apply_append_overrides,
    parse_key_value_pairs,
    read_single_row_csv,
)
from src.dataset.dataset_manager import write_dataset_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage aggregate dataset files: list, add, edit, or delete records in features_example.* and samples_metadata.*."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root path.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List matching dataset records.")
    add_selection_args(list_parser, require_selector=False)
    list_parser.add_argument("--columns", help="Comma-separated columns to include in output.")
    list_parser.add_argument("--limit", type=int, help="Maximum number of rows to print.")

    add_parser = subparsers.add_parser("add", help="Append one extracted sample into the aggregate dataset.")
    add_parser.add_argument("features_csv", help="Path to the one-row features CSV produced by main_extractor.py.")
    add_parser.add_argument("metadata_csv", help="Path to the one-row metadata CSV produced by main_extractor.py.")
    add_parser.add_argument("--label", choices=["0", "1"], help="Explicit dataset label override.")
    add_parser.add_argument("--label-confidence", help="Explicit label confidence override.")
    add_parser.add_argument("--download-date", help="Explicit download/collection date in YYYY-MM-DD format.")
    add_parser.add_argument("--source", help="Optional source override before append.")
    add_parser.add_argument("--source-family", help="Optional source_family override before append.")
    add_parser.add_argument("--av-detection-count", help="Optional AV detection count override before append.")
    add_parser.add_argument("--allow-missing-label", action="store_true", help="Append even if no label can be inferred or provided.")
    add_parser.add_argument("--allow-duplicates", action="store_true", help="Append duplicate SHA-256 rows instead of rejecting them.")
    add_parser.add_argument("--dry-run", action="store_true", help="Show what would be appended without rewriting files.")
    add_update_args(add_parser)

    edit_parser = subparsers.add_parser("edit", help="Edit matching records in the aggregate dataset.")
    add_selection_args(edit_parser, require_selector=True)
    add_update_args(edit_parser)
    edit_parser.add_argument("--metadata-only", action="store_true", help="Update only samples_metadata.*.")
    edit_parser.add_argument("--features-only", action="store_true", help="Update only features_example.*.")
    edit_parser.add_argument("--dry-run", action="store_true", help="Show how many rows would be updated without rewriting files.")

    delete_parser = subparsers.add_parser("delete", help="Delete matching records from the aggregate dataset.")
    add_selection_args(delete_parser, require_selector=True)
    delete_parser.add_argument("--metadata-only", action="store_true", help="Delete only from samples_metadata.*.")
    delete_parser.add_argument("--features-only", action="store_true", help="Delete only from features_example.*.")
    delete_parser.add_argument("--dry-run", action="store_true", help="Show how many rows would be deleted without rewriting files.")
    reindex_parser = subparsers.add_parser(
        "reindex",
        help="Reassign sample_id sequentially from 1 to N, fixing gaps after deletions.",
    )
    reindex_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned renumbering without rewriting files.",
    )
    return parser.parse_args()


def add_selection_args(parser: argparse.ArgumentParser, *, require_selector: bool) -> None:
    parser.add_argument("--all", action="store_true", help="Select all records.")
    parser.add_argument("--sample-id", nargs="+", help="One or more sample_id values.")
    parser.add_argument("--sha256", nargs="+", help="One or more SHA-256 values.")
    parser.add_argument("--row-number", nargs="+", type=int, help="One or more 1-based joined dataset row numbers.")
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="Field selector in the form field=value. Repeat to require multiple field matches.",
    )
    parser.set_defaults(require_selector=require_selector)


def add_update_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="shared_updates",
        action="append",
        default=[],
        help="Shared update field=value for columns present in both datasets: sample_id, sha256, label.",
    )
    parser.add_argument(
        "--set-meta",
        dest="metadata_updates",
        action="append",
        default=[],
        help="Metadata-only update in the form field=value.",
    )
    parser.add_argument(
        "--set-feature",
        dest="feature_updates",
        action="append",
        default=[],
        help="Feature-only update in the form field=value.",
    )


def build_selection(args: argparse.Namespace, *, allow_empty: bool = False) -> DatasetSelection:
    selection = DatasetSelection(
        select_all=bool(getattr(args, "all", False)),
        sample_ids=tuple(str(value) for value in (getattr(args, "sample_id", None) or [])),
        sha256s=tuple(str(value) for value in (getattr(args, "sha256", None) or [])),
        row_numbers=tuple(int(value) for value in (getattr(args, "row_number", None) or [])),
        match_filters=tuple(parse_key_value_pairs(getattr(args, "match", [])).items()),
    )
    if selection.is_empty() and not allow_empty:
        raise ValueError("Provide at least one selector or pass --all.")
    return selection


def parse_columns(raw_columns: str | None) -> list[str] | None:
    if not raw_columns:
        return None
    values = [item.strip() for item in raw_columns.split(",") if item.strip()]
    return values or None


def print_key_value_stats(stats: dict[str, object]) -> None:
    for key, value in stats.items():
        print(f"{key}={value}")


def handle_list(args: argparse.Namespace, manager: DatasetManager) -> int:
    selection = build_selection(args, allow_empty=True)
    if selection.is_empty():
        selection = DatasetSelection(select_all=True)
    rows = manager.list_records(
        selection,
        columns=parse_columns(args.columns),
        limit=args.limit,
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


def handle_add(args: argparse.Namespace, manager: DatasetManager) -> int:
    feature_fieldnames, feature_row = read_single_row_csv(args.features_csv)
    _, metadata_row = read_single_row_csv(args.metadata_csv)

    metadata_row, feature_row = apply_append_overrides(
        metadata_row,
        feature_row,
        label=args.label,
        label_confidence=args.label_confidence,
        download_date=args.download_date,
        source=args.source,
        source_family=args.source_family,
        av_detection_count=args.av_detection_count,
        allow_missing_label=args.allow_missing_label,
        shared_updates=parse_key_value_pairs(args.shared_updates),
        metadata_updates=parse_key_value_pairs(args.metadata_updates),
        feature_updates=parse_key_value_pairs(args.feature_updates),
    )

    stats = manager.add_rows(
        feature_row=feature_row,
        metadata_row=metadata_row,
        feature_fieldnames=feature_fieldnames,
        allow_duplicates=args.allow_duplicates,
        dry_run=args.dry_run,
    )
    print_key_value_stats(stats)
    return 0


def handle_edit(args: argparse.Namespace, manager: DatasetManager) -> int:
    stats = manager.edit_records(
        build_selection(args),
        shared_updates=parse_key_value_pairs(args.shared_updates),
        metadata_updates=parse_key_value_pairs(args.metadata_updates),
        feature_updates=parse_key_value_pairs(args.feature_updates),
        metadata_only=args.metadata_only,
        features_only=args.features_only,
        dry_run=args.dry_run,
    )
    print_key_value_stats(stats)
    return 0


def handle_delete(args: argparse.Namespace, manager: DatasetManager) -> int:
    stats = manager.delete_records(
        build_selection(args),
        metadata_only=args.metadata_only,
        features_only=args.features_only,
        dry_run=args.dry_run,
    )
    print_key_value_stats(stats)
    return 0


def handle_reindex(args: argparse.Namespace, manager: DatasetManager) -> int:
    tables = manager.load_tables()
    meta = tables.metadata_df
    feat = tables.feature_df

    # Sort both tables by current numeric sample_id to preserve original order.
    def _sort_key(df: "pd.DataFrame") -> "pd.DataFrame":
        try:
            return df.assign(_sid_int=df["sample_id"].astype(int)).sort_values("_sid_int").drop(columns=["_sid_int"]).reset_index(drop=True)
        except (ValueError, KeyError):
            return df.reset_index(drop=True)

    meta = _sort_key(meta)
    feat = _sort_key(feat)

    old_meta_ids = list(meta["sample_id"])
    new_ids = [str(i) for i in range(1, len(meta) + 1)]

    # Build a mapping old_id -> new_id for display and feature sync.
    id_map = dict(zip(old_meta_ids, new_ids))

    changed = [(old, new) for old, new in id_map.items() if old != new]
    print(f"total_rows={len(meta)}")
    print(f"changed_ids={len(changed)}")

    if args.dry_run:
        for old, new in changed:
            print(f"  sample_id {old} -> {new}")
        print("dry_run=True — no files written.")
        return 0

    meta["sample_id"] = new_ids

    # Apply the same mapping to the features table (match by old sample_id value).
    feat["sample_id"] = feat["sample_id"].map(id_map).fillna(feat["sample_id"])

    write_dataset_files(
        manager.paths.metadata_csv,
        manager.paths.metadata_parquet,
        meta,
        tables.metadata_header,
    )
    write_dataset_files(
        manager.paths.features_csv,
        manager.paths.features_parquet,
        feat,
        tables.feature_header,
        drop_deprecated_features=True,
    )
    print("status=ok")
    return 0


def main() -> int:
    args = parse_args()
    manager = DatasetManager(args.project_root)

    if args.command == "list":
        return handle_list(args, manager)
    if args.command == "add":
        return handle_add(args, manager)
    if args.command == "edit":
        return handle_edit(args, manager)
    if args.command == "delete":
        return handle_delete(args, manager)
    if args.command == "reindex":
        return handle_reindex(args, manager)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())