"""Append one extracted feature row and metadata row into the aggregate datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import append_dataset_rows, apply_append_overrides, read_single_row_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append extracted single-row CSV files into the dataset stores.")
    parser.add_argument("features_csv", help="Path to the one-row features CSV produced by main_extractor.py.")
    parser.add_argument("metadata_csv", help="Path to the one-row metadata CSV produced by main_extractor.py.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]), help="Project root path.")
    parser.add_argument("--label", choices=["0", "1"], help="Explicit dataset label override.")
    parser.add_argument("--label-confidence", help="Explicit label confidence override.")
    parser.add_argument("--download-date", help="Explicit download/collection date in YYYY-MM-DD format.")
    parser.add_argument("--source", help="Optional source override before append.")
    parser.add_argument("--source-family", help="Optional source_family override before append.")
    parser.add_argument("--av-detection-count", help="Optional AV detection count override before append.")
    parser.add_argument("--allow-missing-label", action="store_true", help="Append even if no label can be inferred or provided.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = Path(args.features_csv).resolve()
    metadata_path = Path(args.metadata_csv).resolve()
    if not features_path.exists() or not metadata_path.exists():
        print("Both feature and metadata CSV files must exist.", file=sys.stderr)
        return 2

    feature_fieldnames, feature_row = read_single_row_csv(features_path)
    _, metadata_row = read_single_row_csv(metadata_path)
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
    )
    assigned_id, *paths = append_dataset_rows(
        feature_row=feature_row,
        metadata_row=metadata_row,
        feature_fieldnames=feature_fieldnames,
        project_root=Path(args.project_root).resolve(),
    )
    print(f"features_csv={paths[0]}")
    print(f"features_parquet={paths[1]}")
    print(f"metadata_csv={paths[2]}")
    print(f"metadata_parquet={paths[3]}")
    print(f"sample_id={assigned_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())