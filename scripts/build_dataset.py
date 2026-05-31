"""Run extraction and dataset append in one step for a file or directory."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_extractor import build_extraction_context, extract_file_rows
from src.dataset import append_dataset_rows


DEFAULT_EXTENSIONS = ("exe", "dll", "sys", "fon")


@dataclass
class DatasetBuildSummary:
    scanned: int = 0
    appended: int = 0
    skipped_duplicates: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)
    last_outputs: tuple[Path, Path, Path, Path] | None = None  # (features_csv, features_parquet, metadata_csv, metadata_parquet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one file or a whole folder and append rows into the aggregate dataset.")
    parser.add_argument("input_path", help="Path to a file or directory with samples.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root path.")
    parser.add_argument("--label", choices=["0", "1"], help="Explicit label for all processed samples.")
    parser.add_argument("--label-confidence", help="Optional label confidence for all processed samples.")
    parser.add_argument("--download-date", help="Optional download/collection date in YYYY-MM-DD format.")
    parser.add_argument("--source", help="Optional source override for all processed samples.")
    parser.add_argument("--source-family", help="Optional source_family override for all processed samples.")
    parser.add_argument("--av-detection-count", help="Optional AV detection count override for all processed samples.")
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS), help="Comma-separated extensions for directory mode (default: exe,dll,sys,fon).")
    parser.add_argument("--all-files", action="store_true", help="In directory mode, analyze every file regardless of extension.")
    parser.add_argument("--non-recursive", action="store_true", help="Do not recurse into subdirectories.")
    parser.add_argument("--allow-missing-label", action="store_true", help="Append samples even if no label can be inferred or provided.")
    parser.add_argument("--allow-duplicates", action="store_true", help="Append duplicate SHA-256 rows instead of skipping them.")
    virustotal_group = parser.add_mutually_exclusive_group()
    virustotal_group.add_argument(
        "--enable-virustotal",
        dest="enable_virustotal",
        action="store_true",
        help="Enable VirusTotal hash lookups for this run.",
    )
    virustotal_group.add_argument(
        "--skip-virustotal",
        dest="enable_virustotal",
        action="store_false",
        help="Disable VirusTotal hash lookups for this run.",
    )
    parser.set_defaults(enable_virustotal=False)
    parser.add_argument("--skip-floss", action="store_true", help="Disable FLOSS string extraction for this run.")
    return parser.parse_args()


def parse_extensions(raw_extensions: str) -> set[str]:
    values = {
        item.strip().casefold().lstrip(".")
        for item in raw_extensions.split(",")
        if item.strip()
    }
    return values or set(DEFAULT_EXTENSIONS)


def collect_input_files(
    input_path: str | Path,
    *,
    recursive: bool = True,
    extensions: set[str] | None = None,
    all_files: bool = False,
) -> list[Path]:
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"Unsupported input path: {path}")

    allowed_extensions = {value.casefold().lstrip(".") for value in (extensions or set(DEFAULT_EXTENSIONS))}
    iterator = path.rglob("*") if recursive else path.iterdir()
    files = []
    for candidate in iterator:
        if not candidate.is_file():
            continue
        if all_files:
            files.append(candidate.resolve())
            continue
        suffix = candidate.suffix.lstrip(".").casefold()
        if suffix in allowed_extensions:
            files.append(candidate.resolve())
    files.sort(key=lambda item: str(item).casefold())
    return files


def load_existing_sha256s(project_root: str | Path) -> set[str]:
    metadata_csv = Path(project_root).resolve() / "data" / "metadata" / "samples_metadata.csv"
    if not metadata_csv.exists():
        return set()
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("sha256", "")).strip().casefold()
            for row in reader
            if str(row.get("sha256", "")).strip()
        }


def run_dataset_pipeline(
    input_path: str | Path,
    *,
    project_root: str | Path,
    label: str | None = None,
    label_confidence: str | None = None,
    download_date: str | None = None,
    source: str | None = None,
    source_family: str | None = None,
    av_detection_count: str | int | None = None,
    extensions: set[str] | None = None,
    recursive: bool = True,
    all_files: bool = False,
    allow_missing_label: bool = False,
    allow_duplicates: bool = False,
    enable_virustotal: bool = False,
    skip_floss: bool = False,
) -> DatasetBuildSummary:
    root = Path(project_root).resolve()
    context = build_extraction_context(root, enable_virustotal=enable_virustotal, skip_floss=skip_floss)
    candidates = collect_input_files(
        input_path,
        recursive=recursive,
        extensions=extensions,
        all_files=all_files,
    )

    summary = DatasetBuildSummary(scanned=len(candidates))
    seen_sha256 = set() if allow_duplicates else load_existing_sha256s(root)

    for candidate in candidates:
        try:
            feature_fieldnames, feature_row, metadata_row = extract_file_rows(
                candidate,
                context=context,
                label=label,
                label_confidence=label_confidence,
                download_date=download_date,
                source=source,
                source_family=source_family,
                av_detection_count=av_detection_count,
            )

            resolved_label = str(metadata_row.get("label", "")).strip()
            if resolved_label == "" and not allow_missing_label:
                raise ValueError(
                    "Label is required. Pass --label, set source/source_family so it can be inferred, or use --allow-missing-label."
                )

            sha256 = str(metadata_row.get("sha256", "")).strip().casefold()
            if not allow_duplicates and sha256 in seen_sha256:
                summary.skipped_duplicates += 1
                print(f"[SKIP] duplicate sha256 for {candidate}")
                continue

            assigned_id, *paths = append_dataset_rows(
                feature_row=feature_row,
                metadata_row=metadata_row,
                feature_fieldnames=feature_fieldnames,
                project_root=root,
            )
            seen_sha256.add(sha256)
            summary.appended += 1
            summary.last_outputs = tuple(paths)
            print(
                f"[OK] {candidate} -> sample_id={assigned_id} "
                f"label={metadata_row['label']} pe_kind={metadata_row['pe_kind']}"
            )
        except Exception as exc:
            summary.failed += 1
            message = f"{candidate}: {exc}"
            summary.failures.append(message)
            print(f"[FAIL] {message}", file=sys.stderr)

    return summary


def main() -> int:
    args = parse_args()
    summary = run_dataset_pipeline(
        args.input_path,
        project_root=args.project_root,
        label=args.label,
        label_confidence=args.label_confidence,
        download_date=args.download_date,
        source=args.source,
        source_family=args.source_family,
        av_detection_count=args.av_detection_count,
        extensions=parse_extensions(args.extensions),
        recursive=not args.non_recursive,
        all_files=args.all_files,
        allow_missing_label=args.allow_missing_label,
        allow_duplicates=args.allow_duplicates,
        enable_virustotal=args.enable_virustotal,
        skip_floss=args.skip_floss,
    )

    project_root = Path(args.project_root).resolve()
    print(f"scanned={summary.scanned}")
    print(f"appended={summary.appended}")
    print(f"skipped_duplicates={summary.skipped_duplicates}")
    print(f"failed={summary.failed}")
    print(f"features_csv={project_root / 'data' / 'features' / 'features_example.csv'}")
    print(f"features_parquet={project_root / 'data' / 'features' / 'features_example.parquet'}")
    print(f"metadata_csv={project_root / 'data' / 'metadata' / 'samples_metadata.csv'}")
    print(f"metadata_parquet={project_root / 'data' / 'metadata' / 'samples_metadata.parquet'}")

    if summary.scanned == 0:
        print("No files matched the requested input/filter.", file=sys.stderr)
        return 2
    if summary.failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())