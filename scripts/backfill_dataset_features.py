"""Re-extract feature rows from raw files and update existing aggregate dataset rows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_extractor import build_extraction_context, extract_file_rows
from scripts.build_dataset import DEFAULT_EXTENSIONS, collect_input_files, parse_extensions
from src.dataset import DatasetManager


@dataclass
class FeatureBackfillSummary:
    scanned: int = 0
    extracted: int = 0
    matched_files: int = 0
    unmatched_files: int = 0
    updated_rows: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)
    added_feature_columns: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-extract features for one file or folder and backfill matching aggregate dataset rows by SHA-256."
        )
    )
    parser.add_argument("input_path", help="Path to a file or directory with raw samples.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root path.")
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated extensions for directory mode (default: exe,dll,sys,fon).",
    )
    parser.add_argument("--all-files", action="store_true", help="In directory mode, analyze every file regardless of extension.")
    parser.add_argument("--non-recursive", action="store_true", help="Do not recurse into subdirectories.")
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
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without rewriting files.")
    return parser.parse_args()


def run_feature_backfill(
    input_path: str | Path,
    *,
    project_root: str | Path,
    extensions: set[str] | None = None,
    recursive: bool = True,
    all_files: bool = False,
    enable_virustotal: bool = False,
    skip_floss: bool = False,
    dry_run: bool = False,
) -> FeatureBackfillSummary:
    root = Path(project_root).resolve()
    context = build_extraction_context(
        root,
        enable_virustotal=enable_virustotal,
        skip_floss=skip_floss,
    )
    candidates = collect_input_files(
        input_path,
        recursive=recursive,
        extensions=extensions,
        all_files=all_files,
    )

    summary = FeatureBackfillSummary(scanned=len(candidates))
    feature_rows: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            _, feature_row, _ = extract_file_rows(candidate, context=context)
            feature_rows.append(feature_row)
            summary.extracted += 1
        except Exception as exc:
            summary.failed += 1
            message = f"{candidate}: {exc}"
            summary.failures.append(message)
            print(f"[FAIL] {message}", file=sys.stderr)

    stats = DatasetManager(root).backfill_feature_rows(feature_rows, dry_run=dry_run)
    summary.matched_files = int(stats.get("matched_files", 0))
    summary.unmatched_files = int(stats.get("unmatched_files", 0))
    summary.updated_rows = int(stats.get("updated_rows", 0))
    summary.added_feature_columns = [str(value) for value in stats.get("added_feature_columns", [])]
    return summary


def main() -> int:
    args = parse_args()
    summary = run_feature_backfill(
        args.input_path,
        project_root=args.project_root,
        extensions=parse_extensions(args.extensions),
        recursive=not args.non_recursive,
        all_files=args.all_files,
        enable_virustotal=args.enable_virustotal,
        skip_floss=args.skip_floss,
        dry_run=args.dry_run,
    )

    print(f"scanned={summary.scanned}")
    print(f"extracted={summary.extracted}")
    print(f"matched_files={summary.matched_files}")
    print(f"unmatched_files={summary.unmatched_files}")
    print(f"updated_rows={summary.updated_rows}")
    print(f"failed={summary.failed}")
    print(f"added_feature_columns={','.join(summary.added_feature_columns)}")

    if summary.scanned == 0:
        print("No files matched the requested input/filter.", file=sys.stderr)
        return 2
    if summary.failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())