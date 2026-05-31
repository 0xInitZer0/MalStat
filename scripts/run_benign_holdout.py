from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_benign_dll_sys_slice import DEFAULT_WINDOWS_ROOT
from scripts.collect_benign_dll_sys_slice import collect_benign_slice
from src.core.config import Config
from src.evaluation import build_evaluation_report
from src.evaluation import load_prediction_table_from_batch_summary
from src.inference import DEFAULT_BATCH_PATTERNS
from src.inference import analyze_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect, score, and evaluate an external benign DLL/SYS holdout in one command."
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Optional YAML config file compatible with src.core.config.Config.load().",
    )
    parser.add_argument(
        "--experiment-log",
        type=Path,
        default=None,
        help="Optional experiment_log.json to evaluate. Defaults to the canonical models/experiment_log.json.",
    )
    parser.add_argument(
        "--slice-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "benign_dll_sys",
        help="Root directory for the collected benign holdout files and manifest.",
    )
    parser.add_argument(
        "--batch-output-root",
        type=Path,
        default=None,
        help="Where analyze_directory artifacts should be written. Defaults to reports/batch_analysis/<split-name>.",
    )
    parser.add_argument(
        "--evaluation-output-root",
        type=Path,
        default=None,
        help="Optional override for evaluation_report.{json,md} output location.",
    )
    parser.add_argument(
        "--external-split-name",
        default=None,
        help="Split name to use in the evaluation report. Defaults to the slice directory name.",
    )
    parser.add_argument(
        "--dll-dir",
        type=Path,
        default=DEFAULT_WINDOWS_ROOT / "System32",
        help="Directory to scan for benign DLL files.",
    )
    parser.add_argument(
        "--driver-dir",
        type=Path,
        default=DEFAULT_WINDOWS_ROOT / "System32" / "drivers",
        help="Directory to scan for benign SYS files.",
    )
    parser.add_argument("--max-dlls", type=int, default=50, help="Maximum number of DLL files to copy.")
    parser.add_argument("--max-drivers", type=int, default=25, help="Maximum number of SYS files to copy.")
    parser.add_argument("--skip-first-dlls", type=int, default=0, help="Skip the first N smallest DLL candidates before copying.")
    parser.add_argument("--skip-first-drivers", type=int, default=0, help="Skip the first N smallest SYS candidates before copying.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear an existing collected slice and stale batch-analysis artifacts before rerunning.",
    )
    parser.add_argument(
        "--write-html",
        action="store_true",
        help="Also render per-file HTML reports during batch analysis.",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=25,
        help="How many FP/FN rows to keep per split in the evaluation report.",
    )
    return parser.parse_args()


def _resolve_batch_output_root(config: Config, split_name: str, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root.resolve()
    return (Path(config.reports_dir) / "batch_analysis" / split_name).resolve()


def _prepare_batch_output_root(batch_output_root: Path, overwrite: bool) -> Path:
    resolved_root = batch_output_root.resolve()
    if overwrite and resolved_root.exists():
        shutil.rmtree(resolved_root)
    return resolved_root


def main() -> int:
    args = parse_args()
    config = Config.load(args.config_file).resolve(PROJECT_ROOT)
    experiment_log_path = Path(args.experiment_log or config.experiment_log_path).resolve()
    slice_root = Path(args.slice_root).resolve()
    split_name = args.external_split_name or slice_root.name
    batch_output_root = _prepare_batch_output_root(
        _resolve_batch_output_root(config, split_name, args.batch_output_root),
        overwrite=args.overwrite,
    )

    slice_paths = collect_benign_slice(
        output_root=slice_root,
        dll_dir=args.dll_dir,
        driver_dir=args.driver_dir,
        max_dlls=args.max_dlls,
        max_drivers=args.max_drivers,
        skip_first_dlls=args.skip_first_dlls,
        skip_first_drivers=args.skip_first_drivers,
        overwrite=args.overwrite,
    )

    batch_result = analyze_directory(
        slice_paths.files_root,
        config=config,
        patterns=DEFAULT_BATCH_PATTERNS,
        recursive=True,
        output_root=batch_output_root,
        write_html=args.write_html,
        fail_fast=False,
    )

    external_table = load_prediction_table_from_batch_summary(
        batch_result.artifact_paths.summary_csv_path,
        slice_paths.manifest_path,
    )
    evaluation_result = build_evaluation_report(
        experiment_log_path,
        output_root=args.evaluation_output_root,
        top_errors=args.top_errors,
        operating_point_policy=config.operating_point_policy_path,
        acceptance_criteria=config.acceptance_criteria_path,
        external_prediction_tables={split_name: external_table},
        external_artifacts={
            "external_batch_summary_path": str(batch_result.artifact_paths.summary_csv_path),
            "external_label_manifest_path": str(slice_paths.manifest_path),
        },
    )

    operating_point = evaluation_result.report.get("operating_point", {})
    acceptance = evaluation_result.report.get("acceptance_criteria", {})
    external_metrics = evaluation_result.report["splits"][split_name]["overall_metrics"]

    print(f"slice_root={slice_paths.output_root}")
    print(f"files_root={slice_paths.files_root}")
    print(f"manifest_path={slice_paths.manifest_path}")
    print(f"batch_summary_csv={batch_result.artifact_paths.summary_csv_path}")
    print(f"processed_count={batch_result.processed_count}")
    print(f"failure_count={batch_result.failure_count}")
    print(f"operating_threshold={operating_point.get('selected_threshold')}")
    print(f"constraints_satisfied={operating_point.get('constraints_satisfied')}")
    print(f"acceptance_satisfied={acceptance.get('criteria_satisfied')}")
    print(f"external_split={split_name}")
    print(f"external_false_positive_count={external_metrics.get('false_positive_count')}")
    print(f"external_false_positive_rate={external_metrics.get('false_positive_rate')}")
    print(f"report_json={evaluation_result.artifact_paths.report_json_path}")
    print(f"report_markdown={evaluation_result.artifact_paths.report_markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())