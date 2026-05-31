from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.evaluation import DEFAULT_EXTERNAL_SPLIT_NAME
from src.evaluation import build_evaluation_report
from src.evaluation import load_prediction_table_from_batch_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a detailed evaluation report from experiment_log.json.")
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
        help="Path to experiment_log.json. Defaults to the canonical models/experiment_log.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory for evaluation_report.{json,md}.",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=25,
        help="How many FP/FN rows to keep per split in the report.",
    )
    parser.add_argument(
        "--policy-file",
        type=Path,
        default=None,
        help="Optional YAML/JSON operating-point policy file. Defaults to Config.operating_point_policy_path.",
    )
    parser.add_argument(
        "--batch-summary",
        type=Path,
        default=None,
        help="Optional summary.csv or summary.json produced by analyze_directory.py for an external labeled holdout slice.",
    )
    parser.add_argument(
        "--label-manifest",
        type=Path,
        default=None,
        help="Manifest CSV for the external holdout slice. Required with --batch-summary.",
    )
    parser.add_argument(
        "--external-split-name",
        default=DEFAULT_EXTERNAL_SPLIT_NAME,
        help="Report split name for the external holdout slice.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT
    config = Config.load(args.config_file).resolve(project_root)
    experiment_log_path = args.experiment_log or config.experiment_log_path

    if bool(args.batch_summary) != bool(args.label_manifest):
        raise ValueError("Pass both --batch-summary and --label-manifest together.")

    external_prediction_tables = None
    external_artifacts = None
    if args.batch_summary is not None and args.label_manifest is not None:
        external_prediction_tables = {
            args.external_split_name: load_prediction_table_from_batch_summary(
                args.batch_summary,
                args.label_manifest,
            )
        }
        external_artifacts = {
            "external_batch_summary_path": str(args.batch_summary.resolve()),
            "external_label_manifest_path": str(args.label_manifest.resolve()),
        }

    result = build_evaluation_report(
        experiment_log_path,
        output_root=args.output_root,
        top_errors=args.top_errors,
        operating_point_policy=args.policy_file or config.operating_point_policy_path,
        acceptance_criteria=config.acceptance_criteria_path,
        external_prediction_tables=external_prediction_tables,
        external_artifacts=external_artifacts,
    )

    operating_point = result.report.get("operating_point", {})
    acceptance = result.report.get("acceptance_criteria", {})
    val_metrics = result.report["splits"]["val"]["overall_metrics"]
    test_metrics = result.report["splits"]["test"]["overall_metrics"]
    print(f"profile_slug={result.profile_slug}")
    print(f"operating_threshold={operating_point.get('selected_threshold')}")
    print(f"suspicious_threshold={operating_point.get('selected_suspicious_threshold')}")
    print(f"val_roc_auc={val_metrics.get('roc_auc')}")
    print(f"val_pr_auc={val_metrics.get('pr_auc')}")
    print(f"test_roc_auc={test_metrics.get('roc_auc')}")
    print(f"test_pr_auc={test_metrics.get('pr_auc')}")
    print(f"constraints_satisfied={operating_point.get('constraints_satisfied')}")
    print(f"acceptance_satisfied={acceptance.get('criteria_satisfied')}")
    print(f"report_json={result.artifact_paths.report_json_path}")
    print(f"report_markdown={result.artifact_paths.report_markdown_path}")
    if args.batch_summary is not None:
        external_metrics = result.report["splits"][args.external_split_name]["overall_metrics"]
        print(f"external_split={args.external_split_name}")
        print(f"external_false_positive_count={external_metrics.get('false_positive_count')}")
        print(f"external_false_positive_rate={external_metrics.get('false_positive_rate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())