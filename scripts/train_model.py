"""Train a minimal PE-only malware classifier from clear training datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.training.pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a minimal PE-only malware classifier. Defaults to the pure_static profile unless --profile is provided."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root path.")
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "data" / "clear_training"),
        help="Root directory containing clear training datasets and model_profiles.json.",
    )
    parser.add_argument(
        "--profile",
        help="Training profile slug. If omitted, the loader uses the recommended default profile from model_profiles.json.",
    )
    parser.add_argument(
        "--config-file",
        help="Optional YAML config file compatible with src.core.config.Config.load().",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config = Config.load(args.config_file).resolve(project_root)

    result = run_training_pipeline(
        project_root=project_root,
        dataset_root=args.dataset_root,
        profile=args.profile,
        config=config,
    )

    print(f"profile_slug={result.profile_slug}")
    print(f"selected_model={result.selected_model_name}")
    if result.operating_point is not None:
        print(f"operating_threshold={result.operating_point.get('selected_threshold')}")
    print(f"validation_roc_auc={result.validation_metrics.get('roc_auc')}")
    print(f"validation_pr_auc={result.validation_metrics.get('pr_auc')}")
    print(f"test_roc_auc={result.test_metrics.get('roc_auc')}")
    print(f"test_pr_auc={result.test_metrics.get('pr_auc')}")
    print(f"wrote_canonical_artifacts={result.wrote_canonical_artifacts}")
    if result.wrote_canonical_artifacts:
        print(f"model_path={result.artifact_paths.model_path}")
        print(f"preprocessor_path={result.artifact_paths.preprocessor_path}")
        print(f"feature_columns_path={result.artifact_paths.feature_columns_path}")
        print(f"experiment_log_path={result.artifact_paths.experiment_log_path}")
    print(f"profile_model_path={result.artifact_paths.profile_model_path}")
    print(f"profile_preprocessor_path={result.artifact_paths.profile_preprocessor_path}")
    print(f"profile_feature_columns_path={result.artifact_paths.profile_feature_columns_path}")
    print(f"profile_experiment_log_path={result.artifact_paths.profile_experiment_log_path}")
    if result.wrote_canonical_artifacts and result.artifact_paths.evaluation_report_path.exists():
        print(f"evaluation_report_path={result.artifact_paths.evaluation_report_path}")
        print(f"evaluation_markdown_path={result.artifact_paths.evaluation_markdown_path}")
    if result.artifact_paths.profile_evaluation_report_path.exists():
        print(f"profile_evaluation_report_path={result.artifact_paths.profile_evaluation_report_path}")
        print(f"profile_evaluation_markdown_path={result.artifact_paths.profile_evaluation_markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())