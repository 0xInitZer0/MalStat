"""Export a filtered training subset without modifying aggregate dataset stores."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.training_subset import export_clean_training_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a clean training subset from aggregate dataset stores without modifying the source dataset."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root containing data/features and data/metadata.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "clear_training"),
        help="Destination directory for the filtered training subset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_paths, summary, task_definition, profiles = export_clean_training_subset(args.project_root, args.output_root)

    print(f"source_rows={summary.source_rows}")
    print(f"kept_rows={summary.kept_rows}")
    print(f"excluded_rows={summary.excluded_rows}")
    print(f"excluded_low_confidence={summary.excluded_low_confidence}")
    print(f"excluded_invalid_pe={summary.excluded_invalid_pe}")
    print(f"excluded_non_pe={summary.excluded_non_pe}")
    print(f"excluded_unlabeled={summary.excluded_unlabeled}")
    print(f"metadata_csv={output_paths.metadata_csv}")
    print(f"metadata_parquet={output_paths.metadata_parquet}")
    print(f"features_csv={output_paths.features_csv}")
    print(f"features_parquet={output_paths.features_parquet}")
    print(f"summary_json={output_paths.summary_json}")
    print(f"task_slug={task_definition.task_slug}")
    print(f"task_definition_json={output_paths.task_definition_json}")
    print(f"model_profiles_json={output_paths.model_profiles_json}")
    print(f"recommended_default_profile={profiles['pure_static']['profile_slug']}")
    print(f"pure_static_features_csv={profiles['pure_static']['paths']['features_csv']}")
    print(f"pure_static_metadata_csv={profiles['pure_static']['paths']['metadata_csv']}")
    print(f"augmented_triage_features_csv={profiles['augmented_triage']['paths']['features_csv']}")
    print(f"augmented_triage_metadata_csv={profiles['augmented_triage']['paths']['metadata_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())