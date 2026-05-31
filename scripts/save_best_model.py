from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.core.path_utils import serialize_relative_path


CANONICAL_ARTIFACTS = [
    "models/calibrated_model.pkl",
    "models/preprocessor.pkl",
    "models/feature_columns.json",
    "models/experiment_log.json",
    "models/val_predictions.csv",
    "models/test_predictions.csv",
    "models/evaluation_report.json",
    "models/evaluation_report.md",
]


def _load_current_version(project_root: Path) -> str:
    experiment_log_path = project_root / "models" / "experiment_log.json"
    experiment_log = json.loads(experiment_log_path.read_text(encoding="utf-8"))
    version = str(experiment_log.get("run_timestamp_utc", "")).strip()
    if not version:
        raise ValueError(f"Could not resolve run_timestamp_utc from {experiment_log_path}")
    return version


def save_best_model_snapshot(project_root: Path) -> tuple[str, Path, Path]:
    resolved_root = project_root.resolve()
    version = _load_current_version(resolved_root)
    best_root = resolved_root / "models" / "best"
    snapshot_root = best_root / version
    snapshot_root.mkdir(parents=True, exist_ok=True)

    for relative_path in CANONICAL_ARTIFACTS:
        source_path = resolved_root / relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"Required artifact is missing: {source_path}")
        shutil.copy2(source_path, snapshot_root / source_path.name)

    created_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    current_manifest = {
        "best_version": version,
        "snapshot_root": serialize_relative_path(snapshot_root, base_dir=best_root),
        "created_at_utc": created_at_utc,
        "source_artifacts": CANONICAL_ARTIFACTS,
    }
    snapshot_manifest = {
        "best_version": version,
        "snapshot_root": serialize_relative_path(snapshot_root, base_dir=snapshot_root),
        "created_at_utc": created_at_utc,
        "source_artifacts": CANONICAL_ARTIFACTS,
    }
    manifest_path = best_root / "current_best.json"
    manifest_path.write_text(json.dumps(current_manifest, indent=2), encoding="utf-8")
    (snapshot_root / "manifest.json").write_text(json.dumps(snapshot_manifest, indent=2), encoding="utf-8")
    return version, snapshot_root, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Save the current canonical model artifacts as the best snapshot.")
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Project root containing the models directory.",
    )
    args = parser.parse_args()

    version, snapshot_root, manifest_path = save_best_model_snapshot(args.project_root)
    print(f"best_version={version}")
    print(f"snapshot_root={snapshot_root}")
    print(f"manifest_path={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())