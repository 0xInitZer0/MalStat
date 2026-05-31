"""
src/core/config.py

Centralized project configuration object.
Stores all paths and thresholds. GUI, CLI, and API entry points use the same Config.

Usage:
    from src.core.config import Config

    # Default configuration:
    config = Config()

    # From a YAML file:
    config = Config.load("configs/project_config.yaml")

    # In entry points:
    builder = ReportBuilder.from_config(config)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    """All project paths and parameters."""

    # --- Model paths ---
    model_path: Path = field(default_factory=lambda: Path("models/calibrated_model.pkl"))
    preprocessor_path: Path = field(default_factory=lambda: Path("models/preprocessor.pkl"))
    feature_columns_path: Path = field(default_factory=lambda: Path("models/feature_columns.json"))
    experiment_log_path: Path = field(default_factory=lambda: Path("models/experiment_log.json"))

    # --- Indicator configs ---
    indicators_dir: Path = field(default_factory=lambda: Path("configs/indicators"))
    verdict_rules_path: Path = field(default_factory=lambda: Path("configs/verdict_rules.yaml"))
    operating_point_policy_path: Path = field(default_factory=lambda: Path("configs/operating_point_policy.yaml"))
    acceptance_criteria_path: Path = field(default_factory=lambda: Path("configs/acceptance_criteria.yaml"))

    # --- Data ---
    raw_malware_dir: Path = field(default_factory=lambda: Path("data/raw/malware"))
    raw_benign_dir: Path = field(default_factory=lambda: Path("data/raw/benign"))
    metadata_path: Path = field(default_factory=lambda: Path("data/metadata/samples_metadata.csv"))
    features_raw_path: Path = field(default_factory=lambda: Path("data/features/features_raw.parquet"))
    features_train_path: Path = field(default_factory=lambda: Path("data/features/features_train.parquet"))
    features_val_path: Path = field(default_factory=lambda: Path("data/features/features_val.parquet"))
    features_test_path: Path = field(default_factory=lambda: Path("data/features/features_test.parquet"))

    # --- Reports ---
    template_path: Path = field(default_factory=lambda: Path("templates/report.html.j2"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))

    # --- Classification thresholds ---
    verdict_threshold: float = 0.5
    """Above this threshold: MALICIOUS."""
    suspicious_threshold: float = 0.35
    """Above this threshold: SUSPICIOUS."""

    # --- Dataset parameters ---
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42

    # --- Training parameters ---
    n_jobs: int = -1
    cv_folds: int = 3
    sample_weight_strategy: str = "benign_driver_hard_negative"
    """Training-time weighting strategy. 'benign_driver_hard_negative' upweights only label=0, pe_kind=driver."""

    def resolve(self, base_dir: Path | None = None) -> "Config":
        """
        Resolve all relative paths against base_dir.

        Args:
            base_dir: Project root. If None, the current working directory is used.
        """
        if base_dir is None:
            base_dir = Path.cwd()
        fields_to_resolve = [
            "model_path", "preprocessor_path", "feature_columns_path",
            "experiment_log_path", "indicators_dir", "verdict_rules_path",
            "operating_point_policy_path", "acceptance_criteria_path",
            "raw_malware_dir", "raw_benign_dir", "metadata_path",
            "features_raw_path", "features_train_path", "features_val_path",
            "features_test_path", "template_path", "reports_dir",
        ]
        for attr in fields_to_resolve:
            p = getattr(self, attr)
            if not Path(p).is_absolute():
                setattr(self, attr, base_dir / p)
        return self

    @classmethod
    def load(cls, config_file: str | Path | None = None) -> "Config":
        """
        Load configuration from a YAML file.
        If no file is provided, return the default Config().

        Args:
            config_file: Path to a YAML file whose keys match Config fields.
        """
        if config_file is None:
            return cls()
        import yaml
        with open(config_file, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        # Convert string paths to Path objects where needed
        path_fields = {
            "model_path", "preprocessor_path", "feature_columns_path",
            "experiment_log_path", "indicators_dir", "verdict_rules_path",
            "operating_point_policy_path", "acceptance_criteria_path",
            "raw_malware_dir", "raw_benign_dir", "metadata_path",
            "features_raw_path", "features_train_path", "features_val_path",
            "features_test_path", "template_path", "reports_dir",
        }
        for key in path_fields:
            if key in data:
                data[key] = Path(data[key])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
