"""
ReportBuilder is the facade for the entire analysis system.

It takes a path to a PE file and returns a fully populated AnalysisReport.
This is the only class the CLI, GUI, or Web API needs to know about.
"""

from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import warnings

from src.core.feature_augmentation import enrich_feature_dict
from src.core.path_utils import resolve_stored_path
from src.reporting.models import AnalysisReport, FeatureVerdict
from src.reporting.verdict_engine import VerdictEngine


def _current_sklearn_version() -> str:
    try:
        import sklearn
    except Exception:
        return ""
    return str(getattr(sklearn, "__version__", "") or "").strip()


def _load_joblib_artifact(path: Path) -> Any:
    import joblib

    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:
        InconsistentVersionWarning = None

    with warnings.catch_warnings():
        if InconsistentVersionWarning is not None:
            warnings.simplefilter("ignore", InconsistentVersionWarning)
        return joblib.load(path)


def _rebuild_runtime_preprocessor(config: Any, feature_meta: dict[str, Any]) -> Any:
    from src.training.loader import load_training_dataset
    from src.training.pipeline import TabularFeaturePreprocessor, _build_group_split, _infer_feature_types

    experiment_log_path = Path(config.experiment_log_path).resolve()
    if not experiment_log_path.exists():
        raise FileNotFoundError(
            "Cannot rebuild the runtime preprocessor because experiment_log.json is missing."
        )

    experiment_log = json.loads(experiment_log_path.read_text(encoding="utf-8"))
    dataset_root_value = experiment_log.get("dataset_root")
    if not dataset_root_value:
        raise ValueError("experiment_log.json does not contain dataset_root for preprocessor rebuild.")

    profile_slug = str(feature_meta.get("profile_slug") or experiment_log.get("profile_slug") or "").strip()
    resolved_profile = None if profile_slug in {"", "canonical"} else profile_slug
    dataset = load_training_dataset(
        resolve_stored_path(dataset_root_value, base_dir=experiment_log_path.parent),
        profile=resolved_profile,
    )

    feature_columns = [str(value) for value in feature_meta.get("feature_columns", [])]
    if not feature_columns:
        feature_columns = list(dataset.feature_columns)

    numeric_columns = [str(value) for value in feature_meta.get("numeric_feature_columns", [])]
    categorical_columns = [str(value) for value in feature_meta.get("categorical_feature_columns", [])]
    if not numeric_columns and not categorical_columns:
        numeric_columns, categorical_columns = _infer_feature_types(dataset.feature_df, feature_columns)

    preprocessor = TabularFeaturePreprocessor(
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    split = _build_group_split(dataset, config)
    preprocessor.fit(split.train_rows[feature_columns], split.train_rows["label"].astype(int))
    return preprocessor


def _resolve_runtime_thresholds(config: Any, feature_meta: dict[str, Any]) -> tuple[float, float]:
    from src.evaluation.model_evaluation import resolve_suspicious_threshold

    default_verdict_threshold = float(getattr(config, "verdict_threshold", 0.5))
    default_suspicious_threshold = float(getattr(config, "suspicious_threshold", 0.35))

    feature_threshold = feature_meta.get("operating_threshold")
    feature_suspicious_threshold = feature_meta.get("suspicious_threshold")
    if feature_threshold is not None:
        try:
            resolved_verdict_threshold = float(feature_threshold)
        except (TypeError, ValueError):
            resolved_verdict_threshold = default_verdict_threshold
        if feature_suspicious_threshold is not None:
            try:
                resolved_suspicious_threshold = float(feature_suspicious_threshold)
            except (TypeError, ValueError):
                resolved_suspicious_threshold = resolve_suspicious_threshold(
                    resolved_verdict_threshold,
                    getattr(config, "operating_point_policy_path", None),
                )
        else:
            resolved_suspicious_threshold = resolve_suspicious_threshold(
                resolved_verdict_threshold,
                getattr(config, "operating_point_policy_path", None),
            )
        return resolved_verdict_threshold, min(resolved_suspicious_threshold, resolved_verdict_threshold)

    experiment_log_path = Path(getattr(config, "experiment_log_path", "")).resolve()
    evaluation_report_path = experiment_log_path.with_name("evaluation_report.json")
    if evaluation_report_path.exists():
        try:
            evaluation_report = json.loads(evaluation_report_path.read_text(encoding="utf-8"))
            operating_point = evaluation_report.get("operating_point", {})
            selected_threshold = operating_point.get("selected_threshold")
            if selected_threshold is not None:
                resolved_verdict_threshold = float(selected_threshold)
                selected_suspicious_threshold = operating_point.get("selected_suspicious_threshold")
                if selected_suspicious_threshold is not None:
                    resolved_suspicious_threshold = float(selected_suspicious_threshold)
                else:
                    resolved_suspicious_threshold = resolve_suspicious_threshold(
                        resolved_verdict_threshold,
                        getattr(config, "operating_point_policy_path", None),
                    )
                return resolved_verdict_threshold, min(resolved_suspicious_threshold, resolved_verdict_threshold)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    return default_verdict_threshold, default_suspicious_threshold


class ReportBuilder:
    """
    Facade: PE file -> AnalysisReport.

    Dependencies are passed through the constructor using dependency injection.
    No hardcoded paths live inside the class.

    Args:
        pipeline:            FeaturePipeline that runs all extractors.
        model:               Calibrated model with predict_proba().
        preprocessor:        FeaturePreprocessor used to transform a new file.
        verdict_engine:      VerdictEngine that turns features into verdicts.
        feature_columns:     Feature list in the same order used during training.
        verdict_threshold:   Threshold above which the verdict is MALICIOUS.
        suspicious_threshold: Threshold above which the verdict is SUSPICIOUS.
    """

    def __init__(
        self,
        pipeline: Any,           # FeaturePipeline, imported only when used
        model: Any,              # IClassifierModel
        preprocessor: Any,       # FeaturePreprocessor
        verdict_engine: VerdictEngine,
        feature_columns: list[str],
        verdict_threshold: float = 0.5,
        suspicious_threshold: float = 0.35,
        model_name: str = "unknown",
        model_version: str = "unknown",
    ):
        self._pipeline = pipeline
        self._model = model
        self._preprocessor = preprocessor
        self._verdict_engine = verdict_engine
        self._feature_columns = feature_columns
        self._verdict_threshold = verdict_threshold
        self._suspicious_threshold = suspicious_threshold
        self._model_name = model_name
        self._model_version = model_version

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def analyze(self, pe_path: str | Path) -> AnalysisReport:
        """
        Run a full analysis of a PE file by path.

        Returns:
            AnalysisReport with populated verdicts, probability, and top factors.
        """
        pe_path = Path(pe_path)
        pe_bytes = pe_path.read_bytes()
        return self.analyze_bytes(pe_bytes, filename=pe_path.name, file_path=pe_path)

    def analyze_bytes(
        self,
        pe_bytes: bytes,
        filename: str = "unknown",
        file_path: str | Path | None = None,
    ) -> AnalysisReport:
        """
        Run a full analysis from PE file bytes.
        Used, for example, in FastAPI when the bytes come from a multipart upload.
        """
        resolved_path = Path(file_path) if file_path is not None else None
        sha256 = hashlib.sha256(pe_bytes).hexdigest()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Validate the PE file
        is_valid, reason = self._validate_pe(pe_bytes)

        report = AnalysisReport(
            filename=filename,
            sha256=sha256,
            file_size=len(pe_bytes),
            analysis_timestamp=timestamp,
            is_pe_valid=is_valid,
            pe_invalid_reason=reason if not is_valid else None,
            model_name=self._model_name,
            model_version=self._model_version,
            verdict_threshold=self._verdict_threshold,
        )

        if not is_valid:
            report.verdict = "out_of_scope"
            return report

        # 2. Extract features
        raw_features = self._pipeline.run(pe_bytes, file_path=resolved_path)
        report.raw_features = raw_features
        model_features = enrich_feature_dict(raw_features)

        # 3. Preprocess and predict
        probability = self._predict(model_features)
        report.probability = probability
        report.verdict = self._assign_verdict(probability)

        # 4. Collect SHAP values, if available
        report.shap_values = self._get_shap(model_features)

        # 5. Compute per-feature verdicts
        verdicts = self._verdict_engine.evaluate(raw_features)
        report.feature_verdicts = verdicts

        # 6. Select top factors, sorted by severity and then SHAP value
        report.top_factors = self._select_top_factors(verdicts, report.shap_values, n=15)

        return report

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _validate_pe(self, pe_bytes: bytes) -> tuple[bool, str | None]:
        if len(pe_bytes) < 512:
            return False, "File is too small (< 512 bytes)"
        if pe_bytes[:2] != b"MZ":
            return False, "Missing MZ signature (not a PE file)"
        try:
            import pefile
            pefile.PE(data=pe_bytes, fast_load=True)
        except Exception as e:
            return False, f"pefile error: {e}"
        return True, None

    def _predict(self, raw_features: dict[str, Any]) -> float:
        import pandas as pd
        import numpy as np

        df = pd.DataFrame([raw_features])
        # Add missing columns with NaN and drop extra ones
        for col in self._feature_columns:
            if col not in df.columns:
                df[col] = None
        df = df[self._feature_columns]

        X = self._preprocessor.transform(df)
        proba = self._model.predict_proba(X)
        # predict_proba returns [[p_benign, p_malware]]
        if hasattr(proba, "tolist"):
            proba = proba.tolist()
        p_malware = proba[0][1]
        return float(p_malware)

    def _assign_verdict(self, probability: float) -> str:
        if probability >= self._verdict_threshold:
            return "malicious"
        if probability >= self._suspicious_threshold:
            return "suspicious"
        return "benign"

    def _get_shap(self, raw_features: dict[str, Any]) -> dict[str, float] | None:
        try:
            import shap
            import pandas as pd

            df = pd.DataFrame([raw_features])
            for col in self._feature_columns:
                if col not in df.columns:
                    df[col] = None
            df = df[self._feature_columns]
            X = self._preprocessor.transform(df)

            explainer = shap.TreeExplainer(self._model)
            shap_vals = explainer.shap_values(X)
            # shap_vals shape: (1, n_features) or [(1, n) x n_classes]
            if isinstance(shap_vals, list):
                shap_arr = shap_vals[1][0]  # malware class
            else:
                shap_arr = shap_vals[0]
            return dict(zip(self._feature_columns, [float(v) for v in shap_arr]))
        except Exception:
            return None

    @staticmethod
    def _select_top_factors(
        verdicts: list[FeatureVerdict],
        shap_values: dict[str, float] | None,
        n: int = 15,
    ) -> list[FeatureVerdict]:
        non_ok = [v for v in verdicts if v.status != "ok"]
        if shap_values:
            non_ok.sort(
                key=lambda v: (v.severity, abs(shap_values.get(v.feature_name, 0))),
                reverse=True,
            )
        else:
            non_ok.sort(key=lambda v: v.severity, reverse=True)
        return non_ok[:n]

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any, *, enable_virustotal: bool = False) -> "ReportBuilder":
        """
        Create a ReportBuilder from a Config object.
        Loads the model, preprocessor, pipeline, and verdict_engine automatically.

        Args:
            config: instance of src.core.config.Config
        """
        import yaml

        from src.core.extraction_support import VirusTotalLookupService
        from src.core.indicator_registry import IndicatorRegistry
        from src.extractors.feature_pipeline import FeaturePipeline
        from src.reporting.verdict_engine import VerdictEngine

        registry = IndicatorRegistry(config.indicators_dir)
        project_root = Path(config.indicators_dir).resolve().parents[1]

        model = _load_joblib_artifact(Path(config.model_path))

        with open(config.feature_columns_path, encoding="utf-8") as f:
            feature_meta = json.load(f)
        feature_columns = feature_meta["feature_columns"]
        preprocessor_path = Path(config.preprocessor_path)
        if not preprocessor_path.exists():
            preprocessor = _rebuild_runtime_preprocessor(config, feature_meta)
        else:
            try:
                preprocessor = _load_joblib_artifact(preprocessor_path)
            except Exception:
                preprocessor = _rebuild_runtime_preprocessor(config, feature_meta)

        vt_features_included = bool(
            feature_meta.get(
                "vt_features_included",
                any(str(column).startswith("vt_") for column in feature_columns),
            )
        )

        if vt_features_included and enable_virustotal:
            vt_lookup = VirusTotalLookupService(project_root=project_root)
            pipeline = FeaturePipeline.default(
                registry,
                enable_virustotal=True,
                virustotal_lookup=vt_lookup,
            )
        else:
            pipeline = FeaturePipeline.default(registry, enable_virustotal=False)

        with open(config.verdict_rules_path, encoding="utf-8") as f:
            rules_data = yaml.safe_load(f)
        verdict_engine = VerdictEngine(rules_data.get("feature_rules", {}))
        verdict_threshold, suspicious_threshold = _resolve_runtime_thresholds(config, feature_meta)

        return cls(
            pipeline=pipeline,
            model=model,
            preprocessor=preprocessor,
            verdict_engine=verdict_engine,
            feature_columns=feature_columns,
            verdict_threshold=verdict_threshold,
            suspicious_threshold=suspicious_threshold,
            model_name=feature_meta.get("model_name", "unknown"),
            model_version=feature_meta.get("version", "unknown"),
        )
