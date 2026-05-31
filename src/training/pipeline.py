"""Minimal reproducible training pipeline for PE-only malware classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import __version__ as sklearn_version

from src.core.config import Config
from src.core.path_utils import serialize_relative_path, serialize_relative_paths
from src.training.loader import LoadedTrainingDataset, build_group_labels, load_training_dataset


KNOWN_CATEGORICAL_FEATURES = frozenset({"ep_section_name"})


@dataclass(frozen=True)
class TrainingArtifactPaths:
    model_path: Path
    preprocessor_path: Path
    feature_columns_path: Path
    experiment_log_path: Path
    val_predictions_path: Path
    test_predictions_path: Path
    evaluation_report_path: Path
    evaluation_markdown_path: Path
    profile_root: Path
    profile_model_path: Path
    profile_preprocessor_path: Path
    profile_feature_columns_path: Path
    profile_experiment_log_path: Path
    profile_val_predictions_path: Path
    profile_test_predictions_path: Path
    profile_evaluation_report_path: Path
    profile_evaluation_markdown_path: Path


@dataclass(frozen=True)
class TrainingSplit:
    train_rows: pd.DataFrame
    val_rows: pd.DataFrame
    test_rows: pd.DataFrame


@dataclass(frozen=True)
class TrainingRunResult:
    profile_slug: str
    selected_model_name: str
    artifact_paths: TrainingArtifactPaths
    feature_columns: list[str]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    experiment_log: dict[str, Any]
    operating_point: dict[str, Any] | None = None
    wrote_canonical_artifacts: bool = True


class TabularFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible preprocessor for mixed PE feature tables."""

    def __init__(
        self,
        feature_columns: list[str],
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        self._transformer: ColumnTransformer | None = None
        self.transformed_feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TabularFeaturePreprocessor":
        prepared = self._prepare_frame(X)
        self._transformer = self._build_transformer()
        self._transformer.fit(prepared, y)
        if hasattr(self._transformer, "get_feature_names_out"):
            self.transformed_feature_names_ = [
                str(value) for value in self._transformer.get_feature_names_out()
            ]
        return self

    def transform(self, X: pd.DataFrame) -> Any:
        if self._transformer is None:
            raise ValueError("Preprocessor is not fitted.")
        self._repair_loaded_transformer_state()
        prepared = self._prepare_frame(X)
        return self._transformer.transform(prepared)

    def get_feature_names_out(self) -> list[str]:
        return list(self.transformed_feature_names_)

    def _build_transformer(self) -> ColumnTransformer:
        transformers: list[tuple[str, Any, list[str]]] = []
        if self.numeric_columns:
            transformers.append(
                (
                    "numeric",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]),
                    self.numeric_columns,
                )
            )
        if self.categorical_columns:
            transformers.append(
                (
                    "categorical",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]),
                    self.categorical_columns,
                )
            )
        if not transformers:
            raise ValueError("No feature columns available for preprocessing.")
        return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)

    def _prepare_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        for column in self.feature_columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame[self.feature_columns].copy()

        for column in self.numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in self.categorical_columns:
            series = frame[column].copy()
            series = series.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            frame[column] = series.astype(object)
        return frame

    def _repair_loaded_transformer_state(self) -> None:
        """Repair sklearn cross-version dtype drift in unpickled transformers."""
        transformer = self._transformer
        if transformer is None:
            return

        named_transformers = getattr(transformer, "named_transformers_", {})
        numeric_pipeline = named_transformers.get("numeric")
        if numeric_pipeline in {None, "drop"}:
            return

        named_steps = getattr(numeric_pipeline, "named_steps", {})
        imputer = named_steps.get("imputer")
        if not isinstance(imputer, SimpleImputer):
            return

        statistics = getattr(imputer, "statistics_", None)
        if statistics is None or getattr(statistics, "dtype", None) != object:
            return

        try:
            imputer.statistics_ = np.asarray(statistics, dtype=float)
        except (TypeError, ValueError):
            return


class CalibratedProbabilityModel(BaseEstimator):
    """Holdout-calibrated binary classifier wrapper compatible with ReportBuilder."""

    def __init__(self, base_estimator: Any, calibrator: Any) -> None:
        self.base_estimator = base_estimator
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: Any) -> np.ndarray:
        calibration_inputs = _calibration_inputs(self.base_estimator, X)
        # Compatibility shim: sklearn 1.6+ removed the `multi_class` attribute from
        # LogisticRegression. Pickled calibrators trained on older sklearn versions
        # lack this attribute and raise AttributeError on predict_proba.
        if not hasattr(self.calibrator, "multi_class"):
            self.calibrator.multi_class = "auto"
        calibrated = np.asarray(self.calibrator.predict_proba(calibration_inputs), dtype=float)
        positive_probabilities = calibrated[:, 1]
        return np.column_stack([1.0 - positive_probabilities, positive_probabilities])


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _infer_feature_types(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in feature_columns:
        if column in KNOWN_CATEGORICAL_FEATURES:
            categorical_columns.append(column)
            continue
        series = frame[column].astype(str).replace("", pd.NA)
        non_missing = series.dropna()
        if non_missing.empty:
            numeric_columns.append(column)
            continue
        converted = pd.to_numeric(non_missing, errors="coerce")
        if converted.notna().all():
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns


def _resolve_artifact_paths(project_root: str | Path, config: Config, profile_slug: str) -> TrainingArtifactPaths:
    root = Path(project_root).resolve()
    profile_root = root / "models" / "profiles" / profile_slug
    return TrainingArtifactPaths(
        model_path=Path(config.model_path),
        preprocessor_path=Path(config.preprocessor_path),
        feature_columns_path=Path(config.feature_columns_path),
        experiment_log_path=Path(config.experiment_log_path),
        val_predictions_path=root / "models" / "val_predictions.csv",
        test_predictions_path=root / "models" / "test_predictions.csv",
        evaluation_report_path=root / "models" / "evaluation_report.json",
        evaluation_markdown_path=root / "models" / "evaluation_report.md",
        profile_root=profile_root,
        profile_model_path=profile_root / "calibrated_model.pkl",
        profile_preprocessor_path=profile_root / "preprocessor.pkl",
        profile_feature_columns_path=profile_root / "feature_columns.json",
        profile_experiment_log_path=profile_root / "experiment_log.json",
        profile_val_predictions_path=profile_root / "val_predictions.csv",
        profile_test_predictions_path=profile_root / "test_predictions.csv",
        profile_evaluation_report_path=profile_root / "evaluation_report.json",
        profile_evaluation_markdown_path=profile_root / "evaluation_report.md",
    )


def _serialize_experiment_log(experiment_log: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    payload = dict(experiment_log)
    dataset_root = payload.get("dataset_root")
    if dataset_root not in {None, ""}:
        payload["dataset_root"] = serialize_relative_path(dataset_root, base_dir=base_dir)

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        payload["artifacts"] = serialize_relative_paths(artifacts, base_dir=base_dir)
    return payload


def _split_has_both_classes(rows: pd.DataFrame) -> bool:
    labels = rows["label"].astype(str)
    return {"0", "1"}.issubset(set(labels))


def _build_group_split(dataset: LoadedTrainingDataset, config: Config, max_attempts: int = 64) -> TrainingSplit:
    train_ratio = config.train_ratio
    val_ratio = config.val_ratio
    test_ratio = config.test_ratio
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Train/val/test ratios must sum to 1.0, got {ratio_sum:.6f}")

    joined_df = dataset.joined_df.reset_index(drop=True)
    groups = dataset.groups.reset_index(drop=True)
    target = dataset.target.reset_index(drop=True)
    val_share_of_temp = val_ratio / (val_ratio + test_ratio)

    for attempt in range(max_attempts):
        seed = config.random_state + attempt
        first_splitter = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
        train_indices, temp_indices = next(first_splitter.split(joined_df, target, groups))

        temp_rows = joined_df.iloc[temp_indices].reset_index(drop=True)
        temp_groups = groups.iloc[temp_indices].reset_index(drop=True)
        temp_target = target.iloc[temp_indices].reset_index(drop=True)

        second_splitter = GroupShuffleSplit(n_splits=1, train_size=val_share_of_temp, random_state=seed)
        val_rel_indices, test_rel_indices = next(second_splitter.split(temp_rows, temp_target, temp_groups))

        train_rows = joined_df.iloc[train_indices].reset_index(drop=True)
        val_rows = temp_rows.iloc[val_rel_indices].reset_index(drop=True)
        test_rows = temp_rows.iloc[test_rel_indices].reset_index(drop=True)

        if all(_split_has_both_classes(rows) for rows in [train_rows, val_rows, test_rows]):
            return TrainingSplit(train_rows=train_rows, val_rows=val_rows, test_rows=test_rows)

    raise ValueError(
        "Failed to produce a group-aware train/val/test split with both classes present in every split."
    )


def _probabilities(estimator: Any, X: Any) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        probabilities = estimator.predict_proba(X)
    if hasattr(probabilities, "tolist"):
        probabilities = probabilities.tolist()
    return np.asarray(probabilities, dtype=float)[:, 1]


def _logit_probabilities(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _calibration_inputs(estimator: Any, X: Any) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")

        decision_function = getattr(estimator, "decision_function", None)
        if callable(decision_function):
            try:
                scores = decision_function(X)
                return np.asarray(scores, dtype=float).reshape(-1, 1)
            except Exception:
                pass

        predict = getattr(estimator, "predict", None)
        if callable(predict):
            try:
                raw_scores = predict(X, raw_score=True)
                return np.asarray(raw_scores, dtype=float).reshape(-1, 1)
            except Exception:
                pass

    return _logit_probabilities(_probabilities(estimator, X)).reshape(-1, 1)


def _compute_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    predictions = (probabilities >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "sample_count": int(len(y_true_array)),
        "positive_count": int(np.sum(y_true_array == 1)),
        "negative_count": int(np.sum(y_true_array == 0)),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true_array, predictions, zero_division=0)),
        "precision": float(precision_score(y_true_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_true_array, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true_array, probabilities)),
        "confusion_matrix": confusion_matrix(y_true_array, predictions, labels=[0, 1]).astype(int).tolist(),
    }

    if len(set(y_true_array.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, probabilities))
        metrics["pr_auc"] = float(average_precision_score(y_true_array, probabilities))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def _compute_breakdowns(rows: pd.DataFrame, probabilities: np.ndarray, fields: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    y_true = rows["label"].astype(int).reset_index(drop=True)
    breakdowns: dict[str, list[dict[str, Any]]] = {}

    for field in fields:
        if field not in rows.columns:
            continue
        entries: list[dict[str, Any]] = []
        grouped = rows.reset_index(drop=True).groupby(rows[field].astype(str), dropna=False)
        for group_name, group_rows in grouped:
            group_indices = group_rows.index.to_numpy()
            group_y = y_true.iloc[group_indices]
            group_probabilities = probabilities[group_indices]
            metrics = _compute_metrics(group_y, group_probabilities)
            entries.append({
                "group": str(group_name),
                **metrics,
            })
        entries.sort(key=lambda item: (-item["sample_count"], item["group"]))
        breakdowns[field] = entries
    return breakdowns


def _build_prediction_table(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    split_name: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    row_count = len(rows)
    table = pd.DataFrame({
        "split": [split_name] * row_count,
        "sample_id": rows["sample_id"].reset_index(drop=True) if "sample_id" in rows.columns else [None] * row_count,
        "sha256": rows["sha256"].reset_index(drop=True) if "sha256" in rows.columns else [None] * row_count,
        "original_name": rows["original_name"].reset_index(drop=True) if "original_name" in rows.columns else [None] * row_count,
        "file_type": rows["file_type"].reset_index(drop=True) if "file_type" in rows.columns else [None] * row_count,
        "pe_kind": rows["pe_kind"].reset_index(drop=True) if "pe_kind" in rows.columns else [None] * row_count,
        "source": rows["source"].reset_index(drop=True) if "source" in rows.columns else [None] * row_count,
        "source_family": rows["source_family"].reset_index(drop=True) if "source_family" in rows.columns else [None] * row_count,
        "label": pd.to_numeric(rows["label"], errors="coerce").fillna(0).astype(int).reset_index(drop=True),
        "probability": np.asarray(probabilities, dtype=float),
    })
    table["threshold"] = float(threshold)
    table["predicted_label"] = (table["probability"] >= threshold).astype(int)
    table["is_correct"] = table["predicted_label"] == table["label"]
    table["prediction_outcome"] = np.select(
        [
            (table["label"] == 1) & (table["predicted_label"] == 1),
            (table["label"] == 0) & (table["predicted_label"] == 1),
            (table["label"] == 1) & (table["predicted_label"] == 0),
            (table["label"] == 0) & (table["predicted_label"] == 0),
        ],
        ["tp", "fp", "fn", "tn"],
        default="unknown",
    )
    table["error_kind"] = np.select(
        [table["prediction_outcome"] == "fp", table["prediction_outcome"] == "fn"],
        ["fp", "fn"],
        default="",
    )
    return table[
        [
            "split",
            "sample_id",
            "sha256",
            "original_name",
            "file_type",
            "pe_kind",
            "source",
            "source_family",
            "label",
            "probability",
            "threshold",
            "predicted_label",
            "is_correct",
            "prediction_outcome",
            "error_kind",
        ]
    ]


def _candidate_estimators(config: Config) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for the minimal training pipeline. Install requirements.txt in the project venv."
        ) from exc

    return {
        "logistic_regression": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=config.random_state,
        ),
        "lightgbm": LGBMClassifier(
            random_state=config.random_state,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            n_jobs=config.n_jobs,
            verbose=-1,
        ),
    }


def _build_training_sample_weights(
    rows: pd.DataFrame,
    config: Config,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    strategy = str(getattr(config, "sample_weight_strategy", "none") or "none").strip().lower()

    if rows.empty:
        return None, {
            "enabled": False,
            "strategy": strategy,
            "row_count": 0,
            "strata": [],
            "min_weight": None,
            "max_weight": None,
            "mean_weight": None,
        }

    label_values = rows["label"].astype(str).fillna("").str.strip()
    if "pe_kind" in rows.columns:
        pe_kind_values = rows["pe_kind"].astype(str).fillna("").str.strip().replace("", "__missing__")
    else:
        pe_kind_values = pd.Series(["__missing__"] * len(rows), index=rows.index, dtype=str)

    stratum_keys = label_values.str.cat(pe_kind_values, sep="::")
    stratum_counts = stratum_keys.value_counts(dropna=False).sort_index()
    stratum_base = len(rows) / max(len(stratum_counts), 1)

    if strategy in {"", "none"}:
        return None, {
            "enabled": False,
            "strategy": "none",
            "row_count": int(len(rows)),
            "strata": [
                {
                    "label": key.split("::", 1)[0],
                    "pe_kind": key.split("::", 1)[1],
                    "row_count": int(count),
                    "sample_weight": 1.0,
                }
                for key, count in stratum_counts.items()
            ],
            "min_weight": 1.0,
            "max_weight": 1.0,
            "mean_weight": 1.0,
        }

    if strategy == "benign_driver_hard_negative":
        negative_mask = label_values.eq("0")
        benign_driver_mask = negative_mask & pe_kind_values.eq("driver")
        raw_weights = pd.Series(np.ones(len(rows), dtype=float), index=rows.index)
        focus_weight = 1.0
        if bool(benign_driver_mask.any()):
            negative_count = int(negative_mask.sum())
            benign_driver_count = int(benign_driver_mask.sum())
            focus_weight = float(np.sqrt(max(negative_count, 1) / benign_driver_count))
            raw_weights.loc[benign_driver_mask] = focus_weight
    elif strategy == "balanced_label_pe_kind":
        raw_weights = stratum_keys.map(lambda key: stratum_base / float(stratum_counts.loc[key])).astype(float)
    elif strategy == "sqrt_balanced_label_pe_kind":
        raw_weights = stratum_keys.map(
            lambda key: float(np.sqrt(stratum_base / float(stratum_counts.loc[key])))
        ).astype(float)
    else:
        raise ValueError(
            "Unsupported sample_weight_strategy "
            f"'{config.sample_weight_strategy}'. Supported values: none, balanced_label_pe_kind, "
            "sqrt_balanced_label_pe_kind, benign_driver_hard_negative."
        )

    mean_weight = float(raw_weights.mean()) if len(raw_weights) else 1.0
    if mean_weight > 0.0:
        normalized_weights = raw_weights / mean_weight
    else:
        normalized_weights = raw_weights

    stratum_weight_lookup = normalized_weights.groupby(stratum_keys).first().to_dict()
    summary = {
        "enabled": True,
        "strategy": strategy,
        "row_count": int(len(rows)),
        "strata": [
            {
                "label": key.split("::", 1)[0],
                "pe_kind": key.split("::", 1)[1],
                "row_count": int(count),
                "sample_weight": float(stratum_weight_lookup[key]),
            }
            for key, count in stratum_counts.items()
        ],
        "min_weight": float(normalized_weights.min()),
        "max_weight": float(normalized_weights.max()),
        "mean_weight": float(normalized_weights.mean()),
    }
    if strategy == "benign_driver_hard_negative":
        summary["focus_group"] = {"label": "0", "pe_kind": "driver"}
        summary["focus_weight_before_normalization"] = float(focus_weight)
    return normalized_weights.to_numpy(dtype=float), summary


def _fit_holdout_calibrator(estimator: Any, X_val: Any, y_val: pd.Series, random_state: int) -> CalibratedProbabilityModel:
    base_probabilities = _calibration_inputs(estimator, X_val)
    calibrator = LogisticRegression(max_iter=1000, random_state=random_state)
    calibrator.fit(base_probabilities, y_val)
    return CalibratedProbabilityModel(estimator, calibrator)


def _select_best_model(validation_results: dict[str, dict[str, Any]]) -> str:
    def score(item: tuple[str, dict[str, Any]]) -> tuple[float, float, str]:
        name, metrics = item
        roc_auc = metrics.get("roc_auc")
        pr_auc = metrics.get("pr_auc")
        return (
            float(roc_auc) if roc_auc is not None else float("-inf"),
            float(pr_auc) if pr_auc is not None else float("-inf"),
            name,
        )

    return max(validation_results.items(), key=score)[0]


def _save_training_artifacts(
    *,
    paths: TrainingArtifactPaths,
    calibrated_model: Any,
    preprocessor: TabularFeaturePreprocessor,
    feature_metadata: dict[str, Any],
    experiment_log: dict[str, Any],
    val_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    write_canonical: bool,
) -> None:
    for path in [
        paths.profile_model_path,
        paths.profile_preprocessor_path,
        paths.profile_feature_columns_path,
        paths.profile_experiment_log_path,
        paths.profile_val_predictions_path,
        paths.profile_test_predictions_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    if write_canonical:
        for path in [
            paths.model_path,
            paths.preprocessor_path,
            paths.feature_columns_path,
            paths.experiment_log_path,
            paths.val_predictions_path,
            paths.test_predictions_path,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)

    model_targets = [paths.profile_model_path]
    preprocessor_targets = [paths.profile_preprocessor_path]
    feature_targets = [paths.profile_feature_columns_path]
    experiment_targets = [paths.profile_experiment_log_path]
    val_prediction_targets = [paths.profile_val_predictions_path]
    test_prediction_targets = [paths.profile_test_predictions_path]
    if write_canonical:
        model_targets.insert(0, paths.model_path)
        preprocessor_targets.insert(0, paths.preprocessor_path)
        feature_targets.insert(0, paths.feature_columns_path)
        experiment_targets.insert(0, paths.experiment_log_path)
        val_prediction_targets.insert(0, paths.val_predictions_path)
        test_prediction_targets.insert(0, paths.test_predictions_path)

    for model_path in model_targets:
        joblib.dump(calibrated_model, model_path)
    for preprocessor_path in preprocessor_targets:
        joblib.dump(preprocessor, preprocessor_path)

    feature_payload = json.dumps(feature_metadata, ensure_ascii=False, indent=2, default=_json_default)
    for feature_path in feature_targets:
        feature_path.write_text(feature_payload, encoding="utf-8")

    for experiment_path in experiment_targets:
        experiment_payload = json.dumps(
            _serialize_experiment_log(experiment_log, base_dir=experiment_path.parent),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        experiment_path.write_text(experiment_payload, encoding="utf-8")
    for prediction_path in val_prediction_targets:
        val_predictions.to_csv(prediction_path, index=False)
    for prediction_path in test_prediction_targets:
        test_predictions.to_csv(prediction_path, index=False)


def _should_write_canonical_artifacts(dataset: LoadedTrainingDataset) -> bool:
    resolution = dataset.resolution
    if resolution.manifest_path is None:
        return True
    default_profile = resolution.recommended_default_profile or "pure_static"
    return resolution.profile_slug == default_profile


def _run_postfit_evaluation(
    *,
    paths: TrainingArtifactPaths,
    write_canonical: bool,
    operating_point_policy: str | Path | dict[str, Any] | None,
    acceptance_criteria: str | Path | dict[str, Any] | None,
) -> dict[str, Any]:
    from src.evaluation import build_evaluation_report

    profile_evaluation = build_evaluation_report(
        paths.profile_experiment_log_path,
        operating_point_policy=operating_point_policy,
        acceptance_criteria=acceptance_criteria,
    )
    canonical_evaluation = None
    if write_canonical:
        canonical_evaluation = build_evaluation_report(
            paths.experiment_log_path,
            operating_point_policy=operating_point_policy,
            acceptance_criteria=acceptance_criteria,
        )
    return {
        "profile": profile_evaluation,
        "canonical": canonical_evaluation,
    }


def run_training_pipeline(
    *,
    project_root: str | Path,
    dataset_root: str | Path,
    profile: str | None = None,
    config: Config | None = None,
) -> TrainingRunResult:
    root = Path(project_root).resolve()
    resolved_config = (config or Config()).resolve(root)
    dataset = load_training_dataset(dataset_root, profile=profile)
    split = _build_group_split(dataset, resolved_config)

    numeric_columns, categorical_columns = _infer_feature_types(dataset.feature_df, dataset.feature_columns)
    preprocessor = TabularFeaturePreprocessor(
        feature_columns=dataset.feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    X_train = split.train_rows[dataset.feature_columns]
    X_val = split.val_rows[dataset.feature_columns]
    X_test = split.test_rows[dataset.feature_columns]
    y_train = split.train_rows["label"].astype(int)
    y_val = split.val_rows["label"].astype(int)
    y_test = split.test_rows["label"].astype(int)
    training_sample_weights, training_sample_weighting = _build_training_sample_weights(
        split.train_rows.reset_index(drop=True),
        resolved_config,
    )

    preprocessor.fit(X_train, y_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    candidate_models = _candidate_estimators(resolved_config)
    fitted_candidates: dict[str, Any] = {}
    validation_metrics: dict[str, dict[str, Any]] = {}

    for model_name, estimator in candidate_models.items():
        fit_kwargs = {}
        if training_sample_weights is not None:
            fit_kwargs["sample_weight"] = training_sample_weights
        estimator.fit(X_train_transformed, y_train, **fit_kwargs)
        fitted_candidates[model_name] = estimator
        validation_probabilities = _probabilities(estimator, X_val_transformed)
        validation_metrics[model_name] = _compute_metrics(y_val, validation_probabilities)

    selected_model_name = _select_best_model(validation_metrics)
    selected_estimator = fitted_candidates[selected_model_name]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated_model = _fit_holdout_calibrator(
            selected_estimator,
            X_val_transformed,
            y_val,
            random_state=resolved_config.random_state,
        )

    validation_probabilities = _probabilities(calibrated_model, X_val_transformed)
    calibrated_validation_metrics = _compute_metrics(y_val, validation_probabilities)
    calibrated_validation_metrics["breakdowns"] = _compute_breakdowns(
        split.val_rows.reset_index(drop=True),
        validation_probabilities,
        fields=("pe_kind", "source", "source_family"),
    )
    val_predictions = _build_prediction_table(
        split.val_rows.reset_index(drop=True),
        validation_probabilities,
        split_name="val",
    )

    test_probabilities = _probabilities(calibrated_model, X_test_transformed)
    test_metrics = _compute_metrics(y_test, test_probabilities)
    test_metrics["breakdowns"] = _compute_breakdowns(
        split.test_rows.reset_index(drop=True),
        test_probabilities,
        fields=("pe_kind", "source", "source_family"),
    )
    test_predictions = _build_prediction_table(
        split.test_rows.reset_index(drop=True),
        test_probabilities,
        split_name="test",
    )

    artifact_paths = _resolve_artifact_paths(root, resolved_config, dataset.resolution.profile_slug)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    feature_metadata = {
        "feature_columns": dataset.feature_columns,
        "model_name": selected_model_name,
        "version": run_timestamp,
        "sklearn_version": sklearn_version,
        "profile_slug": dataset.resolution.profile_slug,
        "calibration_method": "holdout_platt_scaling",
        "calibration_input": "decision_function_or_raw_score_fallback_logit_probability",
        "candidate_models": list(candidate_models.keys()),
        "numeric_feature_columns": numeric_columns,
        "categorical_feature_columns": categorical_columns,
        "transformed_feature_count": len(preprocessor.get_feature_names_out()),
        "vt_features_included": any(column.startswith("vt_") for column in dataset.feature_columns),
    }

    experiment_log = {
        "run_timestamp_utc": run_timestamp,
        "profile_slug": dataset.resolution.profile_slug,
        "dataset_root": Path(dataset_root).resolve(),
        "selected_model": {
            "name": selected_model_name,
            "profile_slug": dataset.resolution.profile_slug,
        },
        "candidate_validation_metrics": validation_metrics,
        "validation_metrics": calibrated_validation_metrics,
        "test_metrics": test_metrics,
        "split_summary": {
            "train_rows": len(split.train_rows),
            "val_rows": len(split.val_rows),
            "test_rows": len(split.test_rows),
            "train_groups": int(build_group_labels(split.train_rows).nunique()),
            "val_groups": int(build_group_labels(split.val_rows).nunique()),
            "test_groups": int(build_group_labels(split.test_rows).nunique()),
        },
        "training_sample_weighting": training_sample_weighting,
        "artifacts": asdict(artifact_paths),
        "feature_metadata": feature_metadata,
    }

    write_canonical = _should_write_canonical_artifacts(dataset)

    _save_training_artifacts(
        paths=artifact_paths,
        calibrated_model=calibrated_model,
        preprocessor=preprocessor,
        feature_metadata=feature_metadata,
        experiment_log=experiment_log,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        write_canonical=write_canonical,
    )

    evaluation_results = _run_postfit_evaluation(
        paths=artifact_paths,
        write_canonical=write_canonical,
        operating_point_policy=resolved_config.operating_point_policy_path,
        acceptance_criteria=resolved_config.acceptance_criteria_path,
    )
    profile_evaluation_report = evaluation_results["profile"].report

    return TrainingRunResult(
        profile_slug=dataset.resolution.profile_slug,
        selected_model_name=selected_model_name,
        artifact_paths=artifact_paths,
        feature_columns=dataset.feature_columns,
        validation_metrics=profile_evaluation_report["splits"]["val"]["overall_metrics"],
        test_metrics=profile_evaluation_report["splits"]["test"]["overall_metrics"],
        experiment_log=experiment_log,
        operating_point=profile_evaluation_report.get("operating_point"),
        wrote_canonical_artifacts=write_canonical,
    )