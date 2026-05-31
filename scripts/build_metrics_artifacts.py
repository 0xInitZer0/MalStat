from __future__ import annotations

import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.loader import load_training_dataset
from src.training.pipeline import (
    TabularFeaturePreprocessor,
    _build_training_sample_weights,
    _candidate_estimators,
    _infer_feature_types,
)
from src.core.config import Config


OUTPUT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
EXTERNAL_REPORT_ROOTS = [
    PROJECT_ROOT / "reports",
    PROJECT_ROOT.parent / "MalStat" / "project" / "reports",
]
GRAY_ZONE_RE = re.compile(r"(crackme|keygen)", re.IGNORECASE)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_external_dir(name: str) -> Path | None:
    for root in EXTERNAL_REPORT_ROOTS:
        candidate = root / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _compat_predict_proba(calibrator: Any, X: Any) -> np.ndarray:
    if not hasattr(calibrator, "multi_class"):
        calibrator.multi_class = "auto"
    return np.asarray(calibrator.predict_proba(X), dtype=float)


def _probabilities(estimator: Any, X: Any) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        probabilities = estimator.predict_proba(X)
    return np.asarray(probabilities, dtype=float)[:, 1]


def _calibrated_probabilities(calibrated_model: Any, X: Any) -> np.ndarray:
    return _probabilities(calibrated_model, X)


def _metrics_from_probabilities(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    y_array = np.asarray(y_true, dtype=int)
    p_array = np.asarray(probabilities, dtype=float)
    predictions = (p_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_array, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / max(tn + fp, 1))
    return {
        "threshold": float(threshold),
        "sample_count": int(len(y_array)),
        "positive_count": int(np.sum(y_array == 1)),
        "negative_count": int(np.sum(y_array == 0)),
        "accuracy": float(accuracy_score(y_array, predictions)),
        "precision": float(precision_score(y_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_array, predictions, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_array, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_array, p_array)) if len(np.unique(y_array)) > 1 else math.nan,
        "pr_auc": float(average_precision_score(y_array, p_array)) if len(np.unique(y_array)) > 1 else math.nan,
        "brier_score": float(brier_score_loss(y_array, p_array)),
        "log_loss": float(log_loss(y_array, p_array, labels=[0, 1])),
        "mae_probability": float(mean_absolute_error(y_array, p_array)),
        "rmse_probability": float(math.sqrt(mean_squared_error(y_array, p_array))),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _load_runtime_artifacts() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, Any, dict[str, Any]]:
    experiment_log = _load_json(MODELS_DIR / "experiment_log.json")
    evaluation_report = _load_json(MODELS_DIR / "evaluation_report.json")
    feature_meta = _load_json(MODELS_DIR / "feature_columns.json")
    val_predictions = pd.read_csv(MODELS_DIR / "val_predictions.csv")
    test_predictions = pd.read_csv(MODELS_DIR / "test_predictions.csv")
    calibrated_model = joblib.load(MODELS_DIR / "calibrated_model.pkl")
    return experiment_log, evaluation_report, val_predictions, test_predictions, calibrated_model, feature_meta


def _build_exact_split(feature_columns: list[str], val_predictions: pd.DataFrame, test_predictions: pd.DataFrame, expected_counts: dict[str, int]) -> dict[str, pd.DataFrame]:
    dataset = load_training_dataset(PROJECT_ROOT / "data" / "clear_training")
    joined_df = dataset.joined_df.copy().reset_index(drop=True)
    joined_df["sample_id"] = joined_df["sample_id"].astype(str)

    val_ids = set(val_predictions["sample_id"].astype(str))
    test_ids = set(test_predictions["sample_id"].astype(str))
    train_mask = ~joined_df["sample_id"].isin(val_ids | test_ids)
    val_mask = joined_df["sample_id"].isin(val_ids)
    test_mask = joined_df["sample_id"].isin(test_ids)

    split = {
        "train": joined_df.loc[train_mask].reset_index(drop=True),
        "val": joined_df.loc[val_mask].reset_index(drop=True),
        "test": joined_df.loc[test_mask].reset_index(drop=True),
    }
    actual_counts = {name: len(rows) for name, rows in split.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"Exact split reconstruction failed: expected {expected_counts}, got {actual_counts}")

    missing_columns = [column for column in feature_columns if column not in joined_df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing expected feature columns: {missing_columns[:10]}")

    return split


def _fit_candidate_models(split: dict[str, pd.DataFrame], feature_columns: list[str]) -> tuple[TabularFeaturePreprocessor, dict[str, Any], dict[str, dict[str, dict[str, Any]]], dict[str, np.ndarray]]:
    config = Config()
    numeric_columns, categorical_columns = _infer_feature_types(split["train"], feature_columns)
    preprocessor = TabularFeaturePreprocessor(
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    X_train = split["train"][feature_columns]
    X_val = split["val"][feature_columns]
    X_test = split["test"][feature_columns]
    y_train = split["train"]["label"].astype(int)
    y_val = split["val"]["label"].astype(int)
    y_test = split["test"]["label"].astype(int)

    preprocessor.fit(X_train, y_train)
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    sample_weights, _ = _build_training_sample_weights(split["train"], config)
    fit_kwargs = {"sample_weight": sample_weights} if sample_weights is not None else {}

    candidates = _candidate_estimators(config)
    fitted: dict[str, Any] = {}
    metrics_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    probabilities_by_model: dict[str, dict[str, np.ndarray]] = {}
    for model_name, estimator in candidates.items():
        estimator.fit(X_train_t, y_train, **fit_kwargs)
        fitted[model_name] = estimator
        train_prob = _probabilities(estimator, X_train_t)
        val_prob = _probabilities(estimator, X_val_t)
        test_prob = _probabilities(estimator, X_test_t)
        probabilities_by_model[model_name] = {
            "train": train_prob,
            "val": val_prob,
            "test": test_prob,
        }
        metrics_by_model[model_name] = {
            "train": _metrics_from_probabilities(y_train, train_prob, threshold=0.5),
            "val": _metrics_from_probabilities(y_val, val_prob, threshold=0.5),
            "test": _metrics_from_probabilities(y_test, test_prob, threshold=0.5),
        }

    return preprocessor, fitted, metrics_by_model, probabilities_by_model


def _build_selected_model_metrics(calibrated_model: Any, split: dict[str, pd.DataFrame], feature_columns: list[str], selected_threshold: float) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], Any]:
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    X_train = preprocessor.transform(split["train"][feature_columns])
    X_val = preprocessor.transform(split["val"][feature_columns])
    X_test = preprocessor.transform(split["test"][feature_columns])

    probabilities = {
        "train": _calibrated_probabilities(calibrated_model, X_train),
        "val": _calibrated_probabilities(calibrated_model, X_val),
        "test": _calibrated_probabilities(calibrated_model, X_test),
    }
    metrics = {
        split_name: _metrics_from_probabilities(rows["label"].astype(int), probabilities[split_name], threshold=selected_threshold)
        for split_name, rows in split.items()
    }
    transformed = {"train": X_train, "val": X_val, "test": X_test}
    return metrics, probabilities, transformed


def _compute_learning_curves(calibrated_model: Any, transformed_split: dict[str, Any], split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base_estimator = calibrated_model.base_estimator
    booster = base_estimator.booster_
    calibrator = calibrated_model.calibrator
    best_iteration = int(getattr(base_estimator, "best_iteration_", 0) or getattr(base_estimator, "n_estimators", 0) or 0)
    if best_iteration <= 0:
        best_iteration = int(getattr(base_estimator, "n_estimators", 300) or 300)

    rows: list[dict[str, Any]] = []
    for iteration in range(1, best_iteration + 1):
        row: dict[str, Any] = {"iteration": iteration}
        for split_name in ("train", "val", "test"):
            raw_scores = booster.predict(transformed_split[split_name], num_iteration=iteration, raw_score=True)
            probabilities = _compat_predict_proba(calibrator, np.asarray(raw_scores, dtype=float).reshape(-1, 1))[:, 1]
            targets = split[split_name]["label"].astype(int)
            row[f"{split_name}_log_loss"] = float(log_loss(targets, probabilities, labels=[0, 1]))
            row[f"{split_name}_brier"] = float(brier_score_loss(targets, probabilities))
        rows.append(row)
    return pd.DataFrame(rows)


def _load_external_reports() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for label_name, directory_name, true_label in (
        ("legit", "LEGIT", 0),
        ("malware", "MALWR", 1),
    ):
        directory = _resolve_external_dir(directory_name)
        if directory is None:
            continue
        for path in sorted(directory.glob("*.analysis.json")):
            payload = _load_json(path)
            filename = str(payload.get("filename") or path.name)
            group = "gray_zone" if GRAY_ZONE_RE.search(filename) else label_name
            verdict = str(payload.get("verdict") or "")
            predicted_label = int(verdict in {"malicious", "suspicious"})
            rows.append(
                {
                    "dataset": group,
                    "source_dir": label_name,
                    "filename": filename,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "verdict": verdict,
                    "probability": float(payload.get("probability", 0.0) or 0.0),
                    "threshold": float(payload.get("verdict_threshold", 0.5) or 0.5),
                    "path": str(path),
                }
            )

    external = pd.DataFrame(rows)
    if external.empty:
        return external, external
    strict = external[external["dataset"].isin(["legit", "malware"])].reset_index(drop=True)
    gray = external[external["dataset"] == "gray_zone"].reset_index(drop=True)
    return strict, gray


def _save_table(data: pd.DataFrame, filename: str) -> None:
    data.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8")


def _plot_metric_bars(candidate_metrics: pd.DataFrame) -> None:
    metrics_to_plot = ["roc_auc", "pr_auc", "f1", "brier_score", "log_loss"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 4.5), constrained_layout=True)
    split_filtered = candidate_metrics[candidate_metrics["split"].isin(["val", "test"])].copy()
    for ax, metric in zip(axes, metrics_to_plot):
        pivot = split_filtered.pivot(index="model", columns="split", values=metric)
        pivot = pivot.loc[[name for name in ["logistic_regression", "lightgbm", "lightgbm_calibrated"] if name in pivot.index]]
        pivot.plot(kind="bar", ax=ax, rot=20)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Candidate metrics: validation and test")
    fig.savefig(OUTPUT_DIR / "candidate_metrics_comparison.png", dpi=200)
    plt.close(fig)


def _plot_learning_curves(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for split_name, color in (("train", "tab:blue"), ("val", "tab:orange"), ("test", "tab:green")):
        axes[0].plot(curves["iteration"], curves[f"{split_name}_log_loss"], label=split_name, color=color)
        axes[1].plot(curves["iteration"], curves[f"{split_name}_brier"], label=split_name, color=color)
    axes[0].set_title("Calibrated LightGBM log-loss by iteration")
    axes[0].set_xlabel("Boosting iteration")
    axes[0].set_ylabel("Log-loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_title("Calibrated LightGBM Brier score by iteration")
    axes[1].set_xlabel("Boosting iteration")
    axes[1].set_ylabel("Brier score")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.savefig(OUTPUT_DIR / "lightgbm_learning_curves.png", dpi=200)
    plt.close(fig)


def _plot_internal_curves(split: dict[str, pd.DataFrame], probabilities: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for split_name, color in (("val", "tab:orange"), ("test", "tab:green")):
        y_true = split[split_name]["label"].astype(int)
        fpr, tpr, _ = roc_curve(y_true, probabilities[split_name])
        precision, recall, _ = precision_recall_curve(y_true, probabilities[split_name])
        axes[0].plot(fpr, tpr, label=f"{split_name} AUC={roc_auc_score(y_true, probabilities[split_name]):.4f}", color=color)
        axes[1].plot(recall, precision, label=f"{split_name} AP={average_precision_score(y_true, probabilities[split_name]):.4f}", color=color)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    axes[0].set_title("ROC curves")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_title("Precision-recall curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.savefig(OUTPUT_DIR / "internal_roc_pr_curves.png", dpi=200)
    plt.close(fig)


def _plot_calibration_curve(split: dict[str, pd.DataFrame], probabilities: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for split_name, color in (("val", "tab:orange"), ("test", "tab:green")):
        y_true = split[split_name]["label"].astype(int)
        frac_pos, mean_pred = calibration_curve(y_true, probabilities[split_name], n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=split_name, color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax.set_title("Calibration curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=200)
    plt.close(fig)


def _plot_confusion_matrix(metrics: dict[str, Any], filename: str, title: str) -> None:
    matrix = np.array([
        [metrics["true_negative"], metrics["false_positive"]],
        [metrics["false_negative"], metrics["true_positive"]],
    ])
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=12)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title(title)
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def _plot_probability_distributions(split: dict[str, pd.DataFrame], probabilities: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, split_name in zip(axes, ["val", "test"]):
        frame = pd.DataFrame({
            "label": split[split_name]["label"].astype(int),
            "probability": probabilities[split_name],
        })
        for label, color in ((0, "tab:blue"), (1, "tab:red")):
            subset = frame[frame["label"] == label]
            ax.hist(subset["probability"], bins=20, alpha=0.6, label=f"label={label}", color=color)
        ax.set_title(f"Probability distribution: {split_name}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(OUTPUT_DIR / "probability_distributions.png", dpi=200)
    plt.close(fig)


def _plot_external_results(external: pd.DataFrame) -> None:
    if external.empty:
        return
    ordered = external.sort_values(["true_label", "probability", "filename"], ascending=[True, False, True]).reset_index(drop=True)
    colors = ordered["true_label"].map({0: "tab:blue", 1: "tab:red"}).tolist()
    markers = ["x" if row.true_label != row.predicted_label else "o" for row in ordered.itertuples()]

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for idx, row in enumerate(ordered.itertuples()):
        ax.scatter(idx, row.probability, color=colors[idx], marker=markers[idx], s=70)
    if not ordered.empty:
        ax.axhline(float(ordered.iloc[0]["threshold"]), linestyle="--", color="gray", alpha=0.8, label="threshold")
    ax.set_title("External spot-check probabilities")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Predicted probability")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(OUTPUT_DIR / "external_spotcheck_probabilities.png", dpi=200)
    plt.close(fig)


def _build_prediction_examples(split: dict[str, pd.DataFrame], probabilities: dict[str, np.ndarray], external: pd.DataFrame) -> pd.DataFrame:
    test_rows = split["test"].copy()
    test_rows["probability"] = probabilities["test"]
    test_rows["predicted_label"] = (test_rows["probability"] >= 0.454123).astype(int)
    test_rows["source_block"] = "internal_test"

    examples = []
    for label, name in ((1, "top_true_malicious"), (0, "top_true_benign")):
        subset = test_rows[test_rows["label"].astype(int) == label].sort_values("probability", ascending=(label == 0)).head(5)
        subset = subset[["sample_id", "original_name", "label", "probability", "predicted_label", "source", "source_family"]].copy()
        subset["example_group"] = name
        examples.append(subset)

    mistakes = test_rows[test_rows["label"].astype(int) != test_rows["predicted_label"].astype(int)]
    if not mistakes.empty:
        mistake_subset = mistakes[["sample_id", "original_name", "label", "probability", "predicted_label", "source", "source_family"]].copy()
        mistake_subset["example_group"] = "internal_errors"
        examples.append(mistake_subset)

    if not external.empty:
        ext = external.copy()
        ext = ext[["filename", "true_label", "probability", "predicted_label", "dataset"]].rename(columns={
            "filename": "original_name",
            "true_label": "label",
        })
        ext["sample_id"] = "external"
        ext["source"] = ext["dataset"]
        ext["source_family"] = ext["dataset"]
        ext["example_group"] = np.where(ext["label"] != ext["predicted_label"], "external_errors", "external_reference")
        examples.append(ext.head(12))

    if not examples:
        return pd.DataFrame()
    return pd.concat(examples, ignore_index=True)


def _write_summary_json(payload: dict[str, Any]) -> None:
    (OUTPUT_DIR / "metrics_overview.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_log, evaluation_report, val_predictions, test_predictions, calibrated_model, feature_meta = _load_runtime_artifacts()

    feature_columns = [str(value) for value in feature_meta["feature_columns"]]
    expected_counts = {
        "train": int(experiment_log["split_summary"]["train_rows"]),
        "val": int(experiment_log["split_summary"]["val_rows"]),
        "test": int(experiment_log["split_summary"]["test_rows"]),
    }
    split = _build_exact_split(feature_columns, val_predictions, test_predictions, expected_counts)

    _, fitted_candidates, candidate_metrics, _ = _fit_candidate_models(split, feature_columns)
    selected_threshold = float(evaluation_report["operating_point"]["selected_threshold"])
    selected_metrics, selected_probabilities, transformed_split = _build_selected_model_metrics(
        calibrated_model,
        split,
        feature_columns,
        selected_threshold,
    )
    learning_curves = _compute_learning_curves(calibrated_model, transformed_split, split)
    external_strict, external_gray = _load_external_reports()

    candidate_rows: list[dict[str, Any]] = []
    for model_name, split_metrics in candidate_metrics.items():
        for split_name, metrics in split_metrics.items():
            row = {"model": model_name, "split": split_name}
            row.update(metrics)
            candidate_rows.append(row)
    for split_name, metrics in selected_metrics.items():
        row = {"model": "lightgbm_calibrated", "split": split_name}
        row.update(metrics)
        candidate_rows.append(row)
    candidate_metrics_df = pd.DataFrame(candidate_rows)

    regression_metrics_df = candidate_metrics_df[[
        "model",
        "split",
        "brier_score",
        "log_loss",
        "mae_probability",
        "rmse_probability",
    ]].copy()
    ranking_metrics_df = candidate_metrics_df[[
        "model",
        "split",
        "roc_auc",
        "pr_auc",
    ]].copy()
    point_metrics_df = candidate_metrics_df[[
        "model",
        "split",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]].copy()

    if not external_strict.empty:
        external_metrics = _metrics_from_probabilities(external_strict["true_label"], external_strict["probability"], threshold=float(external_strict.iloc[0]["threshold"]))
        external_metrics_df = pd.DataFrame([{**{"dataset": "external_strict"}, **external_metrics}])
    else:
        external_metrics_df = pd.DataFrame()

    if not external_gray.empty:
        gray_metrics = _metrics_from_probabilities(external_gray["true_label"], external_gray["probability"], threshold=float(external_gray.iloc[0]["threshold"]))
        gray_metrics_df = pd.DataFrame([{**{"dataset": "external_gray_zone"}, **gray_metrics}])
    else:
        gray_metrics_df = pd.DataFrame()

    examples_df = _build_prediction_examples(split, selected_probabilities, external_strict)

    _save_table(candidate_metrics_df, "candidate_metrics_all.csv")
    _save_table(point_metrics_df, "point_metrics.csv")
    _save_table(ranking_metrics_df, "ranking_metrics.csv")
    _save_table(regression_metrics_df, "regression_metrics.csv")
    _save_table(learning_curves, "lightgbm_learning_curves.csv")
    if not external_metrics_df.empty:
        _save_table(external_metrics_df, "external_strict_metrics.csv")
    if not gray_metrics_df.empty:
        _save_table(gray_metrics_df, "gray_zone_metrics.csv")
    if not external_strict.empty:
        _save_table(external_strict, "external_spotcheck_predictions.csv")
    if not external_gray.empty:
        _save_table(external_gray, "gray_zone_predictions.csv")
    if not examples_df.empty:
        _save_table(examples_df, "prediction_examples.csv")

    _plot_metric_bars(candidate_metrics_df)
    _plot_learning_curves(learning_curves)
    _plot_internal_curves(split, selected_probabilities)
    _plot_calibration_curve(split, selected_probabilities)
    _plot_confusion_matrix(selected_metrics["test"], "test_confusion_matrix.png", "Calibrated LightGBM: test confusion matrix")
    _plot_probability_distributions(split, selected_probabilities)
    if not external_strict.empty:
        _plot_external_results(external_strict)
        _plot_confusion_matrix(
            _metrics_from_probabilities(external_strict["true_label"], external_strict["probability"], threshold=float(external_strict.iloc[0]["threshold"])),
            "external_confusion_matrix.png",
            "External strict spot-check confusion matrix",
        )

    summary_payload = {
        "selected_model": experiment_log["selected_model"],
        "selected_threshold": selected_threshold,
        "split_counts": expected_counts,
        "candidate_metrics_rows": len(candidate_metrics_df),
        "external_strict_count": int(len(external_strict)),
        "external_gray_zone_count": int(len(external_gray)),
        "generated_files": sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file()),
    }
    _write_summary_json(summary_payload)

    print(f"Generated metrics artifacts in: {OUTPUT_DIR}")
    print(f"Reconstructed split sizes: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    if not external_strict.empty:
        print(f"External strict spot-check rows: {len(external_strict)}")
    if not external_gray.empty:
        print(f"Gray-zone rows: {len(external_gray)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())