"""Build detailed evaluation reports from saved and external prediction tables."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatchcase
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.core.path_utils import resolve_stored_path, serialize_relative_path


BREAKDOWN_FIELDS = ("pe_kind", "source", "source_family")
BASELINE_THRESHOLD = 0.5
DEFAULT_EXTERNAL_SPLIT_NAME = "benign_dll_sys_holdout"
DEFAULT_OPERATING_POINT_POLICY: dict[str, Any] = {
    "selection": {
        "metric": "f1",
        "tie_breakers": ["precision", "recall", "proximity_to_0_5"],
        "fallback": "minimum_constraint_violation_then_max_f1",
    },
    "suspicious_band": {
        "mode": "operating_margin",
        "margin": 0.02,
        "minimum_threshold": BASELINE_THRESHOLD,
    },
    "constraints": [
        {
            "name": "benign_dll_fp_guardrail",
            "filters": {"label": 0, "pe_kind": "dll"},
            "max_false_positive_rate": 0.02,
            "max_false_positive_count": 2,
        },
        {
            "name": "benign_driver_fp_guardrail",
            "filters": {"label": 0, "pe_kind": "driver"},
            "max_false_positive_count": 0,
        },
    ],
}


@dataclass(frozen=True)
class EvaluationArtifactPaths:
    report_json_path: Path
    report_markdown_path: Path


@dataclass(frozen=True)
class EvaluationReportResult:
    experiment_log_path: Path
    profile_slug: str
    report: dict[str, Any]
    artifact_paths: EvaluationArtifactPaths


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _load_structured_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.casefold()
    if suffix == ".json":
        return _load_json(path)
    if suffix in {".yaml", ".yml"}:
        import yaml

        with path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Structured config must deserialize to an object: {path}")
        return loaded
    raise ValueError(f"Unsupported structured config format: {path}")


def _normalize_policy(raw_policy: dict[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_OPERATING_POINT_POLICY)
    if not raw_policy:
        return merged

    selection = raw_policy.get("selection", {})
    if isinstance(selection, dict):
        merged["selection"].update(selection)

    suspicious_band = raw_policy.get("suspicious_band", {})
    if isinstance(suspicious_band, dict):
        merged["suspicious_band"].update(suspicious_band)

    constraints = raw_policy.get("constraints")
    if constraints is not None:
        merged["constraints"] = list(constraints)
    return merged


def _load_operating_point_policy(policy: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if policy is None:
        return _normalize_policy(None)
    if isinstance(policy, dict):
        return _normalize_policy(policy)

    resolved_path = Path(policy).resolve()
    if not resolved_path.exists():
        return _normalize_policy(None)
    return _normalize_policy(_load_structured_file(resolved_path))


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def resolve_suspicious_threshold(
    operating_threshold: float,
    policy: dict[str, Any] | str | Path | None = None,
) -> float:
    resolved_policy = _load_operating_point_policy(policy)
    suspicious_band = dict(resolved_policy.get("suspicious_band", {}))
    resolved_operating_threshold = float(operating_threshold)
    minimum_threshold = _coerce_float(
        suspicious_band.get("minimum_threshold"),
        BASELINE_THRESHOLD,
    )
    mode = str(suspicious_band.get("mode", "operating_margin") or "operating_margin").strip().casefold()

    if mode == "fixed":
        candidate_threshold = _coerce_float(
            suspicious_band.get("threshold"),
            resolved_operating_threshold,
        )
    else:
        margin = _coerce_float(suspicious_band.get("margin"), 0.02)
        candidate_threshold = resolved_operating_threshold - max(margin, 0.0)

    clamped_threshold = max(minimum_threshold, candidate_threshold)
    return min(resolved_operating_threshold, clamped_threshold)


def _normalize_acceptance_criteria(raw_criteria: dict[str, Any] | None) -> dict[str, Any]:
    if not raw_criteria:
        return {"criteria": []}
    return {"criteria": list(raw_criteria.get("criteria", []))}


def _load_acceptance_criteria(criteria: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if criteria is None:
        return _normalize_acceptance_criteria(None)
    if isinstance(criteria, dict):
        return _normalize_acceptance_criteria(criteria)

    resolved_path = Path(criteria).resolve()
    if not resolved_path.exists():
        return _normalize_acceptance_criteria(None)
    return _normalize_acceptance_criteria(_load_structured_file(resolved_path))


def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().casefold()


def _resolve_threshold(table: pd.DataFrame, threshold: float | None) -> float:
    if threshold is not None:
        return float(threshold)
    if not table.empty and "threshold" in table.columns:
        return float(table["threshold"].iloc[0])
    return BASELINE_THRESHOLD


def _apply_threshold(table: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    resolved_threshold = _resolve_threshold(table, threshold)
    if table.empty:
        annotated = table.copy()
        annotated["threshold"] = float(resolved_threshold)
        annotated["predicted_label"] = pd.Series(dtype=int)
        annotated["is_correct"] = pd.Series(dtype=bool)
        annotated["prediction_outcome"] = pd.Series(dtype=object)
        annotated["error_kind"] = pd.Series(dtype=object)
        return annotated

    annotated = table.copy().reset_index(drop=True)
    annotated["label"] = pd.to_numeric(annotated["label"], errors="coerce").fillna(0).astype(int)
    annotated["probability"] = pd.to_numeric(annotated["probability"], errors="coerce").fillna(0.0).astype(float)
    annotated["threshold"] = float(resolved_threshold)
    annotated["predicted_label"] = (annotated["probability"] >= resolved_threshold).astype(int)
    annotated["is_correct"] = annotated["predicted_label"] == annotated["label"]
    annotated["prediction_outcome"] = np.select(
        [
            (annotated["label"] == 1) & (annotated["predicted_label"] == 1),
            (annotated["label"] == 0) & (annotated["predicted_label"] == 1),
            (annotated["label"] == 1) & (annotated["predicted_label"] == 0),
            (annotated["label"] == 0) & (annotated["predicted_label"] == 0),
        ],
        ["tp", "fp", "fn", "tn"],
        default="unknown",
    )
    annotated["error_kind"] = np.select(
        [annotated["prediction_outcome"] == "fp", annotated["prediction_outcome"] == "fn"],
        ["fp", "fn"],
        default="",
    )
    return annotated


def _compute_metrics(table: pd.DataFrame, threshold: float | None = None) -> dict[str, Any]:
    resolved_threshold = _resolve_threshold(table, threshold)
    if table.empty:
        return {
            "sample_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "predicted_positive_count": 0,
            "predicted_negative_count": 0,
            "true_negative_count": 0,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "true_positive_count": 0,
            "false_positive_rate": None,
            "false_negative_rate": None,
            "threshold": resolved_threshold,
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "brier_score": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "roc_auc": None,
            "pr_auc": None,
        }

    annotated = _apply_threshold(table, resolved_threshold)
    y_true = annotated["label"].to_numpy(dtype=int)
    probabilities = annotated["probability"].to_numpy(dtype=float)
    predictions = annotated["predicted_label"].to_numpy(dtype=int)
    confusion = confusion_matrix(y_true, predictions, labels=[0, 1]).astype(int)
    true_negative_count, false_positive_count, false_negative_count, true_positive_count = confusion.ravel()
    negative_count = int(np.sum(y_true == 0))
    positive_count = int(np.sum(y_true == 1))

    metrics: dict[str, Any] = {
        "sample_count": int(len(annotated)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "predicted_positive_count": int(np.sum(predictions == 1)),
        "predicted_negative_count": int(np.sum(predictions == 0)),
        "true_negative_count": int(true_negative_count),
        "false_positive_count": int(false_positive_count),
        "false_negative_count": int(false_negative_count),
        "true_positive_count": int(true_positive_count),
        "false_positive_rate": (
            float(false_positive_count / negative_count) if negative_count > 0 else None
        ),
        "false_negative_rate": (
            float(false_negative_count / positive_count) if positive_count > 0 else None
        ),
        "threshold": float(resolved_threshold),
        "accuracy": float(np.mean(predictions == y_true)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": confusion.tolist(),
    }
    if len(set(y_true.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["pr_auc"] = float(average_precision_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def _compute_breakdowns(
    table: pd.DataFrame,
    fields: tuple[str, ...],
    *,
    threshold: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    breakdowns: dict[str, list[dict[str, Any]]] = {}
    if table.empty:
        return breakdowns

    for field in fields:
        if field not in table.columns:
            continue
        entries: list[dict[str, Any]] = []
        grouped = table.reset_index(drop=True).groupby(table[field].fillna("<missing>").astype(str), dropna=False)
        for group_name, group_rows in grouped:
            metrics = _compute_metrics(group_rows.reset_index(drop=True), threshold=threshold)
            entries.append({"group": str(group_name), **metrics})
        entries.sort(key=lambda item: (-item["sample_count"], item["group"]))
        breakdowns[field] = entries
    return breakdowns


def _build_threshold_grid() -> list[float]:
    values = {BASELINE_THRESHOLD}
    values.update(float(value) for value in np.linspace(0.05, 0.95, 19))
    return sorted(round(value, 4) for value in values)


def _build_threshold_grid_from_tables(*tables: pd.DataFrame) -> list[float]:
    values = set(_build_threshold_grid())
    for table in tables:
        if table.empty or "probability" not in table.columns:
            continue
        probabilities = pd.to_numeric(table["probability"], errors="coerce").dropna()
        values.update(round(float(value), 6) for value in probabilities.tolist())
    return sorted(values)


def _filter_table(table: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    if table.empty or not filters:
        return table.reset_index(drop=True)

    mask = pd.Series(True, index=table.index)
    for field, expected in filters.items():
        if field not in table.columns:
            mask &= False
            continue

        series = table[field].map(_normalize_string)
        if isinstance(expected, (list, tuple, set)):
            allowed = {_normalize_string(value) for value in expected}
            mask &= series.isin(allowed)
        else:
            mask &= series == _normalize_string(expected)
    return table.loc[mask].reset_index(drop=True)


def _constraint_excess(actual: float | None, limit: float | None) -> float:
    if actual is None or limit is None or actual <= limit:
        return 0.0
    denominator = max(abs(limit), 0.01)
    return float((actual - limit) / denominator)


def _evaluate_constraint_rule(
    table: pd.DataFrame,
    *,
    threshold: float,
    rule: dict[str, Any],
) -> dict[str, Any]:
    filtered = _filter_table(table, rule.get("filters", {}))
    metrics = _compute_metrics(filtered, threshold=threshold)
    min_sample_count = int(rule.get("min_sample_count", 0) or 0)
    sample_count = int(metrics["sample_count"])
    negative_count = int(metrics["negative_count"])
    false_positive_count = int(metrics["false_positive_count"])
    false_positive_rate = metrics.get("false_positive_rate")
    max_false_positive_count = rule.get("max_false_positive_count")
    max_false_positive_rate = rule.get("max_false_positive_rate")

    applies = sample_count >= min_sample_count and negative_count > 0
    violations: list[str] = []
    violation_score = 0.0

    if applies and max_false_positive_count is not None and false_positive_count > int(max_false_positive_count):
        violations.append(
            f"false_positive_count {false_positive_count} > {int(max_false_positive_count)}"
        )
        violation_score += _constraint_excess(float(false_positive_count), float(max_false_positive_count))

    if applies and max_false_positive_rate is not None and false_positive_rate is not None:
        if false_positive_rate > float(max_false_positive_rate):
            violations.append(
                f"false_positive_rate {false_positive_rate:.4f} > {float(max_false_positive_rate):.4f}"
            )
            violation_score += _constraint_excess(float(false_positive_rate), float(max_false_positive_rate))

    if not applies:
        if sample_count < min_sample_count:
            status = "insufficient_samples"
        elif negative_count == 0:
            status = "no_negative_examples"
        else:
            status = "not_applicable"
        satisfied = True
    else:
        status = "satisfied" if not violations else "violated"
        satisfied = not violations

    return {
        "name": str(rule.get("name", "unnamed_constraint")),
        "filters": dict(rule.get("filters", {})),
        "sample_count": sample_count,
        "negative_count": negative_count,
        "false_positive_count": false_positive_count,
        "false_positive_rate": false_positive_rate,
        "max_false_positive_count": max_false_positive_count,
        "max_false_positive_rate": max_false_positive_rate,
        "min_sample_count": min_sample_count,
        "applies": applies,
        "satisfied": satisfied,
        "status": status,
        "violation_score": float(violation_score),
        "violations": violations,
        "metrics": metrics,
    }


def _evaluate_constraints(
    table: pd.DataFrame,
    *,
    threshold: float,
    policy: dict[str, Any],
) -> dict[str, Any]:
    constraints = list(policy.get("constraints", []))
    if not constraints:
        return {
            "constraint_results": [],
            "constraints_satisfied": True,
            "constraint_violation_score": 0.0,
        }

    results = [
        _evaluate_constraint_rule(table, threshold=threshold, rule=rule)
        for rule in constraints
    ]
    applied_results = [result for result in results if result["applies"]]
    constraints_satisfied = all(result["satisfied"] for result in applied_results)
    violation_score = float(sum(result["violation_score"] for result in applied_results))
    return {
        "constraint_results": results,
        "constraints_satisfied": constraints_satisfied,
        "constraint_violation_score": violation_score,
    }


def _compute_threshold_sweep(
    table: pd.DataFrame,
    thresholds: list[float],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    sweep: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = _compute_metrics(table, threshold=threshold)
        constraints = _evaluate_constraints(table, threshold=threshold, policy=policy)
        sweep.append({
            "threshold": float(threshold),
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "false_positive_rate": metrics["false_positive_rate"],
            "predicted_positive_count": metrics["predicted_positive_count"],
            "predicted_negative_count": metrics["predicted_negative_count"],
            "false_positive_count": metrics["false_positive_count"],
            "false_negative_count": metrics["false_negative_count"],
            "confusion_matrix": metrics["confusion_matrix"],
            **constraints,
        })
    return sweep


def _selection_sort_tuple(item: dict[str, Any], metric: str) -> tuple[float, float, float, float]:
    return (
        float(item.get(metric, 0.0)),
        float(item.get("precision", 0.0)),
        float(item.get("recall", 0.0)),
        -abs(float(item.get("threshold", BASELINE_THRESHOLD)) - BASELINE_THRESHOLD),
    )


def _select_operating_point(sweep: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    selection = dict(policy.get("selection", {}))
    selection_metric = str(selection.get("metric", "f1"))

    if not sweep:
        return {
            "selection_strategy": "max_f1_tie_precision_recall_proximity_to_0.5",
            "selection_split": "val",
            "selected_threshold": BASELINE_THRESHOLD,
            "selection_metric": selection_metric,
            "constraints_requested": bool(policy.get("constraints")),
            "constraints_satisfied": True,
            "constraint_selection_mode": "empty_sweep_fallback",
            "selected_metrics": {},
            "selected_constraint_results": [],
            "selected_constraint_violation_score": 0.0,
        }

    satisfied = [item for item in sweep if item.get("constraints_satisfied", True)]
    if satisfied:
        selected = max(satisfied, key=lambda item: _selection_sort_tuple(item, selection_metric))
        constraint_selection_mode = "all_constraints_satisfied"
        constraints_satisfied = True
    else:
        selected = min(
            sweep,
            key=lambda item: (
                float(item.get("constraint_violation_score", 0.0)),
                -float(item.get(selection_metric, 0.0)),
                -float(item.get("precision", 0.0)),
                -float(item.get("recall", 0.0)),
                abs(float(item.get("threshold", BASELINE_THRESHOLD)) - BASELINE_THRESHOLD),
            ),
        )
        constraint_selection_mode = str(selection.get("fallback", "minimum_constraint_violation_then_max_f1"))
        constraints_satisfied = False

    return {
        "selection_strategy": "max_f1_tie_precision_recall_proximity_to_0.5",
        "selection_split": "val",
        "selected_threshold": float(selected["threshold"]),
        "selection_metric": selection_metric,
        "constraints_requested": bool(policy.get("constraints")),
        "constraints_satisfied": constraints_satisfied,
        "constraint_selection_mode": constraint_selection_mode,
        "selected_metrics": dict(selected),
        "selected_constraint_results": copy.deepcopy(selected.get("constraint_results", [])),
        "selected_constraint_violation_score": float(selected.get("constraint_violation_score", 0.0)),
    }


def _resolve_report_path_matches(report: Any, path: str) -> list[tuple[str, Any]]:
    if not path:
        return []

    segments = [segment for segment in path.split(".") if segment]
    matches: list[tuple[str, Any]] = []

    def visit(current: Any, index: int, prefix: list[str]) -> None:
        if index >= len(segments):
            matches.append((".".join(prefix), current))
            return

        segment = segments[index]
        has_pattern = any(char in segment for char in "*?[]")

        if isinstance(current, dict):
            if has_pattern:
                for key, value in current.items():
                    key_str = str(key)
                    if fnmatchcase(key_str, segment):
                        visit(value, index + 1, [*prefix, key_str])
                return
            if segment in current:
                visit(current[segment], index + 1, [*prefix, segment])
            return

        if isinstance(current, list):
            if has_pattern:
                for list_index, value in enumerate(current):
                    index_str = str(list_index)
                    if fnmatchcase(index_str, segment):
                        visit(value, index + 1, [*prefix, index_str])
                return
            if segment.isdigit():
                list_index = int(segment)
                if 0 <= list_index < len(current):
                    visit(current[list_index], index + 1, [*prefix, segment])

    visit(report, 0, [])
    return matches


def _aggregate_match_values(values: list[Any], aggregation: str | None) -> Any:
    if aggregation is None:
        if len(values) == 1:
            return values[0]
        return values

    available_values = [value for value in values if value is not None]
    if not available_values:
        return None

    resolved_aggregation = aggregation.strip().casefold()
    if resolved_aggregation == "max":
        return max(float(value) for value in available_values)
    if resolved_aggregation == "min":
        return min(float(value) for value in available_values)
    if resolved_aggregation == "any":
        return any(bool(value) for value in available_values)
    if resolved_aggregation == "all":
        return all(bool(value) for value in available_values)
    raise ValueError(f"Unsupported acceptance-criteria aggregation: {aggregation}")


def _evaluate_acceptance_criterion(report: dict[str, Any], criterion: dict[str, Any]) -> dict[str, Any]:
    name = str(criterion.get("name", "unnamed_acceptance_criterion"))
    path = str(criterion.get("path", "")).strip()
    skip_if_missing = bool(criterion.get("skip_if_missing", False))
    aggregation = criterion.get("aggregation")
    matches = _resolve_report_path_matches(report, path)

    if not matches or all(value is None for _, value in matches):
        if skip_if_missing:
            return {
                "name": name,
                "path": path,
                "aggregation": aggregation,
                "match_count": 0,
                "matched_paths": [],
                "observed_value": None,
                "status": "skipped_missing",
                "satisfied": True,
                "violations": [],
            }
        return {
            "name": name,
            "path": path,
            "aggregation": aggregation,
            "match_count": 0,
            "matched_paths": [],
            "observed_value": None,
            "status": "missing",
            "satisfied": False,
            "violations": [f"path not found: {path}"],
        }

    observed_value = _aggregate_match_values([value for _, value in matches], aggregation if isinstance(aggregation, str) else None)
    violations: list[str] = []

    if isinstance(observed_value, list) and any(
        key in criterion for key in ("equals", "min_value", "max_value")
    ):
        return {
            "name": name,
            "path": path,
            "aggregation": aggregation,
            "match_count": len(matches),
            "matched_paths": [matched_path for matched_path, _ in matches],
            "observed_value": observed_value,
            "status": "invalid",
            "satisfied": False,
            "violations": ["multiple matches require an explicit aggregation"],
        }

    if "equals" in criterion and observed_value != criterion.get("equals"):
        violations.append(f"expected equals {criterion.get('equals')}, observed {observed_value}")

    if "min_value" in criterion:
        min_value = float(criterion["min_value"])
        if observed_value is None or float(observed_value) < min_value:
            violations.append(f"expected >= {min_value}, observed {observed_value}")

    if "max_value" in criterion:
        max_value = float(criterion["max_value"])
        if observed_value is None or float(observed_value) > max_value:
            violations.append(f"expected <= {max_value}, observed {observed_value}")

    result = {
        "name": name,
        "path": path,
        "aggregation": aggregation,
        "match_count": len(matches),
        "matched_paths": [matched_path for matched_path, _ in matches],
        "observed_value": observed_value,
        "status": "passed" if not violations else "failed",
        "satisfied": not violations,
        "violations": violations,
    }
    if "equals" in criterion:
        result["expected_equals"] = criterion.get("equals")
    if "min_value" in criterion:
        result["expected_min_value"] = criterion.get("min_value")
    if "max_value" in criterion:
        result["expected_max_value"] = criterion.get("max_value")
    return result


def _evaluate_acceptance_criteria(report: dict[str, Any], criteria_config: dict[str, Any]) -> dict[str, Any]:
    criteria = list(criteria_config.get("criteria", []))
    if not criteria:
        return {
            "configured": False,
            "criteria_count": 0,
            "evaluated_count": 0,
            "passed_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "criteria_satisfied": True,
            "criteria_results": [],
        }

    results = [_evaluate_acceptance_criterion(report, criterion) for criterion in criteria]
    failed_count = sum(1 for result in results if result["status"] == "failed")
    skipped_count = sum(1 for result in results if result["status"].startswith("skipped"))
    passed_count = sum(1 for result in results if result["status"] == "passed")

    return {
        "configured": True,
        "criteria_count": len(criteria),
        "evaluated_count": len(results) - skipped_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "criteria_satisfied": failed_count == 0,
        "criteria_results": results,
    }


def _compute_calibration(table: pd.DataFrame, *, bin_count: int = 10) -> dict[str, Any]:
    if table.empty:
        return {
            "bin_count": bin_count,
            "ece": 0.0,
            "max_calibration_error": 0.0,
            "mean_probability": 0.0,
            "positive_rate": 0.0,
            "bins": [],
        }

    probabilities = pd.to_numeric(table["probability"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    labels = pd.to_numeric(table["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    bin_edges = np.linspace(0.0, 1.0, bin_count + 1)
    bin_indices = np.digitize(probabilities, bin_edges[1:-1], right=False)

    bins: list[dict[str, Any]] = []
    ece = 0.0
    max_calibration_error = 0.0
    total_count = len(probabilities)

    for index in range(bin_count):
        mask = bin_indices == index
        bin_probabilities = probabilities[mask]
        bin_labels = labels[mask]
        sample_count = int(mask.sum())

        if sample_count > 0:
            mean_probability = float(bin_probabilities.mean())
            empirical_positive_rate = float(bin_labels.mean())
            absolute_gap = abs(mean_probability - empirical_positive_rate)
            ece += (sample_count / total_count) * absolute_gap
            max_calibration_error = max(max_calibration_error, absolute_gap)
        else:
            mean_probability = None
            empirical_positive_rate = None
            absolute_gap = None

        bins.append({
            "lower_bound": float(bin_edges[index]),
            "upper_bound": float(bin_edges[index + 1]),
            "sample_count": sample_count,
            "mean_probability": mean_probability,
            "empirical_positive_rate": empirical_positive_rate,
            "absolute_gap": absolute_gap,
        })

    return {
        "bin_count": bin_count,
        "ece": float(ece),
        "max_calibration_error": float(max_calibration_error),
        "mean_probability": float(probabilities.mean()),
        "positive_rate": float(labels.mean()),
        "bins": bins,
    }


def _summarize_errors(
    table: pd.DataFrame,
    *,
    top_errors: int,
    threshold: float | None = None,
) -> dict[str, Any]:
    annotated = _apply_threshold(table, threshold)
    if annotated.empty:
        return {
            "false_positive_count": 0,
            "false_negative_count": 0,
            "top_false_positives": [],
            "top_false_negatives": [],
        }

    false_positives = annotated[annotated["prediction_outcome"] == "fp"].copy()
    false_negatives = annotated[annotated["prediction_outcome"] == "fn"].copy()

    false_positives = false_positives.sort_values(["probability", "sha256"], ascending=[False, True])
    false_negatives = false_negatives.sort_values(["probability", "sha256"], ascending=[True, True])

    columns = [
        "sample_id",
        "sha256",
        "original_name",
        "relative_path",
        "pe_kind",
        "source",
        "source_family",
        "label",
        "predicted_label",
        "probability",
        "threshold",
    ]
    available_columns = [column for column in columns if column in false_positives.columns]
    return {
        "false_positive_count": int(len(false_positives)),
        "false_negative_count": int(len(false_negatives)),
        "top_false_positives": false_positives.loc[:, available_columns].head(top_errors).to_dict(orient="records"),
        "top_false_negatives": false_negatives.loc[:, available_columns].head(top_errors).to_dict(orient="records"),
    }


def _resolve_output_paths(
    experiment_log_path: Path,
    *,
    profile_slug: str,
    output_root: str | Path | None,
) -> EvaluationArtifactPaths:
    if output_root is None:
        base_dir = experiment_log_path.parent
        stem = "evaluation_report"
    else:
        base_dir = Path(output_root).resolve()
        stem = f"{profile_slug}_evaluation_report"
    base_dir.mkdir(parents=True, exist_ok=True)
    return EvaluationArtifactPaths(
        report_json_path=base_dir / f"{stem}.json",
        report_markdown_path=base_dir / f"{stem}.md",
    )


def _render_metric_rows(metrics: dict[str, Any]) -> list[str]:
    return [
        f"| sample_count | {metrics.get('sample_count', 0)} |",
        f"| positive_count | {metrics.get('positive_count', 0)} |",
        f"| negative_count | {metrics.get('negative_count', 0)} |",
        f"| predicted_positive_count | {metrics.get('predicted_positive_count', 0)} |",
        f"| predicted_negative_count | {metrics.get('predicted_negative_count', 0)} |",
        f"| false_positive_rate | {_format_optional(metrics.get('false_positive_rate'))} |",
        f"| false_negative_rate | {_format_optional(metrics.get('false_negative_rate'))} |",
        f"| threshold | {metrics.get('threshold', BASELINE_THRESHOLD):.4f} |",
        f"| accuracy | {metrics.get('accuracy', 0.0):.4f} |",
        f"| f1 | {metrics.get('f1', 0.0):.4f} |",
        f"| precision | {metrics.get('precision', 0.0):.4f} |",
        f"| recall | {metrics.get('recall', 0.0):.4f} |",
        f"| brier_score | {metrics.get('brier_score', 0.0):.4f} |",
        f"| roc_auc | {_format_optional(metrics.get('roc_auc'))} |",
        f"| pr_auc | {_format_optional(metrics.get('pr_auc'))} |",
        f"| confusion_matrix | {metrics.get('confusion_matrix', [[0, 0], [0, 0]])} |",
    ]


def _render_breakdown_table(entries: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| group | samples | negatives | fp | fp_rate | f1 | precision | recall | roc_auc | pr_auc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in entries:
        lines.append(
            "| {group} | {sample_count} | {negative_count} | {false_positive_count} | {false_positive_rate} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {roc_auc} | {pr_auc} |".format(
                group=entry["group"],
                sample_count=entry["sample_count"],
                negative_count=entry["negative_count"],
                false_positive_count=entry["false_positive_count"],
                false_positive_rate=_format_optional(entry.get("false_positive_rate")),
                f1=entry["f1"],
                precision=entry["precision"],
                recall=entry["recall"],
                roc_auc=_format_optional(entry.get("roc_auc")),
                pr_auc=_format_optional(entry.get("pr_auc")),
            )
        )
    return lines


def _render_error_table(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    header_columns = [
        "sha256",
        "relative_path",
        "original_name",
        "pe_kind",
        "source_family",
        "probability",
        "label",
        "predicted_label",
    ]
    display_columns = [column for column in header_columns if column in records[0]]
    lines = [
        "| " + " | ".join(display_columns) + " |",
        "| " + " | ".join("---" if column not in {"probability", "label", "predicted_label"} else "---:" for column in display_columns) + " |",
    ]
    for record in records:
        row_values: list[str] = []
        for column in display_columns:
            value = record.get(column, "")
            if isinstance(value, float):
                row_values.append(f"{value:.4f}")
            else:
                row_values.append(str(value))
        lines.append("| " + " | ".join(row_values) + " |")
    return lines


def _render_threshold_sweep(entries: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| threshold | f1 | precision | recall | fp | fp_rate | violation_score | constraints_satisfied |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in entries:
        lines.append(
            "| {threshold:.4f} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {false_positive_count} | {false_positive_rate} | {constraint_violation_score:.4f} | {constraints_satisfied} |".format(
                threshold=float(entry["threshold"]),
                f1=float(entry["f1"]),
                precision=float(entry["precision"]),
                recall=float(entry["recall"]),
                false_positive_count=int(entry["false_positive_count"]),
                false_positive_rate=_format_optional(entry.get("false_positive_rate")),
                constraint_violation_score=float(entry.get("constraint_violation_score", 0.0)),
                constraints_satisfied=str(bool(entry.get("constraints_satisfied", True))).lower(),
            )
        )
    return lines


def _render_calibration(calibration: dict[str, Any]) -> list[str]:
    lines = [
        f"- ece: {_format_optional(calibration.get('ece'))}",
        f"- max_calibration_error: {_format_optional(calibration.get('max_calibration_error'))}",
        f"- mean_probability: {_format_optional(calibration.get('mean_probability'))}",
        f"- positive_rate: {_format_optional(calibration.get('positive_rate'))}",
        "",
        "| bin | range | samples | mean_probability | empirical_positive_rate | abs_gap |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for index, entry in enumerate(calibration.get("bins", []), start=1):
        lines.append(
            "| {index} | [{lower:.2f}, {upper:.2f}) | {count} | {mean_probability} | {positive_rate} | {gap} |".format(
                index=index,
                lower=entry["lower_bound"],
                upper=entry["upper_bound"],
                count=entry["sample_count"],
                mean_probability=_format_optional(entry.get("mean_probability")),
                positive_rate=_format_optional(entry.get("empirical_positive_rate")),
                gap=_format_optional(entry.get("absolute_gap")),
            )
        )
    return lines


def _render_constraint_results(results: list[dict[str, Any]]) -> list[str]:
    if not results:
        return []
    lines = [
        "| constraint | status | samples | negatives | fp | fp_rate | max_fp | max_fp_rate | violation_score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| {name} | {status} | {sample_count} | {negative_count} | {false_positive_count} | {false_positive_rate} | {max_false_positive_count} | {max_false_positive_rate} | {violation_score:.4f} |".format(
                name=result["name"],
                status=result["status"],
                sample_count=result["sample_count"],
                negative_count=result["negative_count"],
                false_positive_count=result["false_positive_count"],
                false_positive_rate=_format_optional(result.get("false_positive_rate")),
                max_false_positive_count=_format_optional(result.get("max_false_positive_count")),
                max_false_positive_rate=_format_optional(result.get("max_false_positive_rate")),
                violation_score=float(result.get("violation_score", 0.0)),
            )
        )
    return lines


def _render_acceptance_results(acceptance: dict[str, Any]) -> list[str]:
    if not acceptance.get("configured"):
        return ["- configured: false"]

    lines = [
        f"- configured: {str(bool(acceptance.get('configured'))).lower()}",
        f"- criteria_satisfied: {str(bool(acceptance.get('criteria_satisfied'))).lower()}",
        f"- passed_count: {acceptance.get('passed_count', 0)}",
        f"- failed_count: {acceptance.get('failed_count', 0)}",
        f"- skipped_count: {acceptance.get('skipped_count', 0)}",
        "",
        "| criterion | status | observed | path | aggregation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in acceptance.get("criteria_results", []):
        lines.append(
            "| {name} | {status} | {observed} | {path} | {aggregation} |".format(
                name=result.get("name", ""),
                status=result.get("status", ""),
                observed=_format_optional(result.get("observed_value")),
                path=result.get("path", ""),
                aggregation=_format_optional(result.get("aggregation")),
            )
        )
        if result.get("violations"):
            lines.append(f"|  | violations | {'; '.join(result['violations'])} |  |  |")
    return lines


def _render_markdown(report: dict[str, Any]) -> str:
    operating_point = report["operating_point"]
    acceptance = report.get("acceptance_criteria", {})
    lines: list[str] = [
        f"# Evaluation report: {report['profile_slug']}",
        "",
        f"- Generated at (UTC): {report['generated_at_utc']}",
        f"- Experiment log: {report['experiment_log_path']}",
        f"- Selected model: {report['selected_model_name']}",
        f"- Operating threshold: {operating_point['selected_threshold']:.4f}",
        f"- Suspicious threshold: {operating_point.get('selected_suspicious_threshold', BASELINE_THRESHOLD):.4f}",
        f"- Baseline threshold: {operating_point['baseline_threshold']:.4f}",
        f"- Operating-point strategy: {operating_point['selection_strategy']}",
        f"- Constraint selection mode: {operating_point['constraint_selection_mode']}",
        f"- Constraints satisfied on selection split: {str(bool(operating_point.get('constraints_satisfied', True))).lower()}",
    ]

    lines.extend([
        "",
        "## Acceptance criteria",
        "",
        *_render_acceptance_results(acceptance),
    ])

    selected_constraint_results = operating_point.get("selected_constraint_results", [])
    if selected_constraint_results:
        lines.extend([
            "",
            "## Operating-point constraints",
            "",
            *_render_constraint_results(selected_constraint_results),
        ])

    for split_name in report.get("split_order", list(report["splits"].keys())):
        split = report["splits"].get(split_name, {})
        lines.extend([
            "",
            f"## {split_name.upper()} overall at operating point",
            "",
            "| metric | value |",
            "| --- | --- |",
            *_render_metric_rows(split.get("overall_metrics", {})),
            "",
            f"## {split_name.upper()} baseline at threshold {BASELINE_THRESHOLD:.2f}",
            "",
            "| metric | value |",
            "| --- | --- |",
            *_render_metric_rows(split.get("baseline_threshold_metrics", {})),
            "",
            f"## {split_name.upper()} calibration",
            "",
            *_render_calibration(split.get("calibration", {})),
            "",
            f"## {split_name.upper()} threshold sweep",
            "",
            *_render_threshold_sweep(split.get("threshold_sweep", [])),
        ])

        if split.get("constraint_results"):
            lines.extend([
                "",
                f"### {split_name.upper()} operating-point constraints",
                "",
                *_render_constraint_results(split.get("constraint_results", [])),
            ])

        for field, entries in split.get("breakdowns", {}).items():
            lines.extend([
                "",
                f"### {split_name.upper()} breakdown at operating point: {field}",
                "",
                *_render_breakdown_table(entries),
            ])

        error_summary = split.get("error_summary", {})
        lines.extend([
            "",
            f"### {split_name.upper()} errors at operating point",
            "",
            f"- false_positive_count: {error_summary.get('false_positive_count', 0)}",
            f"- false_negative_count: {error_summary.get('false_negative_count', 0)}",
        ])

        fp_rows = error_summary.get("top_false_positives", [])
        fn_rows = error_summary.get("top_false_negatives", [])
        if fp_rows:
            lines.extend(["", f"#### {split_name.upper()} top false positives", "", *_render_error_table(fp_rows)])
        if fn_rows:
            lines.extend(["", f"#### {split_name.upper()} top false negatives", "", *_render_error_table(fn_rows)])

    return "\n".join(lines) + "\n"


def _build_split_report(
    table: pd.DataFrame,
    *,
    thresholds: list[float],
    operating_threshold: float,
    top_errors: int,
    operating_point_policy: dict[str, Any],
) -> dict[str, Any]:
    operating_constraints = _evaluate_constraints(table, threshold=operating_threshold, policy=operating_point_policy)
    baseline_constraints = _evaluate_constraints(table, threshold=BASELINE_THRESHOLD, policy=operating_point_policy)
    return {
        "overall_metrics": _compute_metrics(table, threshold=operating_threshold),
        "baseline_threshold_metrics": _compute_metrics(table, threshold=BASELINE_THRESHOLD),
        "calibration": _compute_calibration(table),
        "threshold_sweep": _compute_threshold_sweep(table, thresholds, operating_point_policy),
        "breakdowns": _compute_breakdowns(table, BREAKDOWN_FIELDS, threshold=operating_threshold),
        "baseline_breakdowns": _compute_breakdowns(table, BREAKDOWN_FIELDS, threshold=BASELINE_THRESHOLD),
        "constraint_results": operating_constraints["constraint_results"],
        "baseline_constraint_results": baseline_constraints["constraint_results"],
        "error_summary": _summarize_errors(table, top_errors=top_errors, threshold=operating_threshold),
        "baseline_error_summary": _summarize_errors(table, top_errors=top_errors, threshold=BASELINE_THRESHOLD),
    }


def _normalize_relative_path(value: Any) -> str:
    return Path(str(value)).as_posix()


def _load_batch_summary_frame(batch_summary_path: str | Path) -> pd.DataFrame:
    resolved_path = Path(batch_summary_path).resolve()
    if resolved_path.suffix.casefold() == ".json":
        payload = _load_json(resolved_path)
        records = payload.get("summary_records", [])
        return pd.DataFrame(records)
    return pd.read_csv(resolved_path)


def load_prediction_table_from_batch_summary(
    batch_summary_path: str | Path,
    label_manifest_path: str | Path,
) -> pd.DataFrame:
    summary_df = _load_batch_summary_frame(batch_summary_path)
    manifest_df = pd.read_csv(label_manifest_path)

    required_summary_columns = {"relative_path", "status", "probability"}
    required_manifest_columns = {"relative_path", "label", "pe_kind", "source", "source_family"}
    missing_summary_columns = required_summary_columns - set(summary_df.columns)
    missing_manifest_columns = required_manifest_columns - set(manifest_df.columns)
    if missing_summary_columns:
        raise ValueError(f"Batch summary is missing required columns: {sorted(missing_summary_columns)}")
    if missing_manifest_columns:
        raise ValueError(f"Label manifest is missing required columns: {sorted(missing_manifest_columns)}")

    summary_df = summary_df.copy()
    manifest_df = manifest_df.copy()
    summary_df["relative_path"] = summary_df["relative_path"].map(_normalize_relative_path)
    manifest_df["relative_path"] = manifest_df["relative_path"].map(_normalize_relative_path)

    ok_summary_df = summary_df[summary_df["status"].astype(str).str.casefold() == "ok"].copy()
    merged = manifest_df.merge(ok_summary_df, on="relative_path", how="left", suffixes=("_manifest", "_summary"))

    missing_predictions = merged["probability"].isna()
    if bool(missing_predictions.any()):
        missing_paths = merged.loc[missing_predictions, "relative_path"].tolist()
        preview = missing_paths[:5]
        raise ValueError(
            "Some manifest rows are missing successful predictions in the batch summary: "
            + ", ".join(preview)
        )

    original_name_series = (
        merged["original_name"]
        if "original_name" in merged.columns
        else merged["relative_path"].map(lambda value: Path(str(value)).name)
    )
    filename_series = (
        merged["filename"]
        if "filename" in merged.columns
        else merged["relative_path"].map(lambda value: Path(str(value)).name)
    )
    sha256_series = merged["sha256"] if "sha256" in merged.columns else pd.Series([""] * len(merged))
    sample_id_series = merged["sample_id"] if "sample_id" in merged.columns else pd.Series([""] * len(merged))

    return pd.DataFrame({
        "sample_id": sample_id_series.fillna("").astype(str),
        "sha256": sha256_series.fillna("").astype(str),
        "original_name": original_name_series.fillna(filename_series).astype(str),
        "filename": filename_series.fillna("").astype(str),
        "relative_path": merged["relative_path"].astype(str),
        "input_path": merged["input_path"].fillna("").astype(str) if "input_path" in merged.columns else "",
        "pe_kind": merged["pe_kind"].astype(str),
        "source": merged["source"].astype(str),
        "source_family": merged["source_family"].astype(str),
        "label": pd.to_numeric(merged["label"], errors="coerce").fillna(0).astype(int),
        "probability": pd.to_numeric(merged["probability"], errors="coerce").fillna(0.0).astype(float),
        "threshold": float(BASELINE_THRESHOLD),
    })


def build_evaluation_report(
    experiment_log_path: str | Path,
    *,
    output_root: str | Path | None = None,
    top_errors: int = 25,
    operating_point_policy: dict[str, Any] | str | Path | None = None,
    acceptance_criteria: dict[str, Any] | str | Path | None = None,
    external_prediction_tables: dict[str, pd.DataFrame] | None = None,
    external_artifacts: dict[str, Any] | None = None,
) -> EvaluationReportResult:
    resolved_experiment_log_path = Path(experiment_log_path).resolve()
    experiment_log = _load_json(resolved_experiment_log_path)
    profile_slug = str(experiment_log.get("profile_slug", "unknown"))
    artifacts = experiment_log.get("artifacts", {})
    experiment_log_dir = resolved_experiment_log_path.parent

    output_paths = _resolve_output_paths(
        resolved_experiment_log_path,
        profile_slug=profile_slug,
        output_root=output_root,
    )
    report_base_dir = output_paths.report_json_path.parent

    val_predictions_path = resolve_stored_path(artifacts["val_predictions_path"], base_dir=experiment_log_dir)
    test_predictions_path = resolve_stored_path(artifacts["test_predictions_path"], base_dir=experiment_log_dir)
    val_table = pd.read_csv(val_predictions_path)
    test_table = pd.read_csv(test_predictions_path)
    resolved_policy = _load_operating_point_policy(operating_point_policy)
    resolved_acceptance_criteria = _load_acceptance_criteria(acceptance_criteria)

    threshold_grid = _build_threshold_grid_from_tables(val_table, test_table, *(external_prediction_tables or {}).values())
    val_threshold_sweep = _compute_threshold_sweep(val_table, threshold_grid, resolved_policy)
    operating_point = _select_operating_point(val_threshold_sweep, resolved_policy)
    operating_point["baseline_threshold"] = BASELINE_THRESHOLD
    operating_point["policy"] = copy.deepcopy(resolved_policy)
    operating_threshold = float(operating_point["selected_threshold"])
    operating_point["selected_suspicious_threshold"] = resolve_suspicious_threshold(
        operating_threshold,
        resolved_policy,
    )

    splits: dict[str, dict[str, Any]] = {
        "val": {
            **_build_split_report(
                val_table,
                thresholds=threshold_grid,
                operating_threshold=operating_threshold,
                top_errors=top_errors,
                operating_point_policy=resolved_policy,
            ),
            "threshold_sweep": val_threshold_sweep,
        },
        "test": _build_split_report(
            test_table,
            thresholds=threshold_grid,
            operating_threshold=operating_threshold,
            top_errors=top_errors,
            operating_point_policy=resolved_policy,
        ),
    }

    for split_name, table in (external_prediction_tables or {}).items():
        splits[split_name] = _build_split_report(
            table,
            thresholds=threshold_grid,
            operating_threshold=operating_threshold,
            top_errors=top_errors,
            operating_point_policy=resolved_policy,
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_log_path": serialize_relative_path(resolved_experiment_log_path, base_dir=report_base_dir),
        "profile_slug": profile_slug,
        "selected_model_name": experiment_log.get("selected_model", {}).get("name", "unknown"),
        "feature_metadata": experiment_log.get("feature_metadata", {}),
        "artifacts": {
            "val_predictions_path": serialize_relative_path(val_predictions_path, base_dir=report_base_dir),
            "test_predictions_path": serialize_relative_path(test_predictions_path, base_dir=report_base_dir),
            **(external_artifacts or {}),
        },
        "operating_point": {
            **operating_point,
            "selection_split_baseline_metrics": _compute_metrics(val_table, threshold=BASELINE_THRESHOLD),
            "selection_split_operating_metrics": _compute_metrics(val_table, threshold=operating_threshold),
        },
        "split_order": list(splits.keys()),
        "splits": splits,
    }
    report["acceptance_criteria"] = _evaluate_acceptance_criteria(report, resolved_acceptance_criteria)
    output_paths.report_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    output_paths.report_markdown_path.write_text(_render_markdown(report), encoding="utf-8")

    return EvaluationReportResult(
        experiment_log_path=resolved_experiment_log_path,
        profile_slug=profile_slug,
        report=report,
        artifact_paths=output_paths,
    )