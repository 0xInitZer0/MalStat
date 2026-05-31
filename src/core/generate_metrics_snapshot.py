"""Generate the README metrics image from an evaluation report JSON artifact."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "models" / "evaluation_report.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "pict03.png"


@dataclass(frozen=True)
class SplitMetrics:
    name: str
    sample_count: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float | None
    pr_auc: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    confusion_matrix: list[list[int]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _split_metrics(split_name: str, payload: dict[str, Any]) -> SplitMetrics:
    metrics = payload.get("splits", {}).get(split_name, {}).get("overall_metrics", {})
    matrix = metrics.get("confusion_matrix") or [[0, 0], [0, 0]]
    return SplitMetrics(
        name=split_name.upper(),
        sample_count=_safe_int(metrics.get("sample_count")),
        accuracy=_safe_float(metrics.get("accuracy")) or 0.0,
        f1=_safe_float(metrics.get("f1")) or 0.0,
        precision=_safe_float(metrics.get("precision")) or 0.0,
        recall=_safe_float(metrics.get("recall")) or 0.0,
        roc_auc=_safe_float(metrics.get("roc_auc")),
        pr_auc=_safe_float(metrics.get("pr_auc")),
        false_positive_rate=_safe_float(metrics.get("false_positive_rate")),
        false_negative_rate=_safe_float(metrics.get("false_negative_rate")),
        confusion_matrix=[[int(cell) for cell in row] for row in matrix],
    )


def _format_percent(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_delta_percent(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.{digits}f}%"


def _humanize_slug(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return text.replace("_", " ")


def _selection_label(strategy: Any) -> str:
    normalized = str(strategy or "").strip().casefold()
    mapping = {
        "max_f1_tie_precision_recall_proximity_to_0.5": "Max F1",
        "minimum_constraint_violation_then_max_f1": "Min violation",
    }
    return mapping.get(normalized, _humanize_slug(strategy).title())


def _criterion_label(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    mapping = {
        "operating_constraints_satisfied": "Operating constraints",
        "test_recall_min": "Test recall min",
        "test_pr_auc_min": "Test PR AUC min",
        "external_benign_fp_rate_max": "External benign FP-rate",
        "external_benign_fp_count_max": "External benign FP-count",
    }
    return mapping.get(normalized, _humanize_slug(value).title())


def _status_label(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    mapping = {
        "passed": "Passed",
        "failed": "Failed",
        "skipped_missing": "Skipped",
    }
    return mapping.get(normalized, _humanize_slug(value).title())


def _generated_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    if "T" in text and text.endswith("Z"):
        date_part, time_part = text[:-1].split("T", 1)
        return f"{date_part} {time_part[:5]}Z"
    return text


def _add_clipped_text(
    ax: Any,
    patch: Any,
    x: float,
    y: float,
    text: str,
    **kwargs: Any,
) -> Any:
    kwargs.setdefault("clip_on", True)
    artist = ax.text(x, y, text, **kwargs)
    artist.set_clip_path(patch)
    return artist


def _draw_card(
    ax: Any,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    value: str,
    subtitle: str = "",
    facecolor: str = "#fffaf0",
    edgecolor: str = "#d7c5a7",
    value_color: str = "#203040",
) -> None:
    compact_value = textwrap.shorten(value, width=22, placeholder="...")
    compact_subtitle = textwrap.shorten(subtitle, width=30, placeholder="...") if subtitle else ""
    value_fontsize = 18 if len(compact_value) <= 16 else 15 if len(compact_value) <= 20 else 12.5
    subtitle_fontsize = 9 if len(compact_subtitle) <= 24 else 8.0

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    _add_clipped_text(ax, patch, x + 0.025, y + height - 0.08, title, fontsize=10, color="#5f6c7b", va="top", ha="left")
    _add_clipped_text(
        ax,
        patch,
        x + 0.025,
        y + height - 0.19,
        compact_value,
        fontsize=value_fontsize,
        color=value_color,
        va="top",
        ha="left",
        weight="bold",
    )
    if compact_subtitle:
        _add_clipped_text(
            ax,
            patch,
            x + 0.025,
            y + 0.08,
            compact_subtitle,
            fontsize=subtitle_fontsize,
            color="#6e7a89",
            va="bottom",
            ha="left",
        )


def _render_header(ax: Any, report: dict[str, Any], operating_point: dict[str, Any], acceptance: dict[str, Any]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    generated_at = str(report.get("generated_at_utc", "unknown"))
    model_name = str(report.get("selected_model_name", "unknown"))
    profile_slug = str(report.get("profile_slug", "unknown"))
    selected_threshold = _safe_float(operating_point.get("selected_threshold")) or 0.0
    suspicious_threshold = _safe_float(operating_point.get("selected_suspicious_threshold")) or selected_threshold
    passed_count = _safe_int(acceptance.get("passed_count"))
    failed_count = _safe_int(acceptance.get("failed_count"))
    skipped_count = _safe_int(acceptance.get("skipped_count"))
    criteria_satisfied = bool(acceptance.get("criteria_satisfied", False))

    ax.text(0.02, 0.95, "MalStat Evaluation Snapshot", fontsize=25, weight="bold", color="#17324d", va="top")
    ax.text(
        0.02,
        0.72,
        "README metrics image generated from models/evaluation_report.json",
        fontsize=10,
        color="#5f6c7b",
        va="top",
    )

    cards = [
        ("Model", model_name, f"Profile: {_humanize_slug(profile_slug)}", "#fff6e8"),
        ("Operating threshold", _format_float(selected_threshold, digits=6), f"Suspicious: {_format_float(suspicious_threshold, digits=6)}", "#eef8f2"),
        ("Acceptance", "PASS" if criteria_satisfied else "CHECK", f"passed={passed_count} failed={failed_count} skipped={skipped_count}", "#eef7ff" if criteria_satisfied else "#fff1f1"),
        (
            "Selection strategy",
            _selection_label(operating_point.get("selection_strategy", "unknown")),
            f"mode: {textwrap.shorten(_humanize_slug(operating_point.get('constraint_selection_mode', 'unknown')), width=24, placeholder='...')}",
            "#f6f2ff",
        ),
        ("Generated", _generated_label(generated_at), f"source: {report.get('experiment_log_path', '')}", "#fffaf0"),
    ]

    left = 0.02
    gap = 0.012
    width = (0.96 - gap * (len(cards) - 1)) / len(cards)
    for index, (title, value, subtitle, facecolor) in enumerate(cards):
        _draw_card(
            ax,
            x=left + index * (width + gap),
            y=0.08,
            width=width,
            height=0.52,
            title=title,
            value=value,
            subtitle=subtitle,
            facecolor=facecolor,
        )


def _render_metric_chart(ax: Any, val_metrics: SplitMetrics, test_metrics: SplitMetrics) -> None:
    metric_names = ["Accuracy", "F1", "Precision", "Recall", "ROC AUC", "PR AUC"]
    val_values = [
        val_metrics.accuracy,
        val_metrics.f1,
        val_metrics.precision,
        val_metrics.recall,
        val_metrics.roc_auc or 0.0,
        val_metrics.pr_auc or 0.0,
    ]
    test_values = [
        test_metrics.accuracy,
        test_metrics.f1,
        test_metrics.precision,
        test_metrics.recall,
        test_metrics.roc_auc or 0.0,
        test_metrics.pr_auc or 0.0,
    ]

    x_positions = np.arange(len(metric_names))
    width = 0.35
    val_bars = ax.bar(x_positions - width / 2, val_values, width=width, color="#1f6f8b", label=val_metrics.name)
    test_bars = ax.bar(x_positions + width / 2, test_values, width=width, color="#f4a261", label=test_metrics.name)

    ax.set_title("Operating-point metrics by split", loc="left", fontsize=14, weight="bold", color="#17324d", pad=10)
    ax.set_xticks(x_positions, metric_names)
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.20)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10.5)


def _render_confusion_matrix(ax: Any, split_metrics: SplitMetrics, accent: str, *, show_y_labels: bool | None = None) -> None:
    matrix = np.array(split_metrics.confusion_matrix, dtype=float)
    vmax = max(float(matrix.max()), 1.0)
    cmap = LinearSegmentedColormap.from_list("malstat_confusion", ["#fff9ef", accent])

    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(f"{split_metrics.name} confusion matrix", fontsize=13, weight="bold", color="#17324d")
    ax.set_xticks([0, 1], ["Pred clean", "Pred malware"])
    resolved_show_y_labels = split_metrics.name == "VAL" if show_y_labels is None else show_y_labels
    ax.set_yticks([0, 1], ["Actual clean", "Actual malware"] if resolved_show_y_labels else ["", ""])
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", labelsize=9.5, labelleft=resolved_show_y_labels)

    threshold = vmax * 0.55
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            text_color = "white" if matrix[row_index, column_index] >= threshold else "#17324d"
            ax.text(column_index, row_index, str(value), ha="center", va="center", fontsize=15, weight="bold", color=text_color)

    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _render_threshold_summary(ax: Any, operating_point: dict[str, Any], val_metrics: SplitMetrics, test_metrics: SplitMetrics) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 1.0, "Threshold and error summary", fontsize=14, weight="bold", color="#17324d", va="top")

    baseline = operating_point.get("selection_split_baseline_metrics", {})
    operating = operating_point.get("selection_split_operating_metrics", {})

    baseline_threshold = _safe_float(operating_point.get("baseline_threshold"))
    selected_threshold = _safe_float(operating_point.get("selected_threshold"))
    f1_delta = (_safe_float(operating.get("f1")) or 0.0) - (_safe_float(baseline.get("f1")) or 0.0)
    recall_delta = (_safe_float(operating.get("recall")) or 0.0) - (_safe_float(baseline.get("recall")) or 0.0)
    fp_delta = (_safe_float(operating.get("false_positive_rate")) or 0.0) - (_safe_float(baseline.get("false_positive_rate")) or 0.0)

    blocks = [
        (
            "Threshold tuning",
            [
                f"Baseline thr: {_format_float(baseline_threshold, digits=4)}",
                f"Operating thr: {_format_float(selected_threshold, digits=6)}",
                f"F1 delta: {_format_delta_percent(f1_delta, digits=2)}",
                f"Recall delta: {_format_delta_percent(recall_delta, digits=2)}",
                f"FP delta: {_format_delta_percent(fp_delta, digits=2)}",
            ],
        ),
        (
            f"{val_metrics.name} split",
            [
                f"samples: {val_metrics.sample_count}",
                f"accuracy: {_format_percent(val_metrics.accuracy)}",
                f"f1: {_format_percent(val_metrics.f1)}",
                f"fp rate: {_format_percent(val_metrics.false_positive_rate)}",
                f"fn rate: {_format_percent(val_metrics.false_negative_rate)}",
            ],
        ),
        (
            f"{test_metrics.name} split",
            [
                f"samples: {test_metrics.sample_count}",
                f"accuracy: {_format_percent(test_metrics.accuracy)}",
                f"f1: {_format_percent(test_metrics.f1)}",
                f"fp rate: {_format_percent(test_metrics.false_positive_rate)}",
                f"fn rate: {_format_percent(test_metrics.false_negative_rate)}",
            ],
        ),
    ]

    width = 0.305
    gap = 0.035
    for index, (title, lines) in enumerate(blocks):
        x = index * (width + gap)
        patch = FancyBboxPatch(
            (x, 0.08),
            width,
            0.78,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=1.1,
            edgecolor="#d7c5a7",
            facecolor="#fffaf0",
        )
        ax.add_patch(patch)
        _add_clipped_text(ax, patch, x + 0.03, 0.80, textwrap.shorten(title, width=18, placeholder="..."), fontsize=10.8, weight="bold", color="#17324d", va="top")
        current_y = 0.67
        for line in lines:
            compact_line = textwrap.shorten(line, width=28, placeholder="...")
            _add_clipped_text(ax, patch, x + 0.03, current_y, compact_line, fontsize=8.9, color="#4b5d70", va="top")
            current_y -= 0.12


def _render_acceptance(ax: Any, acceptance: dict[str, Any], operating_point: dict[str, Any]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 1.0, "Acceptance criteria and guardrails", fontsize=14, weight="bold", color="#17324d", va="top")

    criteria_satisfied = bool(acceptance.get("criteria_satisfied", False))
    header_color = "#1f7a4c" if criteria_satisfied else "#a63c3c"
    summary_text = (
        f"criteria_satisfied={criteria_satisfied} | "
        f"constraints_satisfied={bool(operating_point.get('constraints_satisfied', False))}"
    )
    ax.text(0.0, 0.86, summary_text, fontsize=11, color=header_color, va="top", weight="bold")

    criteria_results = acceptance.get("criteria_results", [])
    shown_results = criteria_results[:5]
    if not shown_results:
        ax.text(0.0, 0.70, "No acceptance criteria found in the report.", fontsize=10, color="#4b5d70", va="top")
        return

    row_height = 0.15
    for index, criterion in enumerate(shown_results):
        y = 0.73 - index * row_height
        status = str(criterion.get("status", "unknown"))
        status_color = {
            "passed": "#1f7a4c",
            "failed": "#a63c3c",
        }.get(status, "#8a6d1f")
        background = {
            "passed": "#eef8f2",
            "failed": "#fff0f0",
        }.get(status, "#fff8e8")
        patch = FancyBboxPatch(
            (0.0, y - 0.112),
            0.98,
            0.118,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=0.8,
            edgecolor="#e0d3bc",
            facecolor=background,
        )
        ax.add_patch(patch)
        observed_value = criterion.get("observed_value")
        if isinstance(observed_value, bool):
            observed_label = str(observed_value).lower()
        elif isinstance(observed_value, (int, float)):
            observed_label = _format_percent(float(observed_value)) if 0.0 <= float(observed_value) <= 1.0 else _format_float(float(observed_value))
        else:
            observed_label = "n/a" if observed_value is None else str(observed_value)
        compact_title = textwrap.shorten(_criterion_label(criterion.get("name", "unnamed")), width=30, placeholder="...")
        compact_observed = textwrap.shorten(f"obs={observed_label}", width=18, placeholder="...")
        _add_clipped_text(ax, patch, 0.02, y - 0.016, compact_title, fontsize=8.9, weight="bold", color="#17324d", va="top")
        _add_clipped_text(ax, patch, 0.02, y - 0.064, compact_observed, fontsize=8.0, color="#4b5d70", va="top")
        _add_clipped_text(ax, patch, 0.83, y - 0.024, _status_label(status), fontsize=9.2, weight="bold", color=status_color, va="top", ha="left")


def generate_metrics_snapshot(report_path: str | Path = DEFAULT_REPORT_PATH, output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    resolved_report_path = Path(report_path).resolve()
    resolved_output_path = Path(output_path).resolve()
    report = _load_json(resolved_report_path)

    operating_point = report.get("operating_point", {})
    acceptance = report.get("acceptance_criteria", {})
    val_metrics = _split_metrics("val", report)
    test_metrics = _split_metrics("test", report)

    figure = plt.figure(figsize=(16, 10), facecolor="#f5f1e8")
    grid = figure.add_gridspec(3, 4, height_ratios=[1.1, 2.2, 1.8], hspace=0.34, wspace=0.28)

    header_ax = figure.add_subplot(grid[0, :])
    metrics_ax = figure.add_subplot(grid[1, 0:2])
    val_ax = figure.add_subplot(grid[1, 2])
    test_ax = figure.add_subplot(grid[1, 3])
    summary_ax = figure.add_subplot(grid[2, 0:2])
    acceptance_ax = figure.add_subplot(grid[2, 2:4])

    _render_header(header_ax, report, operating_point, acceptance)
    _render_metric_chart(metrics_ax, val_metrics, test_metrics)
    _render_confusion_matrix(val_ax, val_metrics, accent="#1f6f8b")
    _render_confusion_matrix(test_ax, test_metrics, accent="#f4a261", show_y_labels=False)
    _render_threshold_summary(summary_ax, operating_point, val_metrics, test_metrics)
    _render_acceptance(acceptance_ax, acceptance, operating_point)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output_path, dpi=220, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return resolved_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pict03.png from models/evaluation_report.json.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Path to evaluation_report.json.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output PNG path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = generate_metrics_snapshot(args.report, args.out)
    print(f"metrics_image={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())