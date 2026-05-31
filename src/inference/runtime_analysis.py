"""Reusable helpers for single-file runtime analysis and artifact writing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.core.config import Config
from src.reporting import HtmlRenderer
from src.reporting import ReportBuilder
from src.reporting.models import AnalysisReport


@dataclass(frozen=True)
class AnalysisOutputPaths:
    json_report_path: Path
    html_report_path: Path | None


@dataclass(frozen=True)
class AnalysisRunResult:
    input_path: Path
    report: AnalysisReport
    payload: dict[str, Any]
    output_paths: AnalysisOutputPaths


def default_analysis_output_paths(
    config: Config,
    input_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> tuple[Path, Path]:
    resolved_input_path = Path(input_path)
    report_root = Path(output_root).resolve() if output_root is not None else Path(config.reports_dir) / "analysis"
    return (
        report_root / f"{resolved_input_path.stem}.analysis.json",
        report_root / f"{resolved_input_path.stem}.analysis.html",
    )


def build_analysis_payload(report: AnalysisReport) -> dict[str, Any]:
    payload = report.to_dict()
    payload["model_version"] = report.model_version
    payload["raw_feature_count"] = len(report.raw_features)
    payload["feature_verdict_count"] = len(report.feature_verdicts)
    payload["pe_invalid_reason"] = report.pe_invalid_reason
    return payload


def _build_html_renderer(config: Config) -> HtmlRenderer:
    try:
        return HtmlRenderer(config.template_path)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "HTML report rendering requires jinja2. Install requirements.txt or rerun with --no-html."
        ) from exc


def analyze_file_path(
    input_path: str | Path,
    *,
    config: Config,
    builder: ReportBuilder | None = None,
    output_root: str | Path | None = None,
    json_out: str | Path | None = None,
    html_out: str | Path | None = None,
    write_html: bool = True,
) -> AnalysisRunResult:
    resolved_input_path = Path(input_path).resolve()
    active_builder = builder or ReportBuilder.from_config(config)

    default_json_out, default_html_out = default_analysis_output_paths(
        config,
        resolved_input_path,
        output_root=output_root,
    )
    resolved_json_out = Path(json_out).resolve() if json_out is not None else default_json_out.resolve()
    resolved_html_out = Path(html_out).resolve() if html_out is not None else default_html_out.resolve()

    report = active_builder.analyze(resolved_input_path)
    payload = build_analysis_payload(report)

    resolved_json_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    html_report_path: Path | None = None
    if write_html:
        renderer = _build_html_renderer(config)
        html_report_path = renderer.render_to_file(report, resolved_html_out).resolve()

    return AnalysisRunResult(
        input_path=resolved_input_path,
        report=report,
        payload=payload,
        output_paths=AnalysisOutputPaths(
            json_report_path=resolved_json_out,
            html_report_path=html_report_path,
        ),
    )