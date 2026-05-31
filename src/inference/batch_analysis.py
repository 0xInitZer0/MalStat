"""Batch runtime analysis helpers built on top of the single-file analysis path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.core.config import Config
from src.inference.runtime_analysis import analyze_file_path
from src.reporting import ReportBuilder


DEFAULT_BATCH_PATTERNS = ("*.exe", "*.dll", "*.sys")


@dataclass(frozen=True)
class BatchAnalysisArtifactPaths:
    output_root: Path
    files_root: Path
    summary_json_path: Path
    summary_csv_path: Path


@dataclass(frozen=True)
class BatchAnalysisResult:
    input_root: Path
    processed_count: int
    failure_count: int
    summary_records: list[dict[str, Any]]
    artifact_paths: BatchAnalysisArtifactPaths


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _iter_input_files(input_root: Path, patterns: Iterable[str], recursive: bool) -> list[Path]:
    discovered: set[Path] = set()
    for pattern in patterns:
        iterator = input_root.rglob(pattern) if recursive else input_root.glob(pattern)
        for path in iterator:
            if path.is_file():
                discovered.add(path.resolve())
    return sorted(discovered)


def _resolve_output_root(config: Config, input_root: Path, output_root: str | Path | None) -> Path:
    if output_root is not None:
        return Path(output_root).resolve()
    return (Path(config.reports_dir) / "batch_analysis" / input_root.name).resolve()


def _per_file_output_paths(output_root: Path, input_root: Path, input_path: Path) -> tuple[Path, Path]:
    relative_path = input_path.relative_to(input_root)
    file_root = output_root / "files" / relative_path.parent
    return (
        file_root / f"{relative_path.stem}.analysis.json",
        file_root / f"{relative_path.stem}.analysis.html",
    )


def analyze_directory(
    input_dir: str | Path,
    *,
    config: Config,
    builder: ReportBuilder | None = None,
    patterns: Iterable[str] | None = None,
    recursive: bool = False,
    output_root: str | Path | None = None,
    write_html: bool = True,
    fail_fast: bool = False,
) -> BatchAnalysisResult:
    resolved_input_root = Path(input_dir).resolve()
    resolved_patterns = tuple(patterns or DEFAULT_BATCH_PATTERNS)
    resolved_output_root = _resolve_output_root(config, resolved_input_root, output_root)
    files_root = resolved_output_root / "files"
    files_root.mkdir(parents=True, exist_ok=True)

    input_files = _iter_input_files(resolved_input_root, resolved_patterns, recursive)
    active_builder = builder or ReportBuilder.from_config(config)
    summary_records: list[dict[str, Any]] = []

    for input_path in input_files:
        relative_path = input_path.relative_to(resolved_input_root)
        json_out, html_out = _per_file_output_paths(resolved_output_root, resolved_input_root, input_path)
        try:
            result = analyze_file_path(
                input_path,
                config=config,
                builder=active_builder,
                json_out=json_out,
                html_out=html_out,
                write_html=write_html,
            )
            summary_records.append({
                "status": "ok",
                "input_path": str(input_path),
                "relative_path": str(relative_path),
                "filename": result.report.filename,
                "sha256": result.report.sha256,
                "file_size": result.report.file_size,
                "is_pe_valid": bool(result.report.is_pe_valid),
                "pe_invalid_reason": result.report.pe_invalid_reason or "",
                "verdict": result.report.verdict,
                "probability": float(result.report.probability),
                "model_name": result.report.model_name,
                "model_version": result.report.model_version,
                "json_report_path": str(result.output_paths.json_report_path),
                "html_report_path": str(result.output_paths.html_report_path) if result.output_paths.html_report_path else "",
                "error_type": "",
                "error_message": "",
            })
        except Exception as exc:
            if fail_fast:
                raise
            summary_records.append({
                "status": "error",
                "input_path": str(input_path),
                "relative_path": str(relative_path),
                "filename": input_path.name,
                "sha256": "",
                "file_size": input_path.stat().st_size if input_path.exists() else "",
                "is_pe_valid": False,
                "pe_invalid_reason": "",
                "verdict": "",
                "probability": "",
                "model_name": "",
                "model_version": "",
                "json_report_path": str(json_out.resolve()),
                "html_report_path": str(html_out.resolve()) if write_html else "",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

    summary_json_path = resolved_output_root / "summary.json"
    summary_csv_path = resolved_output_root / "summary.csv"
    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_root": str(resolved_input_root),
        "patterns": list(resolved_patterns),
        "recursive": bool(recursive),
        "processed_count": int(sum(record["status"] == "ok" for record in summary_records)),
        "failure_count": int(sum(record["status"] != "ok" for record in summary_records)),
        "summary_records": summary_records,
    }
    summary_json_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False)

    return BatchAnalysisResult(
        input_root=resolved_input_root,
        processed_count=summary_payload["processed_count"],
        failure_count=summary_payload["failure_count"],
        summary_records=summary_records,
        artifact_paths=BatchAnalysisArtifactPaths(
            output_root=resolved_output_root,
            files_root=files_root,
            summary_json_path=summary_json_path,
            summary_csv_path=summary_csv_path,
        ),
    )