"""CLI entry point for extracting PE features and metadata from one file."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import pefile

from src.core.config import Config
from src.core.extraction_support import (
    AVDetectionResolver,
    METADATA_FIELDNAMES,
    VirusTotalLookupService,
    build_feature_fieldnames,
    build_feature_row,
    compute_hashes,
    get_authenticode_signature_info,
    infer_label_from_source,
    infer_source_fields,
    load_existing_header,
    today_iso,
)
from src.core.indicator_registry import IndicatorRegistry
from src.extractors.feature_pipeline import FeaturePipeline


@dataclass(frozen=True)
class ExtractionContext:
    """Reusable extraction context for one or many files."""

    project_root: Path
    pipeline: FeaturePipeline
    vt_lookup: VirusTotalLookupService
    feature_fieldnames: list[str]
    feature_count: int
    enable_virustotal: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PE features and metadata for one file.")
    parser.add_argument("file_path", help="Absolute or relative path to the PE file.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent), help="Project root path.")
    parser.add_argument("--features-out", help="Override output CSV path for the one-row features file.")
    parser.add_argument("--metadata-out", help="Override output CSV path for the one-row metadata file.")
    parser.add_argument("--json-out", help="Optional JSON output with both rows and schema.")
    parser.add_argument("--sample-id", help="Optional sample_id override for the one-row metadata file.")
    parser.add_argument("--label", choices=["0", "1"], help="Optional class label override.")
    parser.add_argument("--label-confidence", help="Optional label confidence override.")
    parser.add_argument("--download-date", help="Optional download/collection date in YYYY-MM-DD format.")
    parser.add_argument("--source", help="Optional source override.")
    parser.add_argument("--source-family", help="Optional source_family override.")
    parser.add_argument("--av-detection-count", help="Optional AV detection count override.")
    virustotal_group = parser.add_mutually_exclusive_group()
    virustotal_group.add_argument(
        "--enable-virustotal",
        dest="enable_virustotal",
        action="store_true",
        help="Enable VirusTotal hash lookups for this run.",
    )
    virustotal_group.add_argument(
        "--skip-virustotal",
        dest="enable_virustotal",
        action="store_false",
        help="Disable VirusTotal hash lookups for this run.",
    )
    parser.set_defaults(enable_virustotal=False)
    parser.add_argument("--skip-floss", action="store_true", help="Disable FLOSS string extraction.")
    return parser.parse_args()


def build_extraction_context(
    project_root: str | Path,
    *,
    enable_virustotal: bool = False,
    skip_floss: bool = False,
) -> ExtractionContext:
    root = Path(project_root).resolve()
    config = Config().resolve(root)
    registry = IndicatorRegistry(config.indicators_dir)
    vt_lookup = VirusTotalLookupService(project_root=root)
    pipeline = FeaturePipeline.default(
        registry,
        enable_floss=not skip_floss,
        enable_virustotal=enable_virustotal,
        virustotal_lookup=vt_lookup,
    )
    compatibility_header = load_existing_header(root / "data" / "features" / "features_example.csv")
    feature_fieldnames = build_feature_fieldnames(pipeline.feature_names, compatibility_header)
    return ExtractionContext(
        project_root=root,
        pipeline=pipeline,
        vt_lookup=vt_lookup,
        feature_fieldnames=feature_fieldnames,
        feature_count=len(pipeline.feature_names),
        enable_virustotal=enable_virustotal,
    )


def extract_file_rows(
    file_path: str | Path,
    *,
    context: ExtractionContext,
    sample_id: int | str | None = "",
    label: str | None = None,
    label_confidence: str | None = None,
    download_date: str | None = None,
    source: str | None = None,
    source_family: str | None = None,
    av_detection_count: int | str | None = None,
) -> tuple[list[str], dict[str, object], dict[str, object]]:
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    pe_bytes = file_path.read_bytes()
    features = context.pipeline.run(pe_bytes, file_path=file_path)
    sha256, md5 = compute_hashes(pe_bytes)

    resolved_source, resolved_source_family = infer_source_fields(file_path)
    if source is not None:
        resolved_source = source
    if source_family is not None:
        resolved_source_family = source_family

    explicit_label = label not in (None, "")
    resolved_label = "" if label in (None, "") else str(label)
    if resolved_label == "":
        inferred_label = infer_label_from_source(resolved_source, resolved_source_family)
        resolved_label = "" if inferred_label is None else str(inferred_label)

    feature_row = build_feature_row(
        context.feature_fieldnames,
        features,
        sha256,
        sample_id=sample_id if sample_id is not None else "",
        label=resolved_label,
    )

    is_pe_valid = 1
    try:
        pefile.PE(data=pe_bytes, fast_load=True)
    except pefile.PEFormatError:
        is_pe_valid = 0

    resolved_av_detection_count = AVDetectionResolver(
        context.project_root,
        vt_lookup=context.vt_lookup,
        enable_virustotal=context.enable_virustotal,
    ).resolve(sha256)
    if av_detection_count not in (None, ""):
        resolved_av_detection_count = int(av_detection_count)

    signature_info = get_authenticode_signature_info(file_path)
    resolved_label_confidence = (
        label_confidence
        or ("manual" if explicit_label and resolved_label != "" else "path" if resolved_label != "" else "")
    )
    metadata_row = {
        "sample_id": sample_id if sample_id is not None else "",
        "sha256": sha256,
        "md5": md5,
        "original_name": file_path.name,
        "file_type": file_path.suffix.lstrip(".").casefold(),
        "pe_kind": infer_pe_kind(features, is_pe_valid=is_pe_valid),
        "source": resolved_source,
        "source_family": resolved_source_family,
        "download_date": download_date or today_iso(),
        "file_size_bytes": len(pe_bytes),
        "is_pe_valid": is_pe_valid,
        "is_signed": int(signature_info["is_signed"]),
        "av_detection_count": "" if resolved_av_detection_count is None else resolved_av_detection_count,
        "label": resolved_label,
        "label_confidence": resolved_label_confidence,
    }
    return context.feature_fieldnames, feature_row, metadata_row


def infer_pe_kind(features: dict[str, object], *, is_pe_valid: int) -> str:
    if not is_pe_valid:
        return "non_pe"
    if bool(features.get("is_dll", False)):
        return "dll"
    if bool(features.get("is_driver", False)):
        return "driver"
    if bool(features.get("is_exe", False)):
        return "exe"
    return "other_pe"


def write_extraction_outputs(
    *,
    feature_fieldnames: list[str],
    feature_row: dict[str, object],
    metadata_row: dict[str, object],
    features_out: str | Path,
    metadata_out: str | Path,
    json_out: str | Path | None = None,
) -> None:
    features_path = Path(features_out)
    metadata_path = Path(metadata_out)
    write_csv_row(features_path, feature_fieldnames, feature_row)
    write_csv_row(metadata_path, METADATA_FIELDNAMES, metadata_row)

    if json_out:
        payload = {
            "feature_fieldnames": feature_fieldnames,
            "feature_row": feature_row,
            "metadata_fieldnames": METADATA_FIELDNAMES,
            "metadata_row": metadata_row,
        }
        Path(json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    file_path = Path(args.file_path).resolve()
    if not file_path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        return 2

    context = build_extraction_context(
        project_root,
        enable_virustotal=args.enable_virustotal,
        skip_floss=args.skip_floss,
    )

    feature_fieldnames, feature_row, metadata_row = extract_file_rows(
        file_path,
        context=context,
        sample_id=args.sample_id or "",
        label=args.label,
        label_confidence=args.label_confidence,
        download_date=args.download_date,
        source=args.source,
        source_family=args.source_family,
        av_detection_count=args.av_detection_count,
    )

    features_out = Path(args.features_out) if args.features_out else project_root / "data" / "features" / f"{file_path.stem}_features_full.csv"
    metadata_out = Path(args.metadata_out) if args.metadata_out else project_root / "data" / "metadata" / f"{file_path.stem}_metadata_full.csv"
    write_extraction_outputs(
        feature_fieldnames=feature_fieldnames,
        feature_row=feature_row,
        metadata_row=metadata_row,
        features_out=features_out,
        metadata_out=metadata_out,
        json_out=args.json_out,
    )

    print(f"features_csv={features_out}")
    print(f"metadata_csv={metadata_out}")
    print(f"feature_count={context.feature_count}")
    return 0


def write_csv_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())