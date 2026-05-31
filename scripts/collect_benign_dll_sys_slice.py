from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.extraction_support import get_authenticode_signature_info


DEFAULT_WINDOWS_ROOT = Path(os.environ.get("WINDIR", r"C:\Windows"))


@dataclass(frozen=True)
class SliceArtifactPaths:
    output_root: Path
    files_root: Path
    manifest_path: Path
    summary_path: Path


def _normalize_signature_mode(value: str | None) -> str:
    normalized = str(value or "any").strip().casefold()
    if normalized not in {"any", "signed", "unsigned"}:
        raise ValueError(f"Unsupported driver signature mode: {value!r}")
    return normalized


def _driver_matches_signature_mode(path: Path, signature_mode: str) -> bool:
    normalized_mode = _normalize_signature_mode(signature_mode)
    if normalized_mode == "any":
        return True

    signature_info = get_authenticode_signature_info(path)
    has_signature = bool(signature_info.get("is_signed", False))
    if normalized_mode == "signed":
        return has_signature
    return not has_signature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a labeled benign DLL/SYS evaluation slice from local Windows system files."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "benign_dll_sys",
        help="Where to write files/, manifest.csv and slice_summary.json.",
    )
    parser.add_argument(
        "--dll-dir",
        type=Path,
        default=DEFAULT_WINDOWS_ROOT / "System32",
        help="Directory to scan for benign DLL files.",
    )
    parser.add_argument(
        "--driver-dir",
        type=Path,
        default=DEFAULT_WINDOWS_ROOT / "System32" / "drivers",
        help="Directory to scan for benign SYS driver files.",
    )
    parser.add_argument("--max-dlls", type=int, default=50, help="Maximum number of DLL files to copy.")
    parser.add_argument("--max-drivers", type=int, default=25, help="Maximum number of SYS files to copy.")
    parser.add_argument("--skip-first-dlls", type=int, default=0, help="Skip the first N smallest DLL candidates before copying.")
    parser.add_argument("--skip-first-drivers", type=int, default=0, help="Skip the first N smallest SYS candidates before copying.")
    parser.add_argument(
        "--driver-signature-mode",
        choices=["any", "signed", "unsigned"],
        default="any",
        help="Optional filter for SYS driver candidates based on whether the PE file carries a security directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Clear any existing files under output-root/files before collecting.")
    return parser.parse_args()


def _collect_candidates(
    root: Path,
    suffix: str,
    max_count: int,
    *,
    skip_first: int = 0,
    candidate_filter: Callable[[Path], bool] | None = None,
) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    candidates = sorted(
        (path.resolve() for path in root.glob(f"*.{suffix}") if path.is_file()),
        key=lambda item: (item.stat().st_size, item.name.casefold()),
    )
    if candidate_filter is not None:
        candidates = [path for path in candidates if candidate_filter(path)]
    start_index = max(0, int(skip_first or 0))
    end_index = start_index + max(0, max_count)
    return candidates[start_index:end_index]


def _maybe_clear(files_root: Path) -> None:
    if files_root.exists():
        shutil.rmtree(files_root)


def _has_existing_files(files_root: Path) -> bool:
    if not files_root.exists():
        return False
    return any(path.is_file() for path in files_root.rglob("*"))


def _copy_group(
    files: list[Path],
    *,
    destination_root: Path,
    pe_kind: str,
    source: str,
    source_family: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    destination_root.mkdir(parents=True, exist_ok=True)

    for source_path in files:
        destination_path = destination_root / source_path.name
        shutil.copy2(source_path, destination_path)
        relative_path = destination_path.relative_to(destination_root.parent).as_posix()
        rows.append(
            {
                "relative_path": relative_path,
                "original_name": source_path.name,
                "label": "0",
                "label_confidence": "high",
                "pe_kind": pe_kind,
                "source": source,
                "source_family": source_family,
                "source_path": str(source_path),
                "copied_path": str(destination_path.resolve()),
            }
        )
    return rows


def collect_benign_slice(
    *,
    output_root: str | Path,
    dll_dir: str | Path,
    driver_dir: str | Path,
    max_dlls: int,
    max_drivers: int,
    skip_first_dlls: int = 0,
    skip_first_drivers: int = 0,
    driver_signature_mode: str = "any",
    overwrite: bool,
) -> SliceArtifactPaths:
    resolved_output_root = Path(output_root).resolve()
    files_root = resolved_output_root / "files"
    manifest_path = resolved_output_root / "manifest.csv"
    summary_path = resolved_output_root / "slice_summary.json"
    resolved_driver_signature_mode = _normalize_signature_mode(driver_signature_mode)

    if overwrite:
        _maybe_clear(files_root)
    elif _has_existing_files(files_root):
        raise FileExistsError(
            f"Output root already contains collected files: {files_root}. "
            "Use --overwrite or choose a different --output-root."
        )
    files_root.mkdir(parents=True, exist_ok=True)

    dll_files = _collect_candidates(Path(dll_dir).resolve(), "dll", max_dlls, skip_first=skip_first_dlls)
    driver_files = _collect_candidates(
        Path(driver_dir).resolve(),
        "sys",
        max_drivers,
        skip_first=skip_first_drivers,
        candidate_filter=lambda path: _driver_matches_signature_mode(path, resolved_driver_signature_mode),
    )

    manifest_rows = []
    manifest_rows.extend(
        _copy_group(
            dll_files,
            destination_root=files_root / "dll",
            pe_kind="dll",
            source="trusted_local_windows",
            source_family="microsoft_windows_system32",
        )
    )
    manifest_rows.extend(
        _copy_group(
            driver_files,
            destination_root=files_root / "driver",
            pe_kind="driver",
            source="trusted_local_windows",
            source_family="microsoft_windows_drivers",
        )
    )

    resolved_output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "original_name",
                "label",
                "label_confidence",
                "pe_kind",
                "source",
                "source_family",
                "source_path",
                "copied_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_payload = {
        "output_root": str(resolved_output_root),
        "files_root": str(files_root),
        "manifest_path": str(manifest_path),
        "dll_source_dir": str(Path(dll_dir).resolve()),
        "driver_source_dir": str(Path(driver_dir).resolve()),
        "skip_first_dlls": int(skip_first_dlls),
        "skip_first_drivers": int(skip_first_drivers),
        "driver_signature_mode": resolved_driver_signature_mode,
        "dll_count": len(dll_files),
        "driver_count": len(driver_files),
        "total_count": len(manifest_rows),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return SliceArtifactPaths(
        output_root=resolved_output_root,
        files_root=files_root,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )


def main() -> int:
    args = parse_args()
    result = collect_benign_slice(
        output_root=args.output_root,
        dll_dir=args.dll_dir,
        driver_dir=args.driver_dir,
        max_dlls=args.max_dlls,
        max_drivers=args.max_drivers,
        skip_first_dlls=args.skip_first_dlls,
        skip_first_drivers=args.skip_first_drivers,
        driver_signature_mode=args.driver_signature_mode,
        overwrite=args.overwrite,
    )

    print(f"output_root={result.output_root}")
    print(f"files_root={result.files_root}")
    print(f"manifest_path={result.manifest_path}")
    print(f"summary_path={result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())