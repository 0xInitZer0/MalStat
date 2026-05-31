"""File-level PE features."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.extractors.base import BaseExtractor
from src.extractors.pe_helpers import (
    classify_pe_kind,
    derive_pe_file_flags,
    get_overlay_bytes,
    get_primary_pe_bytes,
)

if TYPE_CHECKING:
    from src.core.indicator_registry import IndicatorRegistry


class FileFeaturesExtractor(BaseExtractor):
    """Extracts basic file-level statistics independent of PE directories."""

    def __init__(self, registry: "IndicatorRegistry") -> None:
        super().__init__(registry)

    @property
    def feature_names(self) -> list[str]:
        return list(self._defaults().keys())

    def extract(
        self,
        pe_bytes: bytes,
        lief_pe: Any,
        file_path: str | Path | None = None,
    ) -> dict[str, Any]:
        try:
            file_size = len(pe_bytes)
            characteristics = 0
            subsystem = 0
            if lief_pe is not None:
                characteristics = int(getattr(lief_pe.FILE_HEADER, "Characteristics", 0) or 0)
                optional_header = getattr(lief_pe, "OPTIONAL_HEADER", None)
                subsystem = int(getattr(optional_header, "Subsystem", 0) or 0)
            overlay = get_overlay_bytes(pe_bytes, lief_pe)
            primary_bytes = get_primary_pe_bytes(pe_bytes, lief_pe)
            printable = sum(1 for byte in primary_bytes if 32 <= byte <= 126)
            flags = derive_pe_file_flags(characteristics, subsystem, file_path=file_path)
            pe_kind = classify_pe_kind(
                characteristics,
                subsystem,
                is_pe_valid=lief_pe is not None,
                file_path=file_path,
            )

            return {
                "file_size": file_size,
                "is_executable_image": flags["is_executable_image"],
                "is_exe": pe_kind == "exe",
                "is_dll": flags["is_dll"],
                "is_driver": flags["is_driver"],
                "overlay_size": len(overlay),
                "overlay_entropy": round(self._safe_entropy(overlay), 4),
                "file_entropy": round(self._safe_entropy(primary_bytes), 4),
                "ratio_printable": round(printable / max(len(primary_bytes), 1), 4),
            }
        except Exception:
            return self._defaults()

    def _defaults(self) -> dict[str, Any]:
        return {
            "file_size": 0,
            "is_executable_image": False,
            "is_exe": False,
            "is_dll": False,
            "is_driver": False,
            "overlay_size": 0,
            "overlay_entropy": 0.0,
            "file_entropy": 0.0,
            "ratio_printable": 0.0,
        }
