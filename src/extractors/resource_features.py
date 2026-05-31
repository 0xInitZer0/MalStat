"""PE resource-based features."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.extractors.base import BaseExtractor

if TYPE_CHECKING:
    from src.core.indicator_registry import IndicatorRegistry


class ResourceFeaturesExtractor(BaseExtractor):
    """Extracts resource indicators and basic embedded-payload heuristics."""

    RT_ICON = 3
    RT_GROUP_ICON = 14
    RT_VERSION = 16
    RT_MANIFEST = 24

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
            if lief_pe is None or not hasattr(lief_pe, "DIRECTORY_ENTRY_RESOURCE"):
                return self._defaults()

            root = lief_pe.DIRECTORY_ENTRY_RESOURCE
            type_count = len(getattr(root, "entries", []) or [])
            leaf_count = 0
            max_entropy = 0.0
            has_version = False
            has_manifest = False
            has_icon = False
            has_embedded_pe = False

            for type_entry in getattr(root, "entries", []) or []:
                type_id = int(getattr(type_entry, "id", 0) or 0)
                has_version = has_version or type_id == self.RT_VERSION
                has_manifest = has_manifest or type_id == self.RT_MANIFEST
                has_icon = has_icon or type_id in {self.RT_ICON, self.RT_GROUP_ICON}
                for payload in self._iter_resource_payloads(lief_pe, type_entry):
                    leaf_count += 1
                    max_entropy = max(max_entropy, self._safe_entropy(payload))
                    if b"MZ" in payload[:4096] or b"PE\x00\x00" in payload[:4096]:
                        has_embedded_pe = True

            return {
                "rsrc_has_version_info": has_version,
                "rsrc_has_manifest": has_manifest,
                "rsrc_has_icon": has_icon,
                "rsrc_has_embedded_pe": has_embedded_pe,
                "rsrc_max_entropy": round(max_entropy, 4),
                "rsrc_type_count": type_count,
                "rsrc_num_entries": leaf_count,
            }
        except Exception:
            return self._defaults()

    def _defaults(self) -> dict[str, Any]:
        return {
            "rsrc_has_version_info": False,
            "rsrc_has_manifest": False,
            "rsrc_has_icon": False,
            "rsrc_has_embedded_pe": False,
            "rsrc_max_entropy": 0.0,
            "rsrc_type_count": 0,
            "rsrc_num_entries": 0,
        }

    def _iter_resource_payloads(self, pe: Any, entry: Any) -> list[bytes]:
        payloads: list[bytes] = []
        directory = getattr(entry, "directory", None)
        if directory is not None:
            for child in getattr(directory, "entries", []) or []:
                payloads.extend(self._iter_resource_payloads(pe, child))
            return payloads

        data = getattr(entry, "data", None)
        struct = getattr(data, "struct", None)
        if struct is None:
            return payloads
        rva = int(getattr(struct, "OffsetToData", 0) or 0)
        size = int(getattr(struct, "Size", 0) or 0)
        if size <= 0:
            return payloads
        try:
            payload = pe.get_data(rva, size)
        except Exception:
            return payloads
        if payload:
            payloads.append(payload)
        return payloads
