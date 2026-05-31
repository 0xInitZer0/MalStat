"""Import and export IoC extraction."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.extractors.base import BaseExtractor

if TYPE_CHECKING:
    from src.core.indicator_registry import IndicatorRegistry


class ImportsFeaturesExtractor(BaseExtractor):
    """Extracts import/export counts and config-driven DLL/API indicators."""

    _KERNEL_MODE_DLLS = {
        "acpi.sys",
        "classpnp.sys",
        "fltmgr.sys",
        "fwpkclnt.sys",
        "hal.dll",
        "hidclass.sys",
        "ksecdd.sys",
        "ks.sys",
        "ndis.sys",
        "netio.sys",
        "ntoskrnl.exe",
        "portcls.sys",
        "scsiport.sys",
        "storport.sys",
        "tcpip.sys",
        "usbd.sys",
        "usbport.sys",
        "wdfldr.sys",
        "wmilib.sys",
    }

    def __init__(self, registry: "IndicatorRegistry") -> None:
        super().__init__(registry)
        self._dll_feature_names = {
            rule["normalized"]: f"has_{self._registry.make_feature_token(rule.get('name'))}"
            for rule in self._registry.suspicious_dlls
        }
        self._api_feature_map: list[tuple[dict[str, Any], str]] = []
        dynamic_api_features: set[str] = set()
        for category, rules in self._registry.api_categories.items():
            for rule in rules:
                token = self._registry.make_feature_token(rule.get("name") or rule.get("pattern"))
                if rule.get("match_type") in ("exact", "prefix"):
                    feature_name = f"has_api_{token}"
                else:
                    feature_name = f"api_indicator_{token}_count"
                self._api_feature_map.append((rule, feature_name))
                dynamic_api_features.add(feature_name)

        category_features = {f"api_{category}_count" for category in self._registry.api_categories}
        self._feature_names = [
            "imports_dll_count",
            "imports_func_count",
            "exports_func_count",
            "imports_by_ordinal_count",
            "ratio_ordinal_imports",
            "suspicious_dll_count",
            "imports_kernel_mode_dll_count",
            "has_dynamic_import_pattern",
            "has_ntoskrnl_import",
            "has_ndis_import",
            "has_only_kernel_mode_dll_imports",
            "api_process_count",
            *sorted(self._dll_feature_names.values()),
            *sorted(category_features),
            *sorted(dynamic_api_features),
        ]

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def extract(
        self,
        pe_bytes: bytes,
        lief_pe: Any,
        file_path: str | Path | None = None,
    ) -> dict[str, Any]:
        try:
            result = self._defaults()
            if lief_pe is None:
                return result

            import_entries = list(getattr(lief_pe, "DIRECTORY_ENTRY_IMPORT", []) or [])
            export_entry = getattr(lief_pe, "DIRECTORY_ENTRY_EXPORT", None)

            imported_dlls: set[str] = set()
            imported_names: set[str] = set()
            imports_func_count = 0
            ordinal_count = 0

            for entry in import_entries:
                dll_name = (getattr(entry, "dll", b"") or b"").decode("utf-8", errors="ignore")
                imported_dlls.add(self._registry.normalize_token(dll_name))
                for imported_symbol in getattr(entry, "imports", []) or []:
                    imports_func_count += 1
                    name = getattr(imported_symbol, "name", None)
                    if name is None:
                        ordinal_count += 1
                        continue
                    imported_names.add(name.decode("utf-8", errors="ignore"))

            result["imports_dll_count"] = len(imported_dlls)
            result["imports_func_count"] = imports_func_count
            result["exports_func_count"] = len(getattr(export_entry, "symbols", []) or []) if export_entry is not None else 0
            result["imports_by_ordinal_count"] = ordinal_count
            result["ratio_ordinal_imports"] = round(ordinal_count / max(imports_func_count, 1), 4)
            kernel_mode_dll_count = len(imported_dlls & self._KERNEL_MODE_DLLS)
            result["imports_kernel_mode_dll_count"] = kernel_mode_dll_count
            result["has_ntoskrnl_import"] = "ntoskrnl.exe" in imported_dlls
            result["has_ndis_import"] = "ndis.sys" in imported_dlls
            result["has_only_kernel_mode_dll_imports"] = bool(imported_dlls) and imported_dlls.issubset(
                self._KERNEL_MODE_DLLS
            )

            suspicious_dll_count = 0
            for dll_normalized, feature_name in self._dll_feature_names.items():
                matched = dll_normalized in imported_dlls
                result[feature_name] = matched
                suspicious_dll_count += int(matched)
            result["suspicious_dll_count"] = suspicious_dll_count

            category_hits: dict[str, set[str]] = {
                category: set() for category in self._registry.api_categories
            }
            pattern_feature_counts: Counter[str] = Counter()
            exact_feature_hits: set[str] = set()

            for import_name in imported_names:
                for rule, feature_name in self._api_feature_map:
                    if not self._registry.rule_matches(import_name, rule):
                        continue
                    category = str(rule.get("category", "other"))
                    if bool(rule.get("count_in_category", True)):
                        category_hits.setdefault(category, set()).add(
                            self._registry.normalize_token(import_name)
                        )
                    if rule.get("match_type") in ("exact", "prefix"):
                        exact_feature_hits.add(feature_name)
                    else:
                        pattern_feature_counts[feature_name] += 1

            for category in self._registry.api_categories:
                result[f"api_{category}_count"] = len(category_hits.get(category, set()))
            process_hits = category_hits.get("injection", set()) | category_hits.get("enumeration", set())
            result["api_process_count"] = len(process_hits)

            for _, feature_name in self._api_feature_map:
                if feature_name.startswith("has_api_"):
                    result[feature_name] = feature_name in exact_feature_hits
                else:
                    result[feature_name] = pattern_feature_counts.get(feature_name, 0)

            result["has_dynamic_import_pattern"] = bool(
                result.get("has_api_loadlibrary", False)
                and result.get("has_api_getprocaddress", False)
                and imports_func_count <= 6
            )
            return result
        except Exception:
            return self._defaults()

    def _defaults(self) -> dict[str, Any]:
        defaults = {name: 0 for name in self._feature_names}
        for name in self._feature_names:
            if name.startswith("has_"):
                defaults[name] = False
        defaults["ratio_ordinal_imports"] = 0.0
        return defaults
