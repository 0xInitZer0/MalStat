"""PE section-based indicators and heuristics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.extractors.base import BaseExtractor
from src.extractors.pe_helpers import decode_section_name, get_section_bytes, section_flags

if TYPE_CHECKING:
    from src.core.indicator_registry import IndicatorRegistry


class SectionFeaturesExtractor(BaseExtractor):
    """Extracts section statistics and packer-name indicators."""

    _TEXT_NAMES = {".text", ".code", "code", "text"}
    _DATA_NAMES = {".data", "data"}
    _RDATA_NAMES = {".rdata", "rdata"}
    _DRIVER_SECTION_PREFIXES = ("init", "page")

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
            if lief_pe is None:
                return self._defaults()

            sections = list(getattr(lief_pe, "sections", []))
            names = [decode_section_name(section) for section in sections]
            normalized_names = {self._registry.normalize_token(name) for name in names}
            entropies: list[float] = []
            virt_raw_ratios: list[float] = []
            executable_count = 0
            wx_count = 0
            zero_rawsize_count = 0
            text_raw_size = 0
            text_entropy = 0.0
            rsrc_size = 0

            packer_hits = {
                "has_upx_section": False,
                "has_vmp_section": False,
                "has_themida_section": False,
                "has_mpress_section": False,
                "has_aspack_section": False,
                "has_petite_section": False,
                "has_nsp_section": False,
                "has_enigma_section": False,
                "has_obsidium_section": False,
            }

            for section in sections:
                name = decode_section_name(section)
                name_normalized = self._registry.normalize_token(name)
                raw = get_section_bytes(pe_bytes, section)
                entropy = self._safe_entropy(raw)
                entropies.append(entropy)

                raw_size = int(getattr(section, "SizeOfRawData", 0) or 0)
                virt_size = int(getattr(section, "Misc_VirtualSize", 0) or 0)
                virt_raw_ratios.append(virt_size / max(raw_size, 1))
                if raw_size == 0 and virt_size > 0:
                    zero_rawsize_count += 1

                is_executable, _, is_writable = section_flags(section)
                executable_count += int(is_executable)
                wx_count += int(is_executable and is_writable)

                if name_normalized in {self._registry.normalize_token(item) for item in self._TEXT_NAMES}:
                    text_raw_size = raw_size
                    text_entropy = entropy
                if name_normalized == ".rsrc":
                    rsrc_size = raw_size

                for rule in self._registry.section_indicators:
                    if self._registry.rule_matches(name, rule):
                        packer_name = self._registry.normalize_token(rule.get("packer"))
                        self._apply_packer_flag(packer_hits, packer_name)

            sec_entropy_mean = sum(entropies) / len(entropies) if entropies else 0.0
            sec_entropy_max = max(entropies) if entropies else 0.0
            sec_virt_raw_ratio_max = max(virt_raw_ratios) if virt_raw_ratios else 0.0
            has_no_text = not bool(normalized_names & {self._registry.normalize_token(item) for item in self._TEXT_NAMES})
            has_no_data = not bool(normalized_names & {self._registry.normalize_token(item) for item in self._DATA_NAMES})
            has_no_rdata = not bool(normalized_names & {self._registry.normalize_token(item) for item in self._RDATA_NAMES})
            has_init_section = any(name.startswith("init") for name in normalized_names)
            has_page_section = any(name.startswith("page") for name in normalized_names)
            driver_named_section_count = sum(
                1
                for name in normalized_names
                if any(name.startswith(prefix) for prefix in self._DRIVER_SECTION_PREFIXES)
            )

            result = {
                "sec_count": len(sections),
                "sec_entropy_mean": round(sec_entropy_mean, 4),
                "sec_entropy_max": round(sec_entropy_max, 4),
                "sec_executable_count": executable_count,
                "sec_wx_count": wx_count,
                "sec_zero_rawsize_count": zero_rawsize_count,
                "sec_virt_raw_ratio_max": round(sec_virt_raw_ratio_max, 4),
                "has_no_text_section": has_no_text,
                "has_missing_text_section": has_no_text,
                "has_no_data_section": has_no_data,
                "has_no_rdata_section": has_no_rdata,
                "has_page_section": has_page_section,
                "has_init_section": has_init_section,
                "driver_named_section_count": driver_named_section_count,
                "has_driver_layout_sections": has_page_section and has_init_section,
                "has_only_one_section": len(sections) == 1,
                "rsrc_size": rsrc_size,
                "text_raw_size": text_raw_size,
                "text_entropy": round(text_entropy, 4),
                "has_packer_section": any(packer_hits.values()),
            }
            result.update(packer_hits)
            return result
        except Exception:
            return self._defaults()

    def _defaults(self) -> dict[str, Any]:
        return {
            "sec_count": 0,
            "sec_entropy_mean": 0.0,
            "sec_entropy_max": 0.0,
            "sec_executable_count": 0,
            "sec_wx_count": 0,
            "sec_zero_rawsize_count": 0,
            "sec_virt_raw_ratio_max": 0.0,
            "has_upx_section": False,
            "has_vmp_section": False,
            "has_themida_section": False,
            "has_mpress_section": False,
            "has_aspack_section": False,
            "has_petite_section": False,
            "has_nsp_section": False,
            "has_enigma_section": False,
            "has_obsidium_section": False,
            "has_no_text_section": False,
            "has_missing_text_section": False,
            "has_no_data_section": False,
            "has_no_rdata_section": False,
            "has_page_section": False,
            "has_init_section": False,
            "driver_named_section_count": 0,
            "has_driver_layout_sections": False,
            "has_only_one_section": False,
            "rsrc_size": 0,
            "text_raw_size": 0,
            "text_entropy": 0.0,
            "has_packer_section": False,
        }

    @staticmethod
    def _apply_packer_flag(packer_hits: dict[str, bool], packer_name: str) -> None:
        mapping = {
            "upx": "has_upx_section",
            "vmprotect": "has_vmp_section",
            "themida": "has_themida_section",
            "winlicense": "has_themida_section",
            "mpress": "has_mpress_section",
            "aspack": "has_aspack_section",
            "petite": "has_petite_section",
            "nspack": "has_nsp_section",
            "enigma": "has_enigma_section",
            "obsidium": "has_obsidium_section",
        }
        feature_name = mapping.get(packer_name)
        if feature_name is not None:
            packer_hits[feature_name] = True
