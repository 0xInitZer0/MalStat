"""Feature extraction pipeline for PE bytes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pefile

from src.core.extraction_support import VirusTotalLookupService
from src.extractors.file_features import FileFeaturesExtractor
from src.extractors.header_features import HeaderFeaturesExtractor
from src.extractors.imports_features import ImportsFeaturesExtractor
from src.extractors.opcode_features import OpcodeExtractor
from src.extractors.resource_features import ResourceFeaturesExtractor
from src.extractors.section_features import SectionFeaturesExtractor
from src.extractors.strings_features import StringsFeaturesExtractor
from src.extractors.virustotal_features import VirusTotalExtractor


class FeaturePipeline:
    """Runs all configured extractors over one PE sample."""

    def __init__(self, extractors: list[Any]):
        self._extractors = extractors

    def run(
        self,
        pe_bytes: bytes,
        file_path: str | Path | None = None,
    ) -> dict[str, Any]:
        parsed_pe = self._parse_pe(pe_bytes)
        features: dict[str, Any] = {}
        for extractor in self._extractors:
            features.update(extractor.extract(pe_bytes, parsed_pe, file_path=file_path))
        return features

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for extractor in self._extractors:
            names.extend(extractor.feature_names)
        return names

    @classmethod
    def default(
        cls,
        registry: Any,
        enable_floss: bool = True,
        enable_virustotal: bool = False,
        virustotal_lookup: VirusTotalLookupService | None = None,
    ) -> "FeaturePipeline":
        extractors: list[Any] = [
            FileFeaturesExtractor(registry),
            HeaderFeaturesExtractor(registry),
            SectionFeaturesExtractor(registry),
            ImportsFeaturesExtractor(registry),
            StringsFeaturesExtractor(registry, enable_floss=enable_floss),
            ResourceFeaturesExtractor(registry),
            OpcodeExtractor(registry),
        ]
        if enable_virustotal:
            extractors.append(VirusTotalExtractor(registry, lookup_service=virustotal_lookup))
        return cls(extractors)

    @staticmethod
    def _parse_pe(pe_bytes: bytes) -> Any | None:
        try:
            pe = pefile.PE(data=pe_bytes, fast_load=False)
            pe.parse_data_directories()
            return pe
        except Exception:
            return None
