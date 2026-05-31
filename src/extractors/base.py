"""
src/extractors/base.py

Abstract base class for all PE extractors.
Every concrete extractor must inherit from BaseExtractor.

Contract:
    - extract() never raises exceptions and should be wrapped in try/except
    - extract() returns _defaults() on failure, a dict with neutral values
    - feature_names lists every key that the returned dict may contain
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.indicator_registry import IndicatorRegistry


class BaseExtractor(ABC):
    """
    Base class for all PE extractors.

    Args:
        registry: IndicatorRegistry with loaded YAML configs.
                  Passed through the constructor via dependency injection.
    """

    def __init__(self, registry: "IndicatorRegistry"):
        self._registry = registry

    @abstractmethod
    def extract(
        self,
        pe_bytes: bytes,
        lief_pe: Any,
        file_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Extract features from a PE file.

        Args:
            pe_bytes: Raw bytes of the PE file.
            lief_pe:  Parsed lief.PE object or None.
            file_path: Original file path, if available.

        Returns:
            Flat dict {feature_name: value}.
            On any error, returns self._defaults() and does not raise.
        """
        ...

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """
        Full list of feature names that extract() may return.
        Used for schema validation and documentation.
        """
        ...

    def _defaults(self) -> dict[str, Any]:
        """
        Default values used when parsing fails.
        By default each feature is None.
        Subclasses may override this with more meaningful defaults.
        """
        return {name: None for name in self.feature_names}

    @staticmethod
    def _safe_entropy(data: bytes) -> float:
        """
        Compute Shannon entropy for a byte sequence.

        Args:
            data: Bytes, usually section contents or the first N bytes of the entry point.

        Returns:
            Entropy in bits per byte (0.0 - 8.0). Returns 0.0 if data is empty.
        """
        import math
        if not data:
            return 0.0
        counts = [0] * 256
        for b in data:
            counts[b] += 1
        n = len(data)
        return -sum(
            (c / n) * math.log2(c / n)
            for c in counts
            if c > 0
        )
