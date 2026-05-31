"""
src/core/interfaces.py

Protocols (abstract interfaces) for all major system components.
They support the Dependency Inversion Principle: high-level modules depend
on these Protocol classes rather than concrete implementations.

Usage:
    from src.core.interfaces import IExtractor, IClassifierModel, IReportRenderer
"""

from __future__ import annotations
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IExtractor(Protocol):
    """Protocol for PE feature extractors."""

    def extract(self, pe_bytes: bytes, lief_pe: Any) -> dict[str, Any]:
        """
        Extract features from a PE file.

        Args:
            pe_bytes: Raw bytes of the PE file.
            lief_pe:  Parsed lief.PE object, or None if parsing failed.

        Returns:
            Flat dict {feature_name: value}.
            Never raises exceptions; returns default values on failure.
        """
        ...

    @property
    def feature_names(self) -> list[str]:
        """List of all keys that may appear in the returned dict."""
        ...


@runtime_checkable
class IClassifierModel(Protocol):
    """Protocol for an ML classifier."""

    def predict_proba(self, X: Any) -> Any:
        """
        Return class probabilities.

        Returns:
            Array shape (n_samples, n_classes).
            Index 0 = benign, index 1 = malware.
        """
        ...

    def predict(self, X: Any) -> Any:
        """Return binary predictions (0 or 1)."""
        ...


@runtime_checkable
class IReportRenderer(Protocol):
    """Protocol for report renderers."""

    def render(self, report: Any) -> str:
        """
        Render an AnalysisReport to a string (HTML, text, etc.).

        Args:
            report: AnalysisReport object.

        Returns:
            String containing the rendered result.
        """
        ...


@runtime_checkable
class IModelStore(Protocol):
    """Protocol for loading and saving models."""

    def load(self, path: str) -> IClassifierModel:
        """Load a model from a file."""
        ...

    def save(self, model: IClassifierModel, path: str) -> None:
        """Save a model to a file."""
        ...


@runtime_checkable
class IPreprocessor(Protocol):
    """Protocol for a feature preprocessor."""

    def fit(self, X: Any, y: Any | None = None) -> "IPreprocessor":
        """Fit the preprocessor on data, typically the training set."""
        ...

    def transform(self, X: Any) -> Any:
        """Apply the transformation."""
        ...

    def fit_transform(self, X: Any, y: Any | None = None) -> Any:
        """Fit and transform in one step."""
        ...
