"""
Dataclasses for the reporting system.
FeatureVerdict is the verdict for a single feature.
AnalysisReport is the complete report for a single PE file.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class FeatureVerdict:
    """Verdict for one extracted feature."""

    feature_name: str
    """Feature name, e.g. 'ep_starts_with_pushad'."""

    value: Any
    """Actual feature value."""

    expected: str
    """String description of the expected value for a legitimate PE."""

    status: Literal["ok", "warning", "suspicious"]
    """Final status for this feature."""

    severity: int
    """Severity from 0 to 10 (0 = neutral, 10 = critical)."""

    explanation: str
    """Human-readable explanation of the verdict."""

    group: str
    """Feature group: entry_point / sections / imports / strings / resources / header / file_general."""

    triggered_indicator: str | None = None
    """Indicator name from the YAML config if one matched (for example, 'UPX1' or 'VirtualAlloc')."""


@dataclass
class AnalysisReport:
    """Complete analysis report for a single PE file."""

    # --- File identification ---
    filename: str
    sha256: str
    file_size: int
    analysis_timestamp: str

    # --- PE validity ---
    is_pe_valid: bool
    pe_invalid_reason: str | None = None

    # --- ML verdict ---
    probability: float = 0.0
    """Probability of belonging to the malware class, from 0.0 to 1.0."""

    verdict: Literal["malicious", "suspicious", "benign", "out_of_scope"] = "benign"
    verdict_threshold: float = 0.5

    # --- Features and verdicts ---
    raw_features: dict[str, Any] = field(default_factory=dict)
    """Flat dictionary of all features extracted by the current pipeline."""

    feature_verdicts: list[FeatureVerdict] = field(default_factory=list)
    """List of verdicts for every extracted feature."""

    top_factors: list[FeatureVerdict] = field(default_factory=list)
    """Top 15 factors by severity for the report summary section."""

    # --- ML metadata ---
    model_name: str = "unknown"
    model_version: str = "unknown"
    shap_values: dict[str, float] | None = None
    """Optional SHAP values in the form {feature_name: shap_value}."""

    def verdict_label(self) -> str:
        """Return the uppercase verdict label."""
        return {
            "malicious": "MALICIOUS",
            "suspicious": "SUSPICIOUS",
            "benign": "BENIGN",
            "out_of_scope": "OUT OF SCOPE",
        }[self.verdict]

    def verdict_css_class(self) -> str:
        """CSS class for the HTML report."""
        return {
            "malicious": "verdict-malicious",
            "suspicious": "verdict-suspicious",
            "benign": "verdict-benign",
            "out_of_scope": "verdict-out-of-scope",
        }[self.verdict]

    def verdicts_by_group(self) -> dict[str, list[FeatureVerdict]]:
        """Return verdicts grouped by the group field."""
        groups: dict[str, list[FeatureVerdict]] = {}
        for v in self.feature_verdicts:
            groups.setdefault(v.group, []).append(v)
        return groups

    def suspicious_count(self) -> int:
        return sum(1 for v in self.feature_verdicts if v.status == "suspicious")

    def warning_count(self) -> int:
        return sum(1 for v in self.feature_verdicts if v.status == "warning")

    def to_dict(self) -> dict:
        """Serialize to a dict for JSON APIs or CLI output."""
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "file_size": self.file_size,
            "analysis_timestamp": self.analysis_timestamp,
            "is_pe_valid": self.is_pe_valid,
            "pe_invalid_reason": self.pe_invalid_reason,
            "probability": round(self.probability, 4),
            "verdict": self.verdict,
            "verdict_threshold": self.verdict_threshold,
            "model_name": self.model_name,
            "suspicious_count": self.suspicious_count(),
            "warning_count": self.warning_count(),
            "top_factors": [
                {
                    "feature": f.feature_name,
                    "value": f.value,
                    "status": f.status,
                    "severity": f.severity,
                    "explanation": f.explanation,
                }
                for f in self.top_factors
            ],
        }
