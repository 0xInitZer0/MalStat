"""VirusTotal cloud reputation extractor.

Performs a hash-only lookup (no file upload) against the VirusTotal API v3.
The lookup path is shared with metadata enrichment via VirusTotalLookupService
so one extraction reuses the same API key loading and in-process cache.

Features returned
-----------------
vt_found               : 1 if the hash was found in VT, else 0
vt_positives           : number of AV engines flagging the file as malicious
vt_suspicious          : number of engines flagging as suspicious
vt_total               : total number of engines that evaluated the file
vt_detection_ratio     : vt_positives / vt_total  (0.0 if not found / no engines)
vt_reputation          : VirusTotal community reputation score (signed int)
vt_first_seen_days_ago : days elapsed since first submission (None if not found)
vt_times_submitted     : how many times the file has been submitted to VT
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.core.extraction_support import VirusTotalLookupService
from src.extractors.base import BaseExtractor

if TYPE_CHECKING:
    from src.core.indicator_registry import IndicatorRegistry


class VirusTotalExtractor(BaseExtractor):
    """Fetches file reputation data from VirusTotal using the file's SHA-256 hash.

    Args:
        registry:  Shared IndicatorRegistry (required by BaseExtractor).
        api_key:   VirusTotal API key.  If *None*, falls back to the
                   ``VT_API_KEY`` environment variable (loaded from .env).
    """

    def __init__(
        self,
        registry: "IndicatorRegistry",
        api_key: str | None = None,
        project_root: str | Path | None = None,
        lookup_service: VirusTotalLookupService | None = None,
    ) -> None:
        super().__init__(registry)
        self._lookup_service = lookup_service or VirusTotalLookupService(
            project_root=project_root,
            api_key=api_key,
            timeout_sec=15,
        )

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        return list(self._defaults().keys())

    def _defaults(self) -> dict[str, Any]:
        return {
            "vt_found":               0,
            "vt_positives":           0,
            "vt_suspicious":          0,
            "vt_total":               0,
            "vt_detection_ratio":     0.0,
            "vt_reputation":          0,
            "vt_first_seen_days_ago": None,
            "vt_times_submitted":     0,
        }

    def extract(
        self,
        pe_bytes: bytes,
        lief_pe: Any,
        file_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Look up the file hash in VirusTotal and return reputation features.

        Always returns a complete feature dict; never raises an exception.
        Returns ``_defaults()`` when the API key is missing, the file is not
        found on VT, or any network / parse error occurs.
        """
        if not self._lookup_service.enabled:
            return self._defaults()
        try:
            sha256 = hashlib.sha256(pe_bytes).hexdigest()
            attrs = self._lookup_service.lookup_attributes(sha256)
            if not attrs:
                return self._defaults()
            return self._parse_attributes(attrs)
        except Exception:
            return self._defaults()

    def _parse_attributes(self, attrs: dict) -> dict[str, Any]:
        stats = attrs.get("last_analysis_stats", {})

        malicious  = int(stats.get("malicious",  0))
        suspicious = int(stats.get("suspicious", 0))
        undetected = int(stats.get("undetected", 0))
        harmless   = int(stats.get("harmless",   0))
        timeout    = int(stats.get("timeout",    0))
        total = malicious + suspicious + undetected + harmless + timeout

        detection_ratio = round(malicious / total, 4) if total > 0 else 0.0

        first_seen_days: int | None = None
        first_ts = attrs.get("first_submission_date")
        if first_ts:
            now = datetime.now(tz=timezone.utc)
            first_dt = datetime.fromtimestamp(int(first_ts), tz=timezone.utc)
            first_seen_days = max(0, (now - first_dt).days)

        return {
            "vt_found":               1,
            "vt_positives":           malicious,
            "vt_suspicious":          suspicious,
            "vt_total":               total,
            "vt_detection_ratio":     detection_ratio,
            "vt_reputation":          int(attrs.get("reputation", 0)),
            "vt_first_seen_days_ago": first_seen_days,
            "vt_times_submitted":     int(attrs.get("times_submitted", 0)),
        }
