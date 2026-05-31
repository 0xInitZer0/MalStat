"""Support utilities for feature extraction and dataset row building."""

from __future__ import annotations

import csv
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable
from urllib import error, request


METADATA_FIELDNAMES = [
    "sample_id",
    "sha256",
    "md5",
    "original_name",
    "file_type",
    "pe_kind",
    "source",
    "source_family",
    "download_date",
    "file_size_bytes",
    "is_pe_valid",
    "is_signed",
    "av_detection_count",
    "label",
    "label_confidence",
]

DEPRECATED_FEATURE_ALIASES = frozenset({
    "num_import_dlls",
    "num_rich_header_entries",
    "ep_is_outside_sections",
})

BENIGN_LABEL_TOKENS = {
    "benign",
    "clean",
    "goodware",
    "legit",
    "legitimate",
    "official_release",
    "trusted",
}

MALICIOUS_LABEL_TOKENS = {
    "bot",
    "loader",
    "malicious",
    "malware",
    "ransomware",
    "stealer",
    "trojan",
    "virus",
}


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def filter_deprecated_feature_aliases(values: Iterable[str]) -> list[str]:
    return [value for value in values if value not in DEPRECATED_FEATURE_ALIASES]


def load_existing_header(csv_path: str | Path) -> list[str]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def build_feature_fieldnames(
    pipeline_feature_names: Iterable[str],
    compatibility_header: Iterable[str] | None = None,
) -> list[str]:
    base_fields = ["sample_id", "sha256", "label"]
    compatibility = filter_deprecated_feature_aliases(list(compatibility_header or []))
    pipeline_fields = filter_deprecated_feature_aliases([
        name for name in pipeline_feature_names if name not in {"sample_id", "sha256", "label"}
    ])
    return unique_preserve_order(base_fields + compatibility + pipeline_fields)


def infer_source_fields(file_path: str | Path) -> tuple[str, str]:
    path = Path(file_path)
    source_family = path.parent.name if path.parent != path else ""
    source = path.parent.parent.name if path.parent.parent != path.parent else ""
    return source, source_family


def normalize_row_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


def build_feature_row(
    feature_fieldnames: list[str],
    features: dict[str, Any],
    sha256: str,
    sample_id: int | str | None = "",
    label: int | str | None = "",
) -> dict[str, Any]:
    row = {name: "" for name in feature_fieldnames}
    row.update({
        "sample_id": sample_id if sample_id is not None else "",
        "sha256": sha256,
        "label": label if label is not None else "",
    })
    for key, value in features.items():
        row[key] = normalize_row_value(value)
    return row


def compute_hashes(pe_bytes: bytes) -> tuple[str, str]:
    return (
        hashlib.sha256(pe_bytes).hexdigest(),
        hashlib.md5(pe_bytes).hexdigest(),
    )


def today_iso() -> str:
    return date.today().isoformat()


def infer_label_from_source(
    source: str | None,
    source_family: str | None,
) -> int | None:
    normalized_values = {
        normalize_token(source),
        normalize_token(source_family),
    }
    if normalized_values & BENIGN_LABEL_TOKENS:
        return 0
    if normalized_values & MALICIOUS_LABEL_TOKENS:
        return 1
    return None


def normalize_token(value: str | None) -> str:
    return (value or "").strip().casefold()


def load_project_env_var(
    project_root: str | Path,
    key: str,
) -> str | None:
    env_value = os.getenv(key)
    if env_value:
        return env_value

    env_path = Path(project_root) / ".env"
    if not env_path.exists():
        return None

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            current_key, current_value = line.split("=", 1)
            if current_key.strip() != key:
                continue
            return current_value.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def get_authenticode_signature_info(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        return {"is_signed": False, "is_valid": False, "status": "missing"}

    escaped = str(path).replace("'", "''")
    command = (
        "$sig = Get-AuthenticodeSignature -LiteralPath '{0}'; "
        "[pscustomobject]@{{Status=[string]$sig.Status; "
        "HasSignerCertificate=($null -ne $sig.SignerCertificate)}} | ConvertTo-Json -Compress"
    ).format(escaped)
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        payload = json.loads(result.stdout.strip() or "{}")
    except Exception:
        return {"is_signed": False, "is_valid": False, "status": "unknown"}

    status = str(payload.get("Status", "Unknown"))
    has_signer = bool(payload.get("HasSignerCertificate", False))
    return {
        "is_signed": has_signer,
        "is_valid": status.casefold() == "valid",
        "status": status,
    }


def find_floss_executable(project_root: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if project_root is not None:
        root = Path(project_root)
        candidates.extend([
            root / "FLOSS" / "floss.exe",
            root / "FLOSS" / "floss",
        ])

    env_path = os.getenv("FLOSS_PATH")
    if env_path:
        candidates.append(Path(env_path))

    for name in ("floss.exe", "floss"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(Path(resolved))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class VirusTotalLookupService:
    """Shared VirusTotal hash lookup with in-process caching."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        api_key: str | None = None,
        timeout_sec: int = 20,
    ) -> None:
        self._project_root = Path(project_root).resolve() if project_root is not None else None
        if api_key is not None:
            self._api_key = api_key
        elif self._project_root is not None:
            self._api_key = load_project_env_var(self._project_root, "VT_API_KEY") or ""
        else:
            self._api_key = os.getenv("VT_API_KEY", "")
        self._timeout_sec = timeout_sec
        self._cache: dict[str, dict[str, Any] | None] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def lookup_attributes(self, sha256: str) -> dict[str, Any] | None:
        normalized = sha256.strip().casefold()
        if not normalized:
            return None
        if normalized in self._cache:
            return self._cache[normalized]
        attrs = self._fetch_attributes(normalized)
        self._cache[normalized] = attrs
        return attrs

    def resolve_detection_count(self, sha256: str) -> int | None:
        attrs = self.lookup_attributes(sha256)
        if not attrs:
            return None
        stats = attrs.get("last_analysis_stats", {})
        if not isinstance(stats, dict):
            return None
        malicious = int(stats.get("malicious", 0) or 0)
        suspicious = int(stats.get("suspicious", 0) or 0)
        return malicious + suspicious

    def _fetch_attributes(self, sha256: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        req = request.Request(
            f"https://www.virustotal.com/api/v3/files/{sha256}",
            headers={"x-apikey": self._api_key},
        )
        try:
            with request.urlopen(req, timeout=self._timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            if exc.code == 404:
                return None
            return None
        except (error.URLError, TimeoutError, json.JSONDecodeError):
            return None

        attrs = payload.get("data", {}).get("attributes", {})
        return attrs if isinstance(attrs, dict) else None


class AVDetectionResolver:
    """Resolve AV detection counts from local metadata or VirusTotal."""

    def __init__(
        self,
        project_root: str | Path,
        vt_lookup: VirusTotalLookupService | None = None,
        enable_virustotal: bool = False,
    ):
        self._project_root = Path(project_root)
        self._metadata_dir = self._project_root / "data" / "metadata"
        self._cache = self._load_local_cache()
        self._enable_virustotal = enable_virustotal
        self._vt_lookup = vt_lookup or VirusTotalLookupService(self._project_root)

    def resolve(self, sha256: str) -> int | None:
        local_value = self._cache.get(sha256.casefold())
        if local_value is not None:
            return local_value

        vt_value = self._resolve_virustotal(sha256)
        if vt_value is not None:
            self._cache[sha256.casefold()] = vt_value
        return vt_value

    def _load_local_cache(self) -> dict[str, int]:
        cache: dict[str, int] = {}
        if not self._metadata_dir.exists():
            return cache

        for path in self._metadata_dir.glob("*.csv"):
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        sha256 = str(row.get("sha256", "")).strip().casefold()
                        raw_count = str(row.get("av_detection_count", "")).strip()
                        if not sha256 or not raw_count:
                            continue
                        try:
                            cache[sha256] = int(float(raw_count))
                        except ValueError:
                            continue
            except Exception:
                continue
        return cache

    def _resolve_virustotal(self, sha256: str) -> int | None:
        if not self._enable_virustotal:
            return None
        return self._vt_lookup.resolve_detection_count(sha256)
