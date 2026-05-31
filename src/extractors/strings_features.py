"""String-based IoC extraction."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, TYPE_CHECKING

from src.core.extraction_support import find_floss_executable
from src.extractors.base import BaseExtractor
from src.extractors.pe_helpers import get_primary_pe_bytes

if TYPE_CHECKING:
    from src.core.indicator_registry import IndicatorRegistry


class StringsFeaturesExtractor(BaseExtractor):
    """Extracts printable ASCII/UTF-16 strings and config-driven IoCs."""

    _MIN_LENGTH = 4
    _MAX_FLOSS_TARGET_SIZE = 0x1000000

    def __init__(
        self,
        registry: "IndicatorRegistry",
        enable_floss: bool = True,
        floss_timeout_sec: int = 120,
    ) -> None:
        super().__init__(registry)
        self._enable_floss = enable_floss
        self._floss_timeout_sec = floss_timeout_sec

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
            primary_bytes = get_primary_pe_bytes(pe_bytes, lief_pe)
            strings = self._extract_strings(primary_bytes)
            unique_strings = set(strings)
            lengths = [len(item) for item in strings]
            total_chars = sum(lengths)

            counts = {
                group: self._count_matches(strings, rules)
                for group, rules in self._registry.strings_patterns.items()
            }
            packer_marker_count = self._count_matches(strings, self._registry.packer_string_markers)
            floss_strings = self._run_floss(file_path, primary_bytes)
            floss_suspicious = self._count_matches(
                floss_strings,
                [
                    *self._all_pattern_rules(),
                    *self._registry.packer_string_markers,
                ],
            )

            return {
                "strings_count": len(strings),
                "strings_avg_len": round(total_chars / max(len(strings), 1), 4),
                "strings_max_len": max(lengths) if lengths else 0,
                "strings_unique_count": len(unique_strings),
                "strings_url_count": counts.get("url", 0),
                "strings_ip_count": counts.get("ip", 0),
                "strings_domain_count": counts.get("domain", 0),
                "strings_registry_count": counts.get("registry", 0),
                "strings_path_count": counts.get("file_path", 0),
                "strings_crypto_count": counts.get("crypto", 0),
                "strings_shell_count": counts.get("shell", 0),
                "strings_mz_count": counts.get("mz_marker", 0),
                "strings_base64_long_count": counts.get("base64", 0),
                "strings_base64_count": counts.get("base64", 0),
                "strings_hex_long_count": counts.get("hex_blob", 0),
                "strings_guid_count": counts.get("guid", 0),
                "strings_named_object_count": counts.get("mutex_pipe_service", 0),
                "strings_embedded_pe_count": counts.get("embedded_pe", 0),
                "strings_packer_marker_count": packer_marker_count,
                "floss_strings_count": len(floss_strings),
                "floss_suspicious_count": floss_suspicious,
                "ratio_printable_strings": round(total_chars / max(len(primary_bytes), 1), 4),
            }
        except Exception:
            return self._defaults()

    def _defaults(self) -> dict[str, Any]:
        return {
            "strings_count": 0,
            "strings_avg_len": 0.0,
            "strings_max_len": 0,
            "strings_unique_count": 0,
            "strings_url_count": 0,
            "strings_ip_count": 0,
            "strings_domain_count": 0,
            "strings_registry_count": 0,
            "strings_path_count": 0,
            "strings_crypto_count": 0,
            "strings_shell_count": 0,
            "strings_mz_count": 0,
            "strings_base64_long_count": 0,
            "strings_base64_count": 0,
            "strings_hex_long_count": 0,
            "strings_guid_count": 0,
            "strings_named_object_count": 0,
            "strings_embedded_pe_count": 0,
            "strings_packer_marker_count": 0,
            "floss_strings_count": 0,
            "floss_suspicious_count": 0,
            "ratio_printable_strings": 0.0,
        }

    def _run_floss(
        self,
        file_path: str | Path | None,
        primary_bytes: bytes,
    ) -> list[str]:
        if not self._enable_floss or file_path is None:
            return []

        path = Path(file_path)
        if not path.exists():
            return []

        project_root = Path(__file__).resolve().parents[2]
        floss_path = find_floss_executable(project_root)
        if floss_path is None:
            return []

        payload: dict[str, Any] | None = None
        try:
            source_size = path.stat().st_size
        except OSError:
            source_size = 0

        if source_size <= self._MAX_FLOSS_TARGET_SIZE:
            payload = self._invoke_floss(floss_path, path)

        if payload is None and 0 < len(primary_bytes) <= self._MAX_FLOSS_TARGET_SIZE:
            suffix = path.suffix or ".bin"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                temp_path = Path(handle.name)
                handle.write(primary_bytes)
            try:
                payload = self._invoke_floss(floss_path, temp_path)
            finally:
                temp_path.unlink(missing_ok=True)
        if payload is None:
            return []

        strings: list[str] = []
        for key in ("decoded_strings", "stack_strings", "tight_strings"):
            for item in payload.get(key, []) or []:
                if isinstance(item, str):
                    strings.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                value = item.get("string") or item.get("decoded_string")
                if isinstance(value, str):
                    strings.append(value)
        return list(dict.fromkeys(strings))

    def _invoke_floss(self, floss_path: Path, target_path: Path) -> dict[str, Any] | None:
        try:
            proc = subprocess.run(
                [
                    str(floss_path),
                    "-j",
                    "--only",
                    "decoded",
                    "stack",
                    "tight",
                    "-n",
                    str(self._MIN_LENGTH),
                    "--",
                    str(target_path),
                ],
                capture_output=True,
                text=True,
                timeout=self._floss_timeout_sec,
            )
        except Exception:
            return None

        if proc.returncode != 0 or not proc.stdout.strip():
            return None

        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return None

    def _all_pattern_rules(self) -> list[dict[str, Any]]:
        rules: list[dict[str, Any]] = []
        for group_rules in self._registry.strings_patterns.values():
            rules.extend(group_rules)
        return rules

    @classmethod
    def _extract_strings(cls, pe_bytes: bytes) -> list[str]:
        ascii_pattern = re.compile(rb"[ -~]{%d,}" % cls._MIN_LENGTH)
        utf16_pattern = re.compile(rb"(?:[ -~]\x00){%d,}" % cls._MIN_LENGTH)

        ascii_strings = [match.decode("utf-8", errors="ignore") for match in ascii_pattern.findall(pe_bytes)]
        utf16_strings = [
            match.decode("utf-16le", errors="ignore")
            for match in utf16_pattern.findall(pe_bytes)
        ]
        return ascii_strings + utf16_strings

    @staticmethod
    def _count_matches(strings: list[str], rules: list[dict[str, Any]]) -> int:
        count = 0
        for text in strings:
            if any(StringsFeaturesExtractor._rule_matches(text, rule) for rule in rules):
                count += 1
        return count

    @staticmethod
    def _rule_matches(text: str, rule: dict[str, Any]) -> bool:
        match_type = rule.get("match_type", "regex")
        normalized_text = text.casefold()
        if match_type == "regex":
            pattern = rule.get("compiled_regex")
            return bool(pattern and pattern.search(text))
        if match_type == "prefix":
            return normalized_text.startswith(rule.get("normalized", ""))
        return normalized_text == rule.get("normalized", "")
