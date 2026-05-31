"""
src/core/indicator_registry.py

Config-driven registry for indicator matching.
Loads YAML indicator definitions once and exposes normalized accessors and
matching helpers for extractors.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml


class IndicatorRegistry:
    """Loads indicator YAML files and provides normalized matching helpers."""

    def __init__(self, indicators_dir: str | Path):
        self._indicators_dir = Path(indicators_dir)
        self._strings_config = self._load_yaml("strings.yaml")
        self._dlls_config = self._load_yaml("dlls.yaml")
        self._apis_config = self._load_yaml("apis.yaml")
        self._sections_config = self._load_yaml("sections.yaml")
        self._packers_config = self._load_yaml("packers.yaml")
        self._opcodes_config = self._load_yaml("opcodes.yaml")

    @property
    def strings_patterns(self) -> dict[str, list[dict[str, Any]]]:
        patterns = self._strings_config.get("patterns", {})
        return {
            key: [self._normalize_rule(item) for item in items]
            for key, items in patterns.items()
        }

    @property
    def suspicious_dlls(self) -> list[dict[str, Any]]:
        return [
            self._normalize_rule(item)
            for item in self._dlls_config.get("suspicious_dlls", [])
        ]

    @property
    def api_categories(self) -> dict[str, list[dict[str, Any]]]:
        categories = self._apis_config.get("categories", {})
        return {
            category: [
                self._normalize_rule(item, extra={"category": category})
                for item in items
            ]
            for category, items in categories.items()
        }

    @property
    def section_indicators(self) -> list[dict[str, Any]]:
        return [
            self._normalize_rule(item)
            for item in self._packers_config.get("section_names", [])
        ]

    @property
    def packer_string_markers(self) -> list[dict[str, Any]]:
        return [
            self._normalize_rule(item)
            for item in self._packers_config.get("string_markers", [])
        ]

    @property
    def ep_byte_markers(self) -> list[dict[str, Any]]:
        return list(self._packers_config.get("ep_byte_markers", []))

    @property
    def standard_section_names(self) -> set[str]:
        return {
            self.normalize_token(name)
            for name in self._sections_config.get("standard_section_names", [])
        }

    @property
    def section_thresholds(self) -> dict[str, Any]:
        return dict(self._sections_config.get("entropy_thresholds", {}))

    @property
    def section_ratio_threshold(self) -> float:
        return float(self._sections_config.get("virt_raw_ratio_suspicious", 5.0))

    @property
    def opcode_categories(self) -> dict[str, list[str]]:
        categories = self._opcodes_config.get("opcode_categories", {})
        return {
            name: list(details.get("mnemonics", []))
            for name, details in categories.items()
        }

    def find_matches(
        self,
        values: list[str] | set[str] | tuple[str, ...],
        rules: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        matched: list[dict[str, Any]] = []
        for value in values:
            for rule in rules:
                if self.rule_matches(value, rule):
                    matched.append(rule)
        return matched

    def rule_matches(self, value: str, rule: dict[str, Any]) -> bool:
        candidate = self.normalize_token(value)
        match_type = rule.get("match_type", "exact")

        if match_type == "exact":
            return candidate == rule.get("normalized", "")
        if match_type == "prefix":
            prefix = rule.get("normalized", "")
            return bool(prefix) and candidate.startswith(prefix)
        if match_type == "regex":
            pattern = rule.get("compiled_regex")
            return bool(pattern and pattern.search(value))
        return False

    @staticmethod
    def normalize_token(value: str | None) -> str:
        return (value or "").strip("\x00 ").casefold()

    @staticmethod
    def make_feature_token(value: str | None) -> str:
        normalized = IndicatorRegistry.normalize_token(value)
        normalized = normalized.replace(".dll", "")
        normalized = normalized.replace(".drv", "")
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
        normalized = normalized.strip("_")
        return normalized or "unknown"

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        path = self._indicators_dir / filename
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            return {}
        return data

    def _normalize_rule(
        self,
        rule: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = dict(rule)
        if extra:
            normalized.update(extra)

        value = normalized.get("name", normalized.get("pattern", ""))
        # Rules with a 'pattern' key are regex by default; 'name' keys default to 'exact'
        default_match_type = "regex" if "pattern" in normalized and "name" not in normalized else "exact"
        normalized["match_type"] = normalized.get("match_type", default_match_type)
        normalized["normalized"] = self.normalize_token(value)

        if normalized["match_type"] == "regex":
            flags = self._parse_regex_flags(normalized.get("flags"))
            pattern = normalized.get("pattern", normalized.get("name", ""))
            normalized["compiled_regex"] = re.compile(pattern, flags)

        return normalized

    @staticmethod
    def _parse_regex_flags(raw_flags: Any) -> int:
        if raw_flags is None:
            return 0
        if isinstance(raw_flags, int):
            return raw_flags
        if isinstance(raw_flags, str):
            flags = 0
            for chunk in re.split(r"[|, ]+", raw_flags):
                token = chunk.strip().upper()
                if token == "IGNORECASE":
                    flags |= re.IGNORECASE
                elif token == "MULTILINE":
                    flags |= re.MULTILINE
                elif token == "DOTALL":
                    flags |= re.DOTALL
            return flags
        return 0