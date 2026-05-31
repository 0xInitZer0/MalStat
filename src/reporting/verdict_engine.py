"""
VerdictEngine converts a flat feature dict into a list of FeatureVerdict values.

Rules are loaded from configs/verdict_rules.yaml.
Every feature receives a verdict, even when no explicit rule exists
(status "ok" with a neutral explanation).
"""

from __future__ import annotations
from typing import Any

from src.reporting.models import FeatureVerdict


class VerdictEngine:
    """Assign a verdict to each numeric or boolean feature."""

    _STATUS_PRIORITY = {"ok": 0, "warning": 1, "suspicious": 2}
    _GROUP_PREFIXES = {
        "ep_": "entry_point",
        "sec_": "sections",
        "imports_": "imports",
        "api_": "imports",
        "has_api_": "imports",
        "has_": "header",
        "strings_": "strings",
        "floss_": "strings",
        "rsrc_": "resources",
        "asm_": "code",
        "file_": "file_general",
    }

    def __init__(self, rules: dict[str, Any]):
        """
        Args:
            rules: rules dictionary from verdict_rules.yaml keyed by feature name.
        """
        self._rules = rules

    def evaluate(self, features: dict[str, Any]) -> list[FeatureVerdict]:
        """
        Evaluate all features and return a list of FeatureVerdict values.

        The order matches the key order in features.
        Features without a rule receive status "ok" with a neutral explanation.
        """
        verdicts: list[FeatureVerdict] = []
        for name, value in features.items():
            rule = self._rules.get(name)
            verdicts.append(self._apply_rule(name, value, rule, features))
        return verdicts

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _apply_rule(
        self,
        name: str,
        value: Any,
        rule: dict | None,
        features: dict[str, Any],
    ) -> FeatureVerdict:
        if rule is None:
            return self._no_rule_verdict(name, value)

        if not self._rule_applies(rule, features):
            return self._not_applicable_verdict(name, value, rule)

        rule_type = rule.get("type", "boolean")

        if rule_type == "boolean":
            return self._boolean_verdict(name, value, rule)
        if rule_type == "info":
            return self._info_verdict(name, value, rule)
        if rule_type == "threshold":
            return self._threshold_verdict(name, value, rule)
        if rule_type == "range":
            return self._range_verdict(name, value, rule)

        return self._no_rule_verdict(name, value)

    def _boolean_verdict(
        self, name: str, value: Any, rule: dict
    ) -> FeatureVerdict:
        expected_value = rule.get("expected_value", False)
        triggered = bool(value) != bool(expected_value)
        if triggered:
            status = rule.get("status_if_triggered", "warning")
            severity = rule.get("severity", 5)
            explanation = rule.get("explanation", "Deviation from the expected value.")
        else:
            status = "ok"
            severity = 0
            explanation = rule.get("explanation_ok", "Value is within the expected range.")
        return FeatureVerdict(
            feature_name=name,
            value=value,
            expected=str(expected_value),
            status=status,
            severity=severity if triggered else 0,
            explanation=explanation,
            group=rule.get("group", "other"),
            triggered_indicator=None,
        )

    def _info_verdict(
        self, name: str, value: Any, rule: dict
    ) -> FeatureVerdict:
        truthy = bool(value)
        explanation = rule.get(
            "explanation_true" if truthy else "explanation_false",
            rule.get("explanation", "Informational feature without an evaluative status."),
        )
        return FeatureVerdict(
            feature_name=name,
            value=value,
            expected=rule.get("expected", "information"),
            status="ok",
            severity=0,
            explanation=explanation,
            group=rule.get("group", "other"),
            triggered_indicator=None,
        )

    def _threshold_verdict(
        self, name: str, value: Any, rule: dict
    ) -> FeatureVerdict:
        try:
            fval = float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return self._no_rule_verdict(name, value)

        suspicious_above = rule.get("suspicious_above")
        warning_above = rule.get("warning_above")
        suspicious_below = rule.get("suspicious_below")
        warning_below = rule.get("warning_below")

        if suspicious_above is not None and fval > suspicious_above:
            return FeatureVerdict(
                feature_name=name,
                value=value,
                expected=f"< {warning_above if warning_above is not None else suspicious_above}",
                status="suspicious",
                severity=rule.get("severity_suspicious", 7),
                explanation=rule.get("explanation_suspicious", f"Value {fval:.3f} exceeds threshold {suspicious_above}."),
                group=rule.get("group", "other"),
            )
        if warning_above is not None and fval > warning_above:
            return FeatureVerdict(
                feature_name=name,
                value=value,
                expected=f"< {warning_above}",
                status="warning",
                severity=rule.get("severity_warning", 3),
                explanation=rule.get("explanation_warning", f"Value {fval:.3f} is elevated."),
                group=rule.get("group", "other"),
            )
        if suspicious_below is not None and fval < suspicious_below:
            return FeatureVerdict(
                feature_name=name,
                value=value,
                expected=f"> {warning_below if warning_below is not None else suspicious_below}",
                status="suspicious",
                severity=rule.get("severity_suspicious", 7),
                explanation=rule.get("explanation_suspicious", f"Value {fval:.3f} is below threshold {suspicious_below}."),
                group=rule.get("group", "other"),
            )
        if warning_below is not None and fval < warning_below:
            return FeatureVerdict(
                feature_name=name,
                value=value,
                expected=f"> {warning_below}",
                status="warning",
                severity=rule.get("severity_warning", 3),
                explanation=rule.get("explanation_warning", f"Value {fval:.3f} is low."),
                group=rule.get("group", "other"),
            )

        return FeatureVerdict(
            feature_name=name,
            value=value,
            expected=self._threshold_expected(rule),
            status="ok",
            severity=0,
            explanation=rule.get("explanation_ok", "Value is within the expected range."),
            group=rule.get("group", "other"),
        )

    def _range_verdict(
        self, name: str, value: Any, rule: dict
    ) -> FeatureVerdict:
        try:
            fval = float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return self._no_rule_verdict(name, value)

        ok_min = rule.get("ok_min")
        ok_max = rule.get("ok_max")
        in_range = (
            (ok_min is None or fval >= ok_min)
            and (ok_max is None or fval <= ok_max)
        )

        if in_range:
            return FeatureVerdict(
                feature_name=name,
                value=value,
                expected=f"{ok_min}–{ok_max}",
                status="ok",
                severity=0,
                explanation=rule.get("explanation_ok", "Value is within the normal range."),
                group=rule.get("group", "other"),
            )
        return FeatureVerdict(
            feature_name=name,
            value=value,
            expected=f"{ok_min}–{ok_max}",
            status="warning",
            severity=rule.get("severity_warning", 4),
            explanation=rule.get("explanation_warning", f"Value {fval} is outside the normal range."),
            group=rule.get("group", "other"),
        )

    @staticmethod
    def _rule_applies(rule: dict[str, Any], features: dict[str, Any]) -> bool:
        conditions = rule.get("apply_if")
        if not isinstance(conditions, dict) or not conditions:
            return True

        for feature_name, expected in conditions.items():
            actual = features.get(feature_name)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
                continue
            if actual != expected:
                return False
        return True

    @staticmethod
    def _threshold_expected(rule: dict[str, Any]) -> str:
        if rule.get("warning_above") is not None:
            return f"< {rule['warning_above']}"
        if rule.get("suspicious_above") is not None:
            return f"< {rule['suspicious_above']}"
        if rule.get("warning_below") is not None:
            return f"> {rule['warning_below']}"
        if rule.get("suspicious_below") is not None:
            return f"> {rule['suspicious_below']}"
        return "threshold not defined"

    @staticmethod
    def _not_applicable_verdict(name: str, value: Any, rule: dict[str, Any]) -> FeatureVerdict:
        return FeatureVerdict(
            feature_name=name,
            value=value,
            expected="rule not applicable",
            status="ok",
            severity=0,
            explanation=rule.get("explanation_not_applicable", "The rule does not apply to this PE type."),
            group=rule.get("group", "other"),
        )

    @staticmethod
    def _no_rule_verdict(name: str, value: Any) -> FeatureVerdict:
        group = VerdictEngine._infer_group(name)
        return FeatureVerdict(
            feature_name=name,
            value=value,
            expected="info",
            status="ok",
            severity=0,
            explanation=VerdictEngine._default_explanation(name, value, group),
            group=group,
        )

    @classmethod
    def _infer_group(cls, name: str) -> str:
        for prefix, group in cls._GROUP_PREFIXES.items():
            if name.startswith(prefix):
                return group
        if any(token in name for token in ("entropy", "overlay", "bitness", "machine", "imagebase")):
            return "header"
        return "other"

    @classmethod
    def _default_explanation(cls, name: str, value: Any, group: str) -> str:
        if name.startswith("has_api_"):
            api_name = name.removeprefix("has_api_")
            state = "present" if bool(value) else "not detected"
            return (
                f"A single import of API '{api_name}' is {state}. By itself this import is not treated as an IOC; "
                "evaluation is handled by the aggregated api_*_count rules."
            )

        if name.startswith("has_") and isinstance(value, bool):
            state = "enabled" if value else "disabled"
            return f"Contextual PE flag '{name}' is {state}. The feature is shown as a descriptor without a dedicated penalizing rule."

        exact_messages = {
            "machine_type": "Machine field from the COFF Header. Used as an architectural PE descriptor.",
            "pe_bitness": "PE bitness (32/64-bit). This is a contextual format descriptor, not a standalone IOC.",
            "size_of_image": "In-memory image size (SizeOfImage). Useful to the model as a structural PE descriptor.",
            "size_of_headers": "Aggregate size of PE headers on disk. Used as a structural build descriptor.",
            "subsystem": "PE Subsystem defines the execution environment type (GUI/CUI/native, etc.).",
            "dll_characteristics": "DllCharacteristics bitfield. Detailed interpretation is handled by separate flags such as ASLR/DEP/CFG.",
            "entrypoint_rva": "Entry point RVA. The raw value is used as a structural descriptor rather than an IOC.",
            "timestamp_raw": "Raw TimeDateStamp value from the COFF Header. Anomaly evaluation is delegated to timestamp_is_zero and timestamp_is_future.",
            "imagebase": "Preferred PE image base. Abnormality is evaluated by the separate imagebase_is_nonstandard rule.",
            "num_rich_entries": "Number of Rich Header entries. This is a build-origin and toolchain-profile descriptor.",
            "ep_section_name": "Name of the section that contains the entry point. Used as context for startup code placement.",
            "ep_first_byte": "First byte at the entry point. Useful as a low-level EP stub descriptor, but not very informative without a dedicated threshold.",
            "overlay_size": "Overlay size after the end of PE structures. Interpreted together with overlay share and overlay entropy.",
            "overlay_entropy": "Overlay entropy. Useful together with overlay_size and should not be interpreted in isolation.",
            "ratio_printable": "Ratio of printable bytes in the main PE body. This is a general statistical content descriptor.",
            "imports_func_count": "Total number of imported functions. For large framework applications, the raw count is not an IOC by itself.",
            "exports_func_count": "Total number of exported functions. Useful as a descriptor of binary type and API surface.",
            "suspicious_dll_count": "Number of DLLs from the watched list of risky dependencies. Interpretation requires the context of the overall import profile.",
            "strings_count": "Total number of extracted strings. For large bundled applications, the absolute count alone is not very informative.",
            "strings_unique_count": "Number of unique strings. Useful as a descriptor of content size and diversity.",
            "strings_avg_len": "Average length of extracted strings. Used as a statistical descriptor, not a standalone IOC.",
            "strings_max_len": "Maximum extracted string length. Useful for finding blob-like or config-like inserts, but not penalized without a dedicated threshold.",
            "asm_total_instructions": "Total number of disassembled instructions. The absolute value depends on program size and nature.",
        }
        if name in exact_messages:
            return exact_messages[name]

        if name.endswith("_count"):
            return f"Count for the '{group}' group. No separate threshold is defined, so the feature remains quantitative context without a penalty."
        if name.endswith("_ratio") or name.startswith("ratio_"):
            return f"Relative metric for the '{group}' group. Used as a statistical descriptor without a dedicated rule-based threshold."
        if name.endswith("_entropy"):
            return f"Entropy metric for the '{group}' group. Without additional context, the value is shown as a neutral descriptor."
        if name.endswith("_size"):
            return f"Size descriptor for the '{group}' group. No dedicated penalizing rule is defined for it."

        return "The feature is shown as a neutral contextual descriptor. No dedicated penalizing rule is defined for it, but it remains available to the model and the analyst."
