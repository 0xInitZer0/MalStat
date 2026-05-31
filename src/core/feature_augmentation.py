from __future__ import annotations

from typing import Any

import pandas as pd


DERIVED_FEATURE_NAMES = [
    "overlay_to_file_ratio",
    "imports_per_mb",
    "exports_per_mb",
    "strings_per_mb",
    "signature_missing",
    "signature_invalid",
    "strings_url_ratio",
    "strings_ip_ratio",
    "strings_shell_ratio",
    "strings_crypto_ratio",
    "strings_base64_ratio",
    "strings_mz_ratio",
    "strings_unique_ratio",
    "api_injection_ratio",
    "api_loading_ratio",
    "api_network_ratio",
    "api_execution_ratio",
    "api_antidebug_ratio",
    "api_privilege_ratio",
    "sec_executable_ratio",
    "sec_wx_ratio",
    "sec_zero_rawsize_ratio",
    "rsrc_to_file_ratio",
    "text_to_file_ratio",
    "suspicious_dll_ratio",
]


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    augmented = pd.DataFrame(frame).copy()
    index = augmented.index

    file_size = _numeric_series(augmented, "file_size", index=index)
    file_size_mb = file_size / float(1024 * 1024)
    imports_func_count = _numeric_series(augmented, "imports_func_count", index=index)
    exports_func_count = _numeric_series(augmented, "exports_func_count", index=index)
    strings_count = _numeric_series(augmented, "strings_count", index=index)
    imports_dll_count = _numeric_series(augmented, "imports_dll_count", index=index)
    sec_count = _numeric_series(augmented, "sec_count", index=index)
    has_signature = _numeric_series(augmented, "has_signature", index=index)
    signature_is_valid = _numeric_series(augmented, "signature_is_valid", index=index)
    signature_present = has_signature > 0.0

    derived_columns = {
        "overlay_to_file_ratio": _safe_ratio(
            _numeric_series(augmented, "overlay_size", index=index),
            file_size,
        ),
        "imports_per_mb": _safe_ratio(imports_func_count, file_size_mb),
        "exports_per_mb": _safe_ratio(exports_func_count, file_size_mb),
        "strings_per_mb": _safe_ratio(strings_count, file_size_mb),
        # Split unsigned files from genuinely invalid signatures.
        "signature_missing": (~signature_present).astype(float),
        "signature_invalid": (signature_present & (signature_is_valid <= 0.0)).astype(float),
        "strings_url_ratio": _safe_ratio(
            _numeric_series(augmented, "strings_url_count", index=index),
            strings_count,
        ),
        "strings_ip_ratio": _safe_ratio(
            _numeric_series(augmented, "strings_ip_count", index=index),
            strings_count,
        ),
        "strings_shell_ratio": _safe_ratio(
            _numeric_series(augmented, "strings_shell_count", index=index),
            strings_count,
        ),
        "strings_crypto_ratio": _safe_ratio(
            _numeric_series(augmented, "strings_crypto_count", index=index),
            strings_count,
        ),
        "strings_base64_ratio": _safe_ratio(
            _numeric_series(augmented, "strings_base64_count", index=index),
            strings_count,
        ),
        "strings_mz_ratio": _safe_ratio(
            _numeric_series(augmented, "strings_mz_count", index=index),
            strings_count,
        ),
        "strings_unique_ratio": _safe_ratio(
            _numeric_series(augmented, "strings_unique_count", index=index),
            strings_count,
        ),
        "api_injection_ratio": _safe_ratio(
            _numeric_series(augmented, "api_injection_count", index=index),
            imports_func_count,
        ),
        "api_loading_ratio": _safe_ratio(
            _numeric_series(augmented, "api_loading_count", index=index),
            imports_func_count,
        ),
        "api_network_ratio": _safe_ratio(
            _numeric_series(augmented, "api_network_count", index=index),
            imports_func_count,
        ),
        "api_execution_ratio": _safe_ratio(
            _numeric_series(augmented, "api_execution_count", index=index),
            imports_func_count,
        ),
        "api_antidebug_ratio": _safe_ratio(
            _numeric_series(augmented, "api_antidebug_count", index=index),
            imports_func_count,
        ),
        "api_privilege_ratio": _safe_ratio(
            _numeric_series(augmented, "api_privilege_count", index=index),
            imports_func_count,
        ),
        "sec_executable_ratio": _safe_ratio(
            _numeric_series(augmented, "sec_executable_count", index=index),
            sec_count,
        ),
        "sec_wx_ratio": _safe_ratio(
            _numeric_series(augmented, "sec_wx_count", index=index),
            sec_count,
        ),
        "sec_zero_rawsize_ratio": _safe_ratio(
            _numeric_series(augmented, "sec_zero_rawsize_count", index=index),
            sec_count,
        ),
        "rsrc_to_file_ratio": _safe_ratio(
            _numeric_series(augmented, "rsrc_size", index=index),
            file_size,
        ),
        "text_to_file_ratio": _safe_ratio(
            _numeric_series(augmented, "text_raw_size", index=index),
            file_size,
        ),
        "suspicious_dll_ratio": _safe_ratio(
            _numeric_series(augmented, "suspicious_dll_count", index=index),
            imports_dll_count,
        ),
    }

    for column_name, values in derived_columns.items():
        augmented[column_name] = values.round(6)

    return augmented


def extend_feature_columns(feature_columns: list[str]) -> list[str]:
    resolved = [name for name in feature_columns if name != "signature_is_valid"]
    for name in DERIVED_FEATURE_NAMES:
        if name not in resolved:
            resolved.append(name)
    return resolved


def enrich_feature_dict(features: dict[str, Any]) -> dict[str, Any]:
    augmented = add_derived_features(pd.DataFrame([features]))
    return dict(augmented.iloc[0].to_dict())


def _numeric_series(frame: pd.DataFrame, column_name: str, *, index: pd.Index) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series(0.0, index=index, dtype=float)
    return pd.to_numeric(frame[column_name], errors="coerce").fillna(0.0).astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(0.0, index=numerator.index, dtype=float)
    valid = denominator > 0.0
    if bool(valid.any()):
        result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result.fillna(0.0)