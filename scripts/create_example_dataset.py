"""
create_example_dataset.py
=========================
Generates example dataset files for the PE ML project:
  - data/features/features_example.parquet  (from features_example.csv)
  - data/metadata/samples_metadata.parquet  (from samples_metadata.csv)

Run from the project root:
    python scripts/create_example_dataset.py

Requirements:
    pip install pandas pyarrow
"""

import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# dtype map for features table — keeps memory low and types explicit
# ---------------------------------------------------------------------------
FEATURES_DTYPE = {
    "sample_id":                   "int32",
    "sha256":                      "string",
    "label":                       "int8",
    # File basics
    "file_size":                   "int64",
    "is_executable_image":        "bool",
    "is_dll":                      "bool",
    "is_exe":                      "bool",
    "is_driver":                  "bool",
    "has_signature":               "bool",
    "overlay_size":                "int64",
    "overlay_entropy":             "float32",
    "file_entropy":                "float32",
    "ratio_printable":             "float32",
    # PE Header
    "num_sections":                "int16",
    "size_of_image":               "int64",
    "size_of_headers":             "int32",
    "subsystem":                   "int16",
    "dll_characteristics":         "int32",
    "entrypoint_rva":              "int64",
    "timestamp_raw":               "int64",
    "timestamp_is_zero":           "bool",
    "timestamp_is_future":         "bool",
    "checksum_is_zero":            "bool",
    "checksum_is_valid":           "bool",
    "imagebase":                   "int64",
    "has_tls":                     "bool",
    "has_debug":                   "bool",
    "has_resources":               "bool",
    "has_relocations":             "bool",
    "is_dotnet":                   "bool",
    # Rich Header
    "has_rich_header":             "bool",
    "num_rich_entries":            "int16",
    "rich_header_is_zeroed":       "bool",
    # Entry Point
    "ep_section_name":             "string",
    "ep_is_in_last_section":       "bool",
    "ep_outside_sections":         "bool",
    "ep_starts_with_pushad":       "bool",
    "ep_starts_with_jmp_near":     "bool",
    "ep_starts_with_nop":          "bool",
    "ep_first_byte":               "int16",
    "ep_first64_entropy":          "float32",
    "ep_section_entropy":          "float32",
    "ep_section_is_wx":            "bool",
    "ep_section_virt_raw_ratio":   "float32",
    # Sections
    "sec_count":                   "int16",
    "sec_entropy_mean":            "float32",
    "sec_entropy_max":             "float32",
    "sec_executable_count":        "int16",
    "sec_wx_count":                "int16",
    "sec_zero_rawsize_count":      "int16",
    "has_upx_section":             "bool",
    "has_vmp_section":             "bool",
    "has_themida_section":         "bool",
    "has_mpress_section":          "bool",
    "has_aspack_section":          "bool",
    "has_petite_section":          "bool",
    "has_nsp_section":             "bool",
    "has_enigma_section":          "bool",
    "has_obsidium_section":        "bool",
    "has_no_text_section":         "bool",
    "has_no_data_section":         "bool",
    "rsrc_size":                   "int64",
    "text_raw_size":               "int64",
    "text_entropy":                "float32",
    # Imports / Exports
    "imports_dll_count":           "int16",
    "imports_func_count":          "int32",
    "exports_func_count":          "int32",
    "imports_by_ordinal_count":    "int32",
    "ratio_ordinal_imports":       "float32",
    "has_wininet":                 "bool",
    "has_ws2_32":                  "bool",
    "has_advapi32":                "bool",
    "has_crypt32":                 "bool",
    "has_shell32":                 "bool",
    "has_urlmon":                  "bool",
    "api_network_count":           "int16",
    "api_injection_count":         "int16",
    "api_enumeration_count":       "int16",
    "api_antidebug_count":         "int16",
    "api_privilege_count":         "int16",
    "api_crypto_count":            "int16",
    "api_registry_count":          "int16",
    "api_process_count":           "int16",
    # Strings
    "strings_count":               "int32",
    "strings_avg_len":             "float32",
    "strings_max_len":             "int32",
    "strings_unique_count":        "int32",
    "strings_url_count":           "int16",
    "strings_ip_count":            "int16",
    "strings_domain_count":        "int16",
    "strings_registry_count":      "int16",
    "strings_path_count":          "int16",
    "strings_crypto_count":        "int16",
    "strings_shell_count":         "int16",
    "strings_mz_count":            "int16",
    "strings_base64_long_count":   "int16",
    "strings_hex_long_count":      "int16",
    "floss_strings_count":         "int32",
    "floss_suspicious_count":      "int16",
    "ratio_printable_strings":     "float32",
    # Resources
    "rsrc_has_version_info":       "bool",
    "rsrc_has_manifest":           "bool",
    "rsrc_has_icon":               "bool",
    "rsrc_has_embedded_pe":        "bool",
    "rsrc_max_entropy":            "float32",
    "rsrc_type_count":             "int16",
    "rsrc_num_entries":            "int32",
}

BOOL_COLS = [k for k, v in FEATURES_DTYPE.items() if v == "bool"]
INT_COLS  = [k for k, v in FEATURES_DTYPE.items() if v.startswith("int")]


def _apply_dtypes(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    for col, dtype in dtype_map.items():
        if col not in df.columns:
            continue
        if dtype == "bool":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float).astype(bool)
        elif dtype == "string":
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)
    return df


def build_features_parquet() -> None:
    src = ROOT / "data" / "features" / "features_example.csv"
    dst = ROOT / "data" / "features" / "features_example.parquet"

    df = pd.read_csv(src, dtype=str)          # read everything as string first
    df = _apply_dtypes(df, FEATURES_DTYPE)

    df.to_parquet(dst, index=False, engine="pyarrow", compression="snappy")
    print(f"[OK] features_example.parquet written → {dst}")
    print(f"     rows={len(df)}, cols={len(df.columns)}")
    print(f"     file size: {dst.stat().st_size:,} bytes")


def build_metadata_parquet() -> None:
    src = ROOT / "data" / "metadata" / "samples_metadata.csv"
    dst = ROOT / "data" / "metadata" / "samples_metadata.parquet"

    df = pd.read_csv(src)
    df.to_parquet(dst, index=False, engine="pyarrow", compression="snappy")
    print(f"[OK] samples_metadata.parquet written → {dst}")


if __name__ == "__main__":
    build_features_parquet()
    build_metadata_parquet()
