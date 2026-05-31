"""Dataset quality check script.

Runs a series of QA checks over aggregate dataset stores and prints
a structured report. Exits with code 1 if any critical issues are found.

Usage:
    python scripts/check_dataset.py
    python scripts/check_dataset.py --strict          # fail on warnings too
    python scripts/check_dataset.py --min-samples 500 # require at least 500 rows
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas pyarrow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Colour helpers (graceful fallback on Windows without ANSI)
# ---------------------------------------------------------------------------

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"

OK    = lambda t: _c(t, "32")
WARN  = lambda t: _c(t, "33")
ERR   = lambda t: _c(t, "31")
BOLD  = lambda t: _c(t, "1")
DIM   = lambda t: _c(t, "2")


# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, name: str, level: str, message: str, detail: str = ""):
        """level: 'ok' | 'warning' | 'error'"""
        self.name = name
        self.level = level
        self.message = message
        self.detail = detail

    def __repr__(self) -> str:
        icon = {"ok": "✓", "warning": "!", "error": "✗"}.get(self.level, "?")
        colour = {"ok": OK, "warning": WARN, "error": ERR}.get(self.level, str)
        line = colour(f"  [{icon}] {self.name}: {self.message}")
        if self.detail:
            for d in self.detail.splitlines():
                line += "\n" + DIM(f"       {d}")
        return line


def _ok(name, msg, detail=""):    return CheckResult(name, "ok",      msg, detail)
def _warn(name, msg, detail=""):  return CheckResult(name, "warning", msg, detail)
def _err(name, msg, detail=""):   return CheckResult(name, "error",   msg, detail)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_files_exist(meta_path: Path, feat_path: Path) -> list[CheckResult]:
    results = []
    for p in [meta_path, feat_path]:
        if p.exists():
            results.append(_ok("files_exist", f"Found {p.name}"))
        else:
            results.append(_err("files_exist", f"Missing: {p}"))
    return results


def check_shape(meta: pd.DataFrame, feat: pd.DataFrame, min_samples: int) -> list[CheckResult]:
    results = []
    n_meta = len(meta)
    n_feat = len(feat)

    if n_meta == 0:
        results.append(_err("row_count", "Metadata is empty — no samples at all."))
    elif n_meta < min_samples:
        results.append(_warn("row_count",
            f"Only {n_meta} samples in metadata (--min-samples={min_samples}).",
            "Consider collecting more samples before training."))
    else:
        results.append(_ok("row_count", f"{n_meta} samples in metadata."))

    if n_meta != n_feat:
        results.append(_err("shape_sync",
            f"Row count mismatch: metadata={n_meta}, features={n_feat}.",
            "Run: python scripts/manage_dataset.py list to investigate."))
    else:
        results.append(_ok("shape_sync", f"Metadata and features row counts match ({n_meta})."))

    results.append(_ok("feature_count", f"{feat.shape[1]} feature columns in features table."))
    return results


def check_sha256_sync(meta: pd.DataFrame, feat: pd.DataFrame) -> list[CheckResult]:
    results = []
    meta_hashes = set(meta["sha256"].dropna().astype(str))
    feat_hashes = set(feat["sha256"].dropna().astype(str))

    only_meta = meta_hashes - feat_hashes
    only_feat = feat_hashes - meta_hashes

    if only_meta:
        sample = ", ".join(list(only_meta)[:3])
        results.append(_err("sha256_sync",
            f"{len(only_meta)} SHA-256 in metadata with no features row.",
            f"Examples: {sample}"))
    elif only_feat:
        sample = ", ".join(list(only_feat)[:3])
        results.append(_err("sha256_sync",
            f"{len(only_feat)} SHA-256 in features with no metadata row.",
            f"Examples: {sample}"))
    else:
        results.append(_ok("sha256_sync", "SHA-256 sets match between metadata and features."))
    return results


def check_duplicates(meta: pd.DataFrame) -> list[CheckResult]:
    results = []
    dup_mask = meta["sha256"].duplicated(keep=False)
    n_dup = dup_mask.sum()
    if n_dup > 0:
        examples = meta.loc[dup_mask, ["sample_id", "sha256", "original_name"]].head(5).to_string(index=False)
        results.append(_err("duplicates",
            f"{n_dup} rows share a duplicated SHA-256 hash.",
            f"Fix: python scripts/clear_dataset.py --match sha256=<hash>\n{examples}"))
    else:
        results.append(_ok("duplicates", "No duplicate SHA-256 hashes found."))
    return results


def check_label_balance(meta: pd.DataFrame) -> list[CheckResult]:
    results = []
    label_col = meta["label"].astype(str)
    counts = label_col.value_counts()
    detail_lines = [f"  label={k}: {v} samples" for k, v in counts.items()]
    detail = "\n".join(detail_lines)

    n_mal = int(counts.get("1", 0))
    n_ben = int(counts.get("0", 0))
    n_unlabeled = int(counts.drop(labels=["0", "1"], errors="ignore").sum())

    if n_unlabeled > 0:
        results.append(_warn("labels",
            f"{n_unlabeled} rows have no label (not 0 or 1).",
            "These rows will be excluded from training. Assign labels or delete them.\n" + detail))
    elif n_mal == 0 or n_ben == 0:
        results.append(_err("labels",
            "One class is completely missing (only malware or only benign).",
            detail))
    else:
        ratio = max(n_mal, n_ben) / max(min(n_mal, n_ben), 1)
        if ratio > 5:
            results.append(_warn("label_balance",
                f"Severe class imbalance: malicious={n_mal}, benign={n_ben} (ratio {ratio:.1f}x).",
                "Use class_weight='balanced' in LightGBM/LogisticRegression.\n" + detail))
        elif ratio > 2:
            results.append(_warn("label_balance",
                f"Moderate class imbalance: malicious={n_mal}, benign={n_ben} (ratio {ratio:.1f}x).",
                detail))
        else:
            results.append(_ok("label_balance",
                f"Class balance OK: malicious={n_mal}, benign={n_ben}.",
                detail))
    return results


def check_label_confidence(meta: pd.DataFrame) -> list[CheckResult]:
    results = []
    conf = meta["label_confidence"].astype(str) if "label_confidence" in meta.columns else pd.Series(dtype=str)
    if conf.empty:
        results.append(_warn("label_confidence", "Column 'label_confidence' missing from metadata."))
        return results

    counts = conf.value_counts()
    n_low = int(counts.get("low", 0))
    n_total = len(meta)
    detail = "\n".join(f"  {k}: {v}" for k, v in counts.items())

    if n_low > 0:
        pct = 100 * n_low / n_total
        results.append(_warn("label_confidence",
            f"{n_low} samples ({pct:.1f}%) have label_confidence=low — exclude from first training run.",
            f"Fix: python scripts/manage_dataset.py delete --match label_confidence=low\n{detail}"))
    else:
        results.append(_ok("label_confidence", "No low-confidence labels found.", detail))
    return results


def check_pe_valid(meta: pd.DataFrame) -> list[CheckResult]:
    results = []
    if "is_pe_valid" not in meta.columns:
        results.append(_warn("pe_valid", "Column 'is_pe_valid' not found in metadata — skipping check."))
        return results

    invalid_mask = meta["is_pe_valid"].astype(str).isin(["0", "False", "false"])
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        examples = meta.loc[invalid_mask, ["sample_id", "original_name", "pe_kind"]].head(5).to_string(index=False)
        results.append(_warn("pe_valid",
            f"{n_invalid} samples have is_pe_valid=0 — these cannot be fully analysed.",
            f"Consider removing them:\n  python scripts/clear_dataset.py --match is_pe_valid=0\n{examples}"))
    else:
        results.append(_ok("pe_valid", "All samples have is_pe_valid=1."))
    return results


def check_pe_kind(meta: pd.DataFrame) -> list[CheckResult]:
    results = []
    if "pe_kind" not in meta.columns:
        results.append(_warn("pe_kind", "Column 'pe_kind' missing — skipping check."))
        return results

    counts = meta["pe_kind"].astype(str).value_counts()
    detail = "\n".join(f"  {k}: {v}" for k, v in counts.items())
    n_non_pe = int(counts.get("non_pe", 0))

    if n_non_pe > 0:
        results.append(_warn("pe_kind",
            f"{n_non_pe} samples classified as non_pe — may not have full feature coverage.",
            f"Distribution:\n{detail}"))
    else:
        results.append(_ok("pe_kind", "No non_pe samples found.", f"Distribution:\n{detail}"))
    return results


def check_nan_features(feat: pd.DataFrame) -> list[CheckResult]:
    results = []
    service_cols = {"sample_id", "sha256", "label"}
    feat_cols = [c for c in feat.columns if c not in service_cols]
    nan_counts = feat[feat_cols].isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if len(nan_cols) == 0:
        results.append(_ok("nan_features", "No NaN values found in feature columns."))
    else:
        total_nan = nan_cols.sum()
        top = nan_cols.sort_values(ascending=False).head(10)
        detail = "\n".join(f"  {col}: {cnt} NaN ({100*cnt/len(feat):.1f}%)" for col, cnt in top.items())
        if len(nan_cols) > 10:
            detail += f"\n  ... and {len(nan_cols) - 10} more columns"
        results.append(_warn("nan_features",
            f"{len(nan_cols)} feature columns contain NaN ({total_nan} total NaN cells).",
            f"Top columns by NaN count:\n{detail}\n"
            "These will be filled during preprocessing (FeaturePreprocessor)."))
    return results


def check_sample_id_integrity(meta: pd.DataFrame, feat: pd.DataFrame) -> list[CheckResult]:
    results = []
    meta_ids = set(meta["sample_id"].dropna().astype(str))
    feat_ids = set(feat["sample_id"].dropna().astype(str))
    diff = meta_ids.symmetric_difference(feat_ids)
    if diff:
        results.append(_err("sample_id_integrity",
            f"{len(diff)} sample_id values present in one table but not the other.",
            f"Mismatched IDs (first 10): {sorted(diff, key=str)[:10]}"))
    else:
        results.append(_ok("sample_id_integrity", f"sample_id values are consistent across both tables."))
    return results


def check_source_distribution(meta: pd.DataFrame) -> list[CheckResult]:
    results = []
    if "source" not in meta.columns:
        return results
    counts = meta["source"].astype(str).value_counts()
    detail = "\n".join(f"  {k}: {v}" for k, v in counts.items())
    results.append(_ok("source_distribution", f"{len(counts)} unique sources.", detail))
    return results


def check_family_leakage_risk(meta: pd.DataFrame) -> list[CheckResult]:
    """Warn if any family has so many samples it could dominate a random split."""
    results = []
    if "source_family" not in meta.columns:
        return results

    counts = meta["source_family"].astype(str).value_counts()
    n_total = len(meta)
    dominant = counts[counts / n_total > 0.20]

    if not dominant.empty:
        detail = "\n".join(f"  {k}: {v} ({100*v/n_total:.1f}%)" for k, v in dominant.items())
        results.append(_warn("family_leakage_risk",
            f"{len(dominant)} source_family/families cover >20% of dataset each.",
            f"Use family-stratified split to avoid data leakage:\n{detail}"))
    else:
        top5 = counts.head(5)
        detail = "\n".join(f"  {k}: {v}" for k, v in top5.items())
        results.append(_ok("family_leakage_risk",
            "No single family dominates (>20%) the dataset.",
            f"Top-5 families:\n{detail}"))
    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_checks(project_root: Path, min_samples: int, strict: bool) -> int:
    meta_path = project_root / "data" / "metadata" / "samples_metadata.parquet"
    feat_path = project_root / "data" / "features" / "features_example.parquet"

    # Fall back to CSV if parquet missing
    if not meta_path.exists():
        meta_path = project_root / "data" / "metadata" / "samples_metadata.csv"
    if not feat_path.exists():
        feat_path = project_root / "data" / "features" / "features_example.csv"

    print(BOLD("\n═══════════════════════════════════════════"))
    print(BOLD("  Dataset QA Report"))
    print(BOLD("═══════════════════════════════════════════\n"))

    # File existence check first
    file_results = check_files_exist(meta_path, feat_path)
    for r in file_results:
        print(r)
    if any(r.level == "error" for r in file_results):
        print(ERR("\n  Cannot continue — dataset files are missing.\n"))
        return 1

    # Load
    print()
    try:
        meta = pd.read_parquet(meta_path) if str(meta_path).endswith(".parquet") else pd.read_csv(meta_path)
        feat = pd.read_parquet(feat_path) if str(feat_path).endswith(".parquet") else pd.read_csv(feat_path)
    except Exception as e:
        print(_err("load", f"Failed to load dataset: {e}"))
        return 1

    # Run all checks grouped by category
    all_results: list[CheckResult] = []

    sections = [
        ("Shape & Sync",         check_shape(meta, feat, min_samples)
                                 + check_sha256_sync(meta, feat)
                                 + check_sample_id_integrity(meta, feat)),
        ("Duplicates",           check_duplicates(meta)),
        ("Labels",               check_label_balance(meta) + check_label_confidence(meta)),
        ("PE Validity",          check_pe_valid(meta) + check_pe_kind(meta)),
        ("Feature Quality",      check_nan_features(feat)),
        ("Corpus Distribution",  check_source_distribution(meta)
                                 + check_family_leakage_risk(meta)),
    ]

    n_errors = 0
    n_warnings = 0

    for section_name, results in sections:
        print(BOLD(f"  ── {section_name}"))
        for r in results:
            print(r)
            if r.level == "error":
                n_errors += 1
            elif r.level == "warning":
                n_warnings += 1
        all_results.extend(results)
        print()

    # Summary
    print(BOLD("═══════════════════════════════════════════"))
    if n_errors == 0 and n_warnings == 0:
        print(OK(f"  All checks passed. Dataset is ready for training.\n"))
        return 0
    elif n_errors == 0:
        print(WARN(f"  {n_warnings} warning(s), 0 errors."))
        if strict:
            print(WARN("  --strict mode: treating warnings as errors.\n"))
            return 1
        else:
            print(WARN("  Dataset can proceed to training, but review warnings above.\n"))
            return 0
    else:
        print(ERR(f"  {n_errors} error(s), {n_warnings} warning(s). Fix errors before training.\n"))
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dataset quality checks before training."
    )
    parser.add_argument(
        "--project-root", default=str(PROJECT_ROOT),
        help="Project root directory (default: auto-detected)."
    )
    parser.add_argument(
        "--min-samples", type=int, default=100,
        help="Minimum expected number of samples (default: 100)."
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 if any warnings are found."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_checks(Path(args.project_root), args.min_samples, args.strict))
