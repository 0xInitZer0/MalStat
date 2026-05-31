from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.inference import DEFAULT_BATCH_PATTERNS
from src.inference import analyze_directory
from src.reporting import ReportBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze every PE file in a directory with the canonical trained model.")
    parser.add_argument("input_dir", type=Path, help="Directory containing files to analyze.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional directory for batch-analysis outputs.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern to include. Repeatable. Defaults to *.exe, *.dll, *.sys.",
    )
    parser.add_argument("--enable-virustotal", action="store_true", help="Enable VirusTotal hash lookups during analysis.")
    parser.add_argument("--no-html", action="store_true", help="Skip per-file HTML report rendering.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first analysis error.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config.load(args.config).resolve(PROJECT_ROOT)
    builder = ReportBuilder.from_config(config, enable_virustotal=args.enable_virustotal)
    result = analyze_directory(
        args.input_dir,
        config=config,
        builder=builder,
        patterns=args.pattern or DEFAULT_BATCH_PATTERNS,
        recursive=args.recursive,
        output_root=args.output_root,
        write_html=not args.no_html,
        fail_fast=args.fail_fast,
    )

    verdict_counter = Counter(
        record["verdict"]
        for record in result.summary_records
        if record["status"] == "ok" and record["verdict"]
    )

    print(f"input_root={result.input_root}")
    print(f"processed_count={result.processed_count}")
    print(f"failure_count={result.failure_count}")
    print(f"summary_json={result.artifact_paths.summary_json_path}")
    print(f"summary_csv={result.artifact_paths.summary_csv_path}")
    for verdict, count in sorted(verdict_counter.items()):
        print(f"verdict_{verdict}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())