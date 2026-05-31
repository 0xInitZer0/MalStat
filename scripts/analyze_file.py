from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _maybe_reexec_with_project_venv() -> None:
    expected_python = (PROJECT_ROOT / ".venv" / "Scripts" / "python.exe").resolve()
    if not expected_python.exists():
        return
    if os.environ.get("MALSTAT_ALREADY_REEXEC") == "1":
        return

    current_python = Path(sys.executable).resolve()
    if current_python == expected_python:
        return

    env = dict(os.environ)
    env["MALSTAT_ALREADY_REEXEC"] = "1"
    completed = subprocess.run(
        [str(expected_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        env=env,
        check=False,
    )
    raise SystemExit(completed.returncode)


_maybe_reexec_with_project_venv()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.inference import analyze_file_path
from src.reporting import ReportBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a PE file with the canonical trained model.")
    parser.add_argument("input_path", type=Path, help="Path to the file to analyze.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument("--json-out", type=Path, default=None, help="Where to save the JSON analysis payload.")
    parser.add_argument("--html-out", type=Path, default=None, help="Where to save the HTML report.")
    parser.add_argument("--enable-virustotal", action="store_true", help="Enable VirusTotal hash lookups during analysis.")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report rendering.")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT
    config = Config.load(args.config).resolve(project_root)
    builder = ReportBuilder.from_config(config, enable_virustotal=args.enable_virustotal)
    result = analyze_file_path(
        args.input_path,
        config=config,
        builder=builder,
        json_out=args.json_out,
        html_out=args.html_out,
        write_html=not args.no_html,
    )

    print(f"input={result.input_path}")
    print(f"verdict={result.report.verdict}")
    print(f"probability={result.report.probability:.6f}")
    print(f"verdict_threshold={result.report.verdict_threshold}")
    print(f"model_name={result.report.model_name}")
    print(f"model_version={result.report.model_version}")
    print(f"json_report={result.output_paths.json_report_path}")
    if not args.no_html:
        print(f"html_report={result.output_paths.html_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())