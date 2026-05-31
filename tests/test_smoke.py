from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.core.config import Config
from src.training.pipeline import TabularFeaturePreprocessor


class ProjectSmokeTests(unittest.TestCase):
    def test_canonical_artifacts_exist(self) -> None:
        required_paths = [
            PROJECT_ROOT / "models" / "calibrated_model.pkl",
            PROJECT_ROOT / "models" / "preprocessor.pkl",
            PROJECT_ROOT / "models" / "feature_columns.json",
            PROJECT_ROOT / "configs" / "verdict_rules.yaml",
            PROJECT_ROOT / "templates" / "report.html.j2",
        ]
        for path in required_paths:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), f"Missing required artifact: {path}")

    def test_config_resolves_to_existing_paths(self) -> None:
        config = Config().resolve(PROJECT_ROOT)
        self.assertTrue(config.model_path.exists())
        self.assertTrue(config.preprocessor_path.exists())
        self.assertTrue(config.feature_columns_path.exists())
        self.assertTrue(config.template_path.exists())

    def test_analyze_file_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "analyze_file.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Analyze a PE file", result.stdout)
        self.assertIn("--enable-virustotal", result.stdout)

    def test_runtime_out_of_scope_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            json_out = Path(temp_dir) / "smoke.analysis.json"
            config_path = Path(temp_dir) / "smoke_config.yaml"
            config_path.write_text("experiment_log_path: models/not_shipped.json\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "analyze_file.py"),
                    str(PROJECT_ROOT / "VERSION"),
                    "--config",
                    str(config_path),
                    "--no-html",
                    "--json-out",
                    str(json_out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertTrue(json_out.exists(), "Expected smoke JSON report was not created.")

            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("verdict"), "out_of_scope")

    def test_tabular_preprocessor_repairs_object_statistics(self) -> None:
        preprocessor = TabularFeaturePreprocessor(
            feature_columns=["file_size", "ep_section_name"],
            numeric_columns=["file_size"],
            categorical_columns=["ep_section_name"],
        )
        fit_frame = pd.DataFrame([
            {"file_size": 1024.0, "ep_section_name": ".text"},
            {"file_size": np.nan, "ep_section_name": ".rsrc"},
        ])

        preprocessor.fit(fit_frame, pd.Series([0, 1], dtype=int))

        numeric_imputer = preprocessor._transformer.named_transformers_["numeric"].named_steps["imputer"]
        numeric_imputer.statistics_ = np.asarray(numeric_imputer.statistics_, dtype=object)

        transformed = preprocessor.transform(pd.DataFrame([
            {"file_size": 2048.0, "ep_section_name": ".idata"},
        ]))

        self.assertEqual(transformed.shape[0], 1)
        self.assertEqual(numeric_imputer.statistics_.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()