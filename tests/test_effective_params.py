import importlib
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

ikp = importlib.import_module("ikp_estimate")
ikp_v2 = importlib.import_module("ikp_estimate_v2")


class EffectiveParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = json.loads(
            (ROOT / "data/results/effective_params.json").read_text()
        )

    def test_artifact_calibration_matches_declared_cohort(self):
        configs = json.loads((ROOT / "configs/all_models.json").read_text())["models"]
        summary = json.loads(
            (ROOT / "data/results/evaluation_summary.json").read_text()
        )
        excluded = {
            "minimax-m1-think", "hunyuan-a13b", "hunyuan-a13b-think",
            "hermes-3-405b", "ling-2.6-flash", "deepseek-v3.1-nex-n1",
            "intellect-3-think", "nemotron-ultra-253b",
        }
        rows = []
        for model in summary:
            config = configs.get(model["model"], {})
            params = model.get("params_B") or config.get("params_B")
            if (config.get("type") == "open" and params and params > 0
                    and model.get("accuracy") is not None
                    and model["model"] not in excluded):
                rows.append((math.log10(float(params)), model["accuracy"]))

        slope, intercept, r, _, _ = stats.linregress(
            np.array([row[0] for row in rows]),
            np.array([row[1] for row in rows]),
        )
        calibration = self.artifact["calibration"]
        self.assertEqual(calibration["n"], len(rows))
        self.assertAlmostEqual(calibration["slope"], slope, places=12)
        self.assertAlmostEqual(calibration["intercept"], intercept, places=12)
        self.assertAlmostEqual(calibration["r_squared"], r ** 2, places=12)

    def test_both_estimators_share_calibration(self):
        self.assertEqual(ikp_v2._SLOPE, ikp.ACC_SLOPE)
        self.assertEqual(ikp_v2._INTERCEPT, ikp.ACC_INTERCEPT)
        self.assertEqual(ikp.CALIB_N, self.artifact["calibration"]["n"])

    def test_readme_example_uses_current_curve(self):
        self.assertAlmostEqual(ikp.estimate_params(0.639), 315.24, places=1)

    def test_refusal_summary_reports_adjusted_interval(self):
        tiers = {
            tier: {"correct": 80, "wrong": 10, "refusal": 10, "total": 100}
            for tier in ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
        }
        result = ikp.refusal_summary(tiers)
        self.assertEqual(result["confidence"], "Caution")
        self.assertAlmostEqual(result["refusal_rate"], 0.10)
        self.assertGreater(result["n_eff_adjusted_B"], result["n_eff_floor_B"])

    def test_family_effect_is_adjusted_and_tested(self):
        effect = self.artifact["density"]["family_effect"]
        self.assertIn("omega_squared", effect)
        self.assertIn("permutation_p", effect)
        self.assertLessEqual(effect["omega_squared"], effect["eta_squared"])


if __name__ == "__main__":
    unittest.main()
