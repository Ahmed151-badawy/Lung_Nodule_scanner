import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.metrics import (auroc, sensitivity_at_specificity, expected_calibration_error,
                           brier, grouped_bootstrap_ci)


class TestMetrics(unittest.TestCase):
    def test_auroc_known(self):
        self.assertAlmostEqual(auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertEqual(auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)   # all ties
        self.assertTrue(np.isnan(auroc([1, 1], [0.2, 0.9])))

    def test_sens_at_spec(self):
        y = np.array([0] * 10 + [1] * 4)
        p = np.array(list(np.linspace(0.0, 0.45, 10)) + [0.2, 0.6, 0.7, 0.9])
        self.assertAlmostEqual(sensitivity_at_specificity(y, p, 0.9), 0.75)

    def test_ece_calibrated_vs_overconfident(self):
        rng = np.random.RandomState(0)
        p = rng.uniform(0, 1, 20000)
        y = (rng.uniform(0, 1, 20000) < p).astype(int)
        self.assertLess(expected_calibration_error(y, p), 0.02)
        self.assertGreater(expected_calibration_error(y, np.where(p > 0.5, 0.99, 0.01)), 0.2)
        self.assertAlmostEqual(brier([1, 0], [1.0, 0.0]), 0.0)

    def test_grouped_bootstrap(self):
        rng = np.random.RandomState(1)
        groups = np.repeat(np.arange(40), 3)
        y = rng.randint(0, 2, 120)
        p = np.clip(y * 0.4 + rng.uniform(0, 0.6, 120), 0, 1)
        lo, hi = grouped_bootstrap_ci(auroc, y, p, groups, n_boot=200)
        self.assertLessEqual(lo, auroc(y, p)); self.assertGreaterEqual(hi, auroc(y, p)); self.assertLess(lo, hi)


class TestSplit(unittest.TestCase):
    def test_three_way_disjoint_and_stratified(self):
        try:
            from data.dataset import patient_level_split_3way
        except Exception as e:
            self.skipTest(f"dataset import needs torch: {e}")
        rng = np.random.RandomState(0)
        samples = [{"patient_id": f"P{i}", "label": int(i % 3 == 0 or rng.rand() < 0.1)} for i in range(60) for _ in range(4)]
        tr, va, te = patient_level_split_3way(samples, 0.2, 0.2, seed=5)
        ids = lambda g: {s["patient_id"] for s in g}
        self.assertFalse(ids(tr) & ids(va) or ids(tr) & ids(te) or ids(va) & ids(te))
        self.assertEqual(len(tr) + len(va) + len(te), len(samples))
        for g in (tr, va, te):
            self.assertEqual({s["label"] for s in g}, {0, 1})


if __name__ == "__main__":
    unittest.main()
