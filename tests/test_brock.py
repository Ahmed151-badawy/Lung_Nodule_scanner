import math, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expert_system.brock_model import BrockUniversityRiskCalculator as B


class TestBrock(unittest.TestCase):
    def test_published_example_4mm_male_62(self):
        # 62y male, 4 mm solid, not upper lobe, 1 nodule, no other factors -> 0.14%
        r = B.calculate_risk(4, 62, False, is_upper_lobe=False)
        self.assertAlmostEqual(r.risk_percentage, 0.14, places=2)

    def test_hand_computed_logit(self):
        r = B.calculate_risk(10, 70, True, True, True, "part_solid", True, 2, True)
        x = (-6.7892 + 0.0287 * 8 + 0.6011 + 0.2961 + 0.2953
             - 5.3854 * (1.0 - 1.58113883) + 0.3770 + 0.6581
             - 0.0824 * (2 - 4) + 0.7729)
        self.assertAlmostEqual(r.malignancy_probability, 1 / (1 + math.exp(-x)), places=4)

    def test_monotonic_in_size(self):
        p = [B.calculate_risk(d, 62, False).malignancy_probability for d in (4, 8, 15, 25)]
        self.assertEqual(p, sorted(p))

    def test_out_of_range_warns(self):
        self.assertTrue(B.calculate_risk(300, 64, True).warnings)
        self.assertFalse(B.calculate_risk(12, 64, True).warnings)

    def test_zero_size(self):
        self.assertEqual(B.calculate_risk(0, 60, False).risk_tier, "NEGLIGIBLE")


if __name__ == "__main__":
    unittest.main()
