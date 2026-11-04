import sys, unittest
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.nodule_segmenter import AnatomicalNoduleSegmenter

SP = (0.75, 0.75)


def make_slice(nodule_r=12, center=(160, 230), extra=None):
    s = np.full((512, 512), -1000.0, np.float32)
    cv2.ellipse(s, (256, 256), (220, 180), 0, 0, 360, 40.0, -1)      # body
    cv2.circle(s, (256, 400), 25, 400.0, -1)                         # spine
    cv2.ellipse(s, (170, 250), (75, 120), 0, 0, 360, -800.0, -1)     # lungs
    cv2.ellipse(s, (342, 250), (75, 120), 0, 0, 360, -800.0, -1)
    if nodule_r:
        cv2.circle(s, center, nodule_r, 60.0, -1)
    if extra:
        extra(s)
    return s


class TestSegmenter(unittest.TestCase):
    def setUp(self):
        self.seg = AnatomicalNoduleSegmenter()

    def test_measures_nodule_not_body(self):
        hu = make_slice()
        for seed in ((160, 230), None):
            m = self.seg.segment_candidate_roi(hu, seed_point=seed, spacing_mm=SP)
            self.assertGreater(m.sum(), 0)
            self.assertLess(m.sum(), 1000)                 # body would be ~100k px
            r = self.seg.measure_recist_11(m, hu, SP)
            self.assertAlmostEqual(r.longest_diameter_mm, 18.0, delta=2.0)
            self.assertEqual(r.warnings, [])

    def test_seed_near_nodule_snaps(self):
        hu = make_slice()
        m = self.seg.segment_candidate_roi(hu, seed_point=(160, 245), spacing_mm=SP)  # ~3 px outside
        self.assertGreater(m.sum(), 0)

    def test_seed_in_air_returns_empty(self):
        m = self.seg.segment_candidate_roi(make_slice(), seed_point=(200, 330), spacing_mm=SP)
        self.assertEqual(int(m.sum()), 0)

    def test_oversized_mass_rejected(self):
        hu = make_slice(nodule_r=55)                       # ~82 mm
        m = self.seg.segment_candidate_roi(hu, seed_point=(160, 230), spacing_mm=SP)
        self.assertEqual(int(m.sum()), 0)

    def test_no_lungs_returns_empty(self):
        m = self.seg.segment_candidate_roi(np.full((256, 256), 40.0, np.float32))
        self.assertEqual(int(m.sum()), 0)

    def test_measure_warns_on_huge_object(self):
        mask = np.zeros((512, 512), np.uint8)
        cv2.ellipse(mask, (256, 256), (220, 180), 0, 0, 360, 1, -1)
        r = self.seg.measure_recist_11(mask, np.full((512, 512), 40.0, np.float32), SP)
        self.assertTrue(any("not a nodule-scale" in w for w in r.warnings))


if __name__ == "__main__":
    unittest.main()
