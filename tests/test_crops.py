import sys, unittest
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.crops import crop_patch_2d


def _slice(row_sp, col_sp, diam_mm=18.0, center=(300, 210)):
    s = np.full((512, 512), -800.0, np.float32)
    ax = (int(round(diam_mm / 2 / col_sp)), int(round(diam_mm / 2 / row_sp)))   # (x, y) radii in px
    cv2.ellipse(s, center, ax, 0, 0, 360, 60.0, -1)
    return s


class TestCrop(unittest.TestCase):
    def test_physical_scale_and_centering_with_anisotropic_spacing(self):
        for sp in ((0.7, 0.7), (0.6, 0.9), (1.0, 1.0)):
            s = _slice(*sp)
            p = crop_patch_2d(s, (300, 210), sp, 64.0, 64)
            self.assertEqual(p.shape, (64, 64))
            m = p > -300
            d_eq = 2 * np.sqrt(m.sum() / np.pi)                    # 1 mm per output pixel
            self.assertAlmostEqual(d_eq, 18.0, delta=1.5, msg=str(sp))
            ys, xs = np.nonzero(m)
            self.assertAlmostEqual(xs.mean(), 31.5, delta=0.7)
            self.assertAlmostEqual(ys.mean(), 31.5, delta=0.7)

    def test_same_nodule_same_patch_across_spacings(self):
        a = crop_patch_2d(_slice(0.6, 0.6), (300, 210), (0.6, 0.6))
        b = crop_patch_2d(_slice(0.9, 0.9), (300, 210), (0.9, 0.9))
        self.assertLess(float(np.mean((a > -300) != (b > -300))), 0.03)

    def test_outside_image_is_filled(self):
        p = crop_patch_2d(np.zeros((100, 100), np.float32), (2, 2), (1.0, 1.0), 64.0, 64, fill_hu=-1000.0)
        self.assertEqual(float(p[0, 0]), -1000.0)
        self.assertEqual(float(p[40, 40]), 0.0)


    def test_int16_storage_is_lossless_enough_for_training(self):
        from utils.image_utils import normalize_fixed_hu
        p = crop_patch_2d(_slice(0.7, 0.7), (300, 210), (0.7, 0.7))
        stored = np.clip(p, -1024, 3071).astype(np.int16)           # what build_lidc_crops.py saves
        diff = np.abs(normalize_fixed_hu(p) - normalize_fixed_hu(stored.astype(np.float32)))
        self.assertLess(float(diff.max()), 0.0015)                  # <= ~2 HU


if __name__ == "__main__":
    unittest.main()
