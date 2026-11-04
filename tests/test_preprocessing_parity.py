"""Training and inference must map HU -> model input identically."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.image_utils import (preprocess_hu_slice, apply_lung_window,
                               lung_window_uint8_to_hu, CT_NORM_MIN_HU, CT_NORM_MAX_HU)


def _slice():
    rng = np.random.RandomState(0)
    hu = rng.uniform(-1000, 150, (128, 128)).astype(np.float32)  # inside lung-window range
    hu[10:20, 10:20] = 60.0
    return hu


class TestParity(unittest.TestCase):
    def test_air_and_range(self):
        out = preprocess_hu_slice(np.full((64, 64), -1000.0, np.float32), (64, 64))
        self.assertTrue(np.allclose(out, 0.0))
        out = preprocess_hu_slice(np.full((64, 64), 400.0, np.float32), (64, 64))
        self.assertTrue(np.allclose(out, 1.0))

    def test_display_image_roundtrip_matches_hu_path(self):
        hu = _slice()
        exact = preprocess_hu_slice(hu, (128, 128))
        approx = preprocess_hu_slice(lung_window_uint8_to_hu(apply_lung_window(hu)), (128, 128))
        # only 8-bit quantisation error (~6 HU of a 1400 HU range)
        self.assertLess(float(np.abs(exact - approx).max()), 0.006)

    def test_old_inference_mapping_was_wrong(self):
        # old path: uint8/255 of lung-windowed image. Air (-1000 HU) mapped to ~0.23, training mapped to 0.0
        air = np.full((8, 8), -1000.0, np.float32)
        old = apply_lung_window(air).astype(np.float32) / 255.0
        new = preprocess_hu_slice(air, (8, 8))
        self.assertGreater(float(old.mean() - new.mean()), 0.2)


try:
    import torch  # noqa
    from model.model import preprocess_image

    class TestModelPath(unittest.TestCase):
        def test_hu_path_equals_training_path(self):
            hu = _slice()
            t = preprocess_image(None, hu_slice=hu)
            self.assertEqual(tuple(t.shape), (1, 1, 256, 256))
            self.assertTrue(np.allclose(t[0, 0].numpy(), preprocess_hu_slice(hu, (256, 256))))

        def test_crop_path_matches_training_extraction(self):
            from utils.crops import crop_patch_2d
            from utils.image_utils import normalize_fixed_hu
            hu = _slice()
            crop = {"center_xy": (60.0, 64.0), "spacing_mm": (0.7, 0.7), "patch_mm": 64.0, "out_px": 64}
            t = preprocess_image(None, hu_slice=hu, crop=crop)
            self.assertEqual(tuple(t.shape), (1, 1, 64, 64))
            expect = normalize_fixed_hu(crop_patch_2d(hu, (60.0, 64.0), (0.7, 0.7), 64.0, 64))
            self.assertTrue(np.allclose(t[0, 0].numpy(), expect))
except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
