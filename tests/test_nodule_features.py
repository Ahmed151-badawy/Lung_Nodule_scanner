"""Shape features must describe the segmented nodule, not whatever is brightest on the slice."""
import sys, unittest
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.model import extract_features

SP = (0.75, 0.75)


def _img_with_nodule():
    img = np.full((512, 512), 30, np.uint8)
    cv2.ellipse(img, (256, 256), (220, 180), 0, 0, 360, 120, -1)   # big bright body
    cv2.circle(img, (160, 230), 12, 200, -1)
    mask = np.zeros((512, 512), np.uint8)
    cv2.circle(mask, (160, 230), 12, 1, -1)
    return img, mask


class TestFeatures(unittest.TestCase):
    def test_mask_features_describe_nodule(self):
        img, mask = _img_with_nodule()
        f = extract_features(img, spacing_mm=SP, nodule_mask=mask)
        self.assertEqual(f["source"], "nodule_mask")
        self.assertAlmostEqual(f["size_mm"], 18.0, delta=1.5)
        self.assertEqual(f["margin"], "smooth")
        self.assertFalse(f["spiculation"])
        self.assertGreater(f["sphericity"], 0.8)

    def test_legacy_whole_image_is_flagged_and_wrong(self):
        img, _ = _img_with_nodule()
        f = extract_features(img, spacing_mm=SP)
        self.assertEqual(f["source"], "whole_image_otsu")
        self.assertGreater(f["size_mm"], 100)      # measures the body, not the nodule

    def test_empty_mask_gives_no_size(self):
        img, _ = _img_with_nodule()
        f = extract_features(img, spacing_mm=SP, nodule_mask=np.zeros((512, 512), np.uint8))
        self.assertIsNone(f["size_mm"])
        self.assertEqual(f["margin"], "indeterminate")
        self.assertFalse(f["spiculation"])


if __name__ == "__main__":
    unittest.main()
