import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.lidc_labels import consensus_label as c


class T(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(c([1, 2, 2, 1])["label"], 0)
        self.assertEqual(c([4, 5, 4])["label"], 1)
        self.assertIsNone(c([3, 3, 3]))            # ambiguous
        self.assertIsNone(c([2, 4, 3]))            # median 3
        self.assertIsNone(c([5, 5]))               # too few readers
        self.assertEqual(c([2, 2, 5, 4, 2])["median_malignancy"], 2.0)


if __name__ == "__main__":
    unittest.main()
