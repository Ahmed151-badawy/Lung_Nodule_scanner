"""Dataset over nodule patches produced by scripts/build_lidc_crops.py."""
import csv, random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from utils.image_utils import normalize_fixed_hu


def read_manifest(crops_dir: str) -> List[Dict[str, Any]]:
    root = Path(crops_dir)
    with open(root / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        r["diameter_mm"] = float(r["diameter_mm"])
        r["abs_path"] = str(root / r["patch_path"])
    return rows


def augment_patch(hu: np.ndarray) -> np.ndarray:
    """Label-preserving augmentation on an HU patch (training only). Uses `random` so
    DataLoader workers get distinct streams. No scaling: physical size carries information."""
    h, w = hu.shape
    if random.random() < 0.5:
        hu = hu[:, ::-1]
    if random.random() < 0.5:
        hu = hu[::-1, :]
    hu = np.rot90(hu, random.randint(0, 3))
    ang = random.uniform(-15, 15)
    tx, ty = random.uniform(-3, 3), random.uniform(-3, 3)          # <= 3 mm at 1 mm/px
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), ang, 1.0)
    M[:, 2] += (tx, ty)
    hu = cv2.warpAffine(np.ascontiguousarray(hu, dtype=np.float32), M, (w, h),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1000.0)
    return hu + random.gauss(0.0, 15.0)                             # global HU offset (scanner/kernel drift)


class NoduleCropDataset(Dataset):
    """Returns (x[1,H,W] in [0,1], y[1], meta) - same contract as LungNoduleDataset."""

    def __init__(self, rows: List[Dict[str, Any]], augment: bool = False):
        self.rows, self.augment = rows, augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        hu = np.load(r["abs_path"]).astype(np.float32)
        if self.augment:
            hu = augment_patch(hu)
        x = torch.from_numpy(normalize_fixed_hu(hu)).unsqueeze(0).float()
        y = torch.tensor([float(r["label"])], dtype=torch.float32)
        return x, y, {"patient_id": r["patient_id"], "nodule_id": r["nodule_id"]}
