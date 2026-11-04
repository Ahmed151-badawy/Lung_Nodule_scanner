"""
Phase 1: Dataset Loader & Patient-Level Cross-Validation Splitter
Intended for LIDC-IDRI / LUNA16 style multi-radiologist labels (loader for real data still to be written)
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.volume_ood import VolumetricContextExtractor
from utils.image_utils import (
    convert_to_hounsfield_units,
    resize_with_padding,
    preprocess_hu_slice,
    CT_NORM_MIN_HU,
    CT_NORM_MAX_HU
)

logger = logging.getLogger(__name__)


class LungNoduleDataset(Dataset):
    """
    Production-grade Dataset for 2D and 2.5D CT nodule candidates.
    Supports patient-level splits to guarantee no data leakage.
    Consensus malignancy ratings:
      1-2: Benign
      4-5: Malignant
      3: Indeterminate / Ambiguous (excluded or weighted)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        is_25d: bool = False,
        target_size: Tuple[int, int] = (256, 256),
        transform = None
    ):
        """
        Args:
            samples: List of sample dicts:
                     {'patient_id': str, 'slices': [np.ndarray], 'label': int, 'diameter_mm': float}
            is_25d: If True, uses 3 adjacent slices [z-1, z, z+1]
        """
        self.samples = samples
        self.is_25d = is_25d
        self.target_size = target_size
        self.transform = transform
        self.extractor_25d = VolumetricContextExtractor(target_size=target_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample = self.samples[idx]
        slices = sample["slices"]
        label = sample["label"]

        if self.is_25d:
            if len(slices) == 3:
                tensor = self.extractor_25d.build_25d_tensor(slices)
            else:
                # Fallback replicate slice if 3 slices unavailable
                replicated = [slices[0], slices[0], slices[0]]
                tensor = self.extractor_25d.build_25d_tensor(replicated)
            tensor = tensor.squeeze(0)  # [3, H, W]
        else:
            # Single slice 2D
            center_slice = slices[1] if len(slices) == 3 else slices[0]
            resized = preprocess_hu_slice(center_slice.astype(np.float32), self.target_size)
            tensor = torch.from_numpy(resized).unsqueeze(0)  # [1, H, W]

        label_tensor = torch.tensor([float(label)], dtype=torch.float32)

        meta = {
            "patient_id": sample.get("patient_id", "ANON"),
            "diameter_mm": sample.get("diameter_mm", 0.0),
            "label": label
        }

        return tensor, label_tensor, meta


def patient_level_split(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split samples strictly by Patient ID to ensure zero slice leakage.
    """
    rng = np.random.RandomState(seed)
    patient_ids = list(set(s["patient_id"] for s in samples))
    rng.shuffle(patient_ids)

    n_val = max(1, int(len(patient_ids) * val_ratio))
    val_pids = set(patient_ids[:n_val])
    train_pids = set(patient_ids[n_val:])

    train_samples = [s for s in samples if s["patient_id"] in train_pids]
    val_samples = [s for s in samples if s["patient_id"] in val_pids]

    logger.info(
        "Patient-level Split: %d train patients (%d samples), %d val patients (%d samples)",
        len(train_pids), len(train_samples), len(val_pids), len(val_samples)
    )
    return train_samples, val_samples



def patient_level_split_3way(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Train / validation / test split with NO patient shared between splits, stratified by
    whether the patient has any malignant nodule so each split sees both classes.
      train      -> fit weights
      validation -> choose checkpoint AND fit temperature scaling
      test       -> touched once, for the final reported metrics
    """
    rng = np.random.RandomState(seed)
    has_mal: Dict[str, bool] = {}
    for sm in samples:
        has_mal[sm["patient_id"]] = has_mal.get(sm["patient_id"], False) or bool(sm["label"])

    train_p, val_p, test_p = set(), set(), set()
    for flag in (True, False):
        pids = sorted(pid for pid, m in has_mal.items() if m == flag)
        rng.shuffle(pids)
        n_test = int(round(len(pids) * test_ratio))
        n_val = int(round(len(pids) * val_ratio))
        if len(pids) >= 3:                      # keep at least one patient per split when possible
            n_test, n_val = max(1, n_test), max(1, n_val)
        test_p.update(pids[:n_test])
        val_p.update(pids[n_test:n_test + n_val])
        train_p.update(pids[n_test + n_val:])

    assert not (train_p & val_p or train_p & test_p or val_p & test_p), "patient leakage between splits"
    pick = lambda group: [sm for sm in samples if sm["patient_id"] in group]
    tr, va, te = pick(train_p), pick(val_p), pick(test_p)
    logger.info("3-way split: %d/%d/%d patients (%d/%d/%d samples)",
                len(train_p), len(val_p), len(test_p), len(tr), len(va), len(te))
    return tr, va, te
