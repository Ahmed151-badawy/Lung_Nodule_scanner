"""
Tier 1: 2.5D Volumetric Pipeline & Out-of-Distribution (OOD) Quality Gatekeeper
Input plausibility checks. Not a validated medical-device safety gate.
"""

import logging
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch

from utils.image_utils import (
    convert_to_hounsfield_units,
    apply_lung_window,
    resize_with_padding,
    CT_NORM_MIN_HU,
    CT_NORM_MAX_HU
)

logger = logging.getLogger(__name__)


class CTQualityAndOODValidator:
    """
    Validates physical plausibility of CT inputs prior to neural inference.
    Rejects: Non-CT modalities, extreme noise, empty slices, contrast artifacts.
    """
    # Expected thoracic CT physiological thresholds (HU)
    AIR_HU_MIN = -1050.0
    AIR_HU_MAX = -700.0
    BONE_HU_MIN = 250.0
    MIN_LUNG_VOLUME_RATIO = 0.08  # Lung field should occupy at least 8% of thoracic slice
    MAX_NOISE_STD = 600.0

    @classmethod
    def validate_slice(cls, slice_hu: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """
        Run statistical and physical checks on a Hounsfield slice.
        Returns: (is_valid, rejection_reasons)
        """
        rejections = []
        meta = metadata or {}

        # 1. Modality Gate
        modality = str(meta.get("modality", "CT")).upper()
        if modality not in ("CT", "IMAGE"):
            rejections.append(f"Invalid modality '{modality}': Only CT slices permitted.")

        # 2. Dynamic Range Check
        min_hu, max_hu = float(slice_hu.min()), float(slice_hu.max())
        if max_hu < 0:
            rejections.append(f"Absence of soft tissue/bone: Slice max HU is {max_hu:.1f} (Likely blank/air slice).")
        if min_hu > -200:
            rejections.append(f"Absence of lung parenchyma/air: Slice min HU is {min_hu:.1f} (Non-thoracic or corrupted scan).")

        # 3. Thoracic Lung Parenchyma Area Ratio
        lung_pixels = np.logical_and(slice_hu >= -950.0, slice_hu <= -600.0)
        lung_ratio = float(np.sum(lung_pixels)) / float(slice_hu.size)
        if lung_ratio < cls.MIN_LUNG_VOLUME_RATIO:
            rejections.append(
                f"OOD Warning: Lung parenchyma covers only {lung_ratio:.1%} of slice "
                f"(threshold: {cls.MIN_LUNG_VOLUME_RATIO:.1%}). Possible abdominal, neck, or non-lung slice."
            )

        # 4. Sensor Saturation & High Noise Check
        noise_std = float(np.std(slice_hu))
        if noise_std > cls.MAX_NOISE_STD:
            rejections.append(f"Severe noise/artifact detected: Image STD={noise_std:.1f} HU exceeds {cls.MAX_NOISE_STD}.")

        is_valid = len(rejections) == 0
        return is_valid, rejections


class VolumetricContextExtractor:
    """
    Extracts 2.5D slab [z-1, z, z+1] with normalized HU mapping.
    """
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size

    def build_25d_tensor(
        self,
        slice_stack_hu: List[np.ndarray],  # Must contain exactly 3 adjacent slices [z-1, z, z+1]
        spacing_mm: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """
        Convert 3 adjacent HU slices into a normalized [1, 3, H, W] tensor.
        Intensity mapping: Fixed physiological window [-1000, +400] HU -> [0.0, 1.0].
        """
        if len(slice_stack_hu) != 3:
            raise ValueError(f"2.5D context requires exactly 3 slices [z-1, z, z+1], received {len(slice_stack_hu)}")

        channels = []
        for sl in slice_stack_hu:
            # Fixed clip [-1000, 400] HU
            clipped = np.clip(sl, CT_NORM_MIN_HU, CT_NORM_MAX_HU)
            norm = (clipped - CT_NORM_MIN_HU) / (CT_NORM_MAX_HU - CT_NORM_MIN_HU)
            resized = resize_with_padding(norm.astype(np.float32), self.target_size)
            channels.append(resized)

        # Stack into [3, H, W]
        stacked = np.stack(channels, axis=0)  # Shape [3, H, W]
        tensor = torch.from_numpy(stacked).unsqueeze(0)  # Shape [1, 3, H, W]
        return tensor

