"""
Physical-scale nodule patch extraction, shared by TRAINING (scripts/build_lidc_crops.py)
and INFERENCE (model.preprocess_image). One function => no train/serve drift.
"""
from typing import Tuple
import numpy as np
import cv2


def crop_patch_2d(
    hu_slice: np.ndarray,
    center_xy: Tuple[float, float],
    spacing_mm: Tuple[float, float],
    patch_mm: float = 64.0,
    out_px: int = 64,
    fill_hu: float = -1000.0,
) -> np.ndarray:
    """
    Resample a patch_mm x patch_mm square of the axial slice, centred on center_xy
    (x=column, y=row, pixel-centre index coordinates), to out_px x out_px pixels.
    Output pixel size = patch_mm/out_px mm (1 mm for the defaults) regardless of the
    scan's native spacing. spacing_mm = (row_spacing, col_spacing). Outside the image -> fill_hu.
    Returns float32 HU, shape [out_px, out_px].
    """
    if hu_slice.ndim != 2:
        raise ValueError("crop_patch_2d expects a 2D HU slice")
    row_sp, col_sp = float(spacing_mm[0]), float(spacing_mm[1])
    if row_sp <= 0 or col_sp <= 0:
        raise ValueError("pixel spacing must be positive")
    cx, cy = float(center_xy[0]), float(center_xy[1])
    step = patch_mm / out_px
    offs = (np.arange(out_px, dtype=np.float32) + 0.5) * step - patch_mm / 2.0   # mm from centre
    map_x = np.tile(cx + offs / col_sp, (out_px, 1)).astype(np.float32)
    map_y = np.tile((cy + offs / row_sp)[:, None], (1, out_px)).astype(np.float32)
    return cv2.remap(
        hu_slice.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=float(fill_hu),
    )
