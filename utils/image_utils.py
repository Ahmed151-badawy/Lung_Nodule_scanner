"""
Clinical CT & DICOM Preprocessing Utilities
Hounsfield Unit Calibration, Standardized Lung Windowing, and Aspect-Preserving Geometry
"""

import logging
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Standard Lung Window Definitions (HU)
LUNG_WINDOW_CENTER = -600.0
LUNG_WINDOW_WIDTH = 1500.0

# Fixed CT Clipping Range for Deep Learning Normalization
CT_NORM_MIN_HU = -1000.0
CT_NORM_MAX_HU = 400.0


def extract_dicom_value(val: Any) -> float:
    """Safely extract float from pydicom MultiValue, list, or scalar."""
    if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
        try:
            return float(list(val)[0])
        except (IndexError, TypeError, ValueError):
            return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def convert_to_hounsfield_units(
    pixel_array: np.ndarray,
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0
) -> np.ndarray:
    """
    Convert raw CT scanner stored values to standardized Hounsfield Units (HU):
        HU = pixel_value * RescaleSlope + RescaleIntercept
    """
    hu = pixel_array.astype(np.float32) * float(rescale_slope) + float(rescale_intercept)
    return hu


def apply_window_level(
    image_hu: np.ndarray,
    center: float = LUNG_WINDOW_CENTER,
    width: float = LUNG_WINDOW_WIDTH
) -> np.ndarray:
    """
    Apply clinical CT Window/Level transformation in Hounsfield Units:
        Lower Bound = Center - Width / 2
        Upper Bound = Center + Width / 2
    """
    img_min = center - (width / 2.0)
    img_max = center + (width / 2.0)
    windowed = np.clip(image_hu, img_min, img_max)
    # Scale linearly to [0, 255]
    scaled = (windowed - img_min) / (img_max - img_min + 1e-8) * 255.0
    return scaled.astype(np.uint8)


def apply_lung_window(image_hu: np.ndarray) -> np.ndarray:
    """
    Apply standard clinical lung parenchyma window (-600 HU center, 1500 HU width).
    Covers range [-1350 HU, +150 HU].
    """
    return apply_window_level(image_hu, center=LUNG_WINDOW_CENTER, width=LUNG_WINDOW_WIDTH)


def normalize_fixed_hu(
    image_hu: np.ndarray,
    min_hu: float = CT_NORM_MIN_HU,
    max_hu: float = CT_NORM_MAX_HU
) -> np.ndarray:
    """
    Standardize CT intensities using a fixed physiological range [-1000, 400] HU.
    Prevents per-slice contrast drift across scans.
    Returns float32 array in [0.0, 1.0].
    """
    clipped = np.clip(image_hu, min_hu, max_hu)
    norm = (clipped - min_hu) / (max_hu - min_hu)
    return norm.astype(np.float32)


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_val: Union[int, Tuple[int, int, int]] = 0
) -> np.ndarray:
    """
    Resize image to target_size (width, height) preserving aspect ratio via symmetric padding.
    Avoids geometric distortion of nodule circularity/sphericity.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if image.ndim == 3:
        padded = np.full((target_h, target_w, image.shape[2]), pad_val, dtype=image.dtype)
    else:
        padded = np.full((target_h, target_w), pad_val, dtype=image.dtype)

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return padded


def preprocess_hu_slice(
    hu: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    SINGLE source of truth for model input intensity/geometry.
    Used by training (data/dataset.py) AND inference (model.preprocess_image).
    HU -> clip [-1000, 400] -> [0, 1] -> aspect-preserving resize/pad.
    Returns float32 [H, W].
    """
    norm = normalize_fixed_hu(hu)
    return resize_with_padding(norm, target_size)


def lung_window_uint8_to_hu(img_uint8: np.ndarray) -> np.ndarray:
    """
    Approximate inverse of apply_lung_window() for 8-bit display images.
    LOSSY: 8-bit quantisation (~5.9 HU steps) and everything above +150 HU is
    saturated. Only valid if the image was produced with the standard lung
    window (-600/1500). Prefer passing true HU wherever it is available.
    """
    lo = LUNG_WINDOW_CENTER - LUNG_WINDOW_WIDTH / 2.0
    return img_uint8.astype(np.float32) / 255.0 * LUNG_WINDOW_WIDTH + lo


def load_dicom_image(dicom_path: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Load DICOM file with full Hounsfield calibration and metadata preservation.

    Returns:
        (image_bgr_uint8, metadata_dict)
        metadata includes:
            - 'pixel_spacing': (row_spacing_mm, col_spacing_mm) or None
            - 'slice_thickness': float or None
            - 'patient_id': str
            - 'study_date': str
            - 'rescale_slope': float
            - 'rescale_intercept': float
    """
    meta: Dict[str, Any] = {
        'pixel_spacing': None,
        'slice_thickness': None,
        'patient_id': 'UNKNOWN',
        'study_date': 'UNKNOWN',
        'modality': 'CT'
    }

    try:
        import pydicom
    except ImportError:
        logger.error("pydicom is required for loading DICOM files. Install with: pip install pydicom")
        return None, meta

    try:
        dcm = pydicom.dcmread(dicom_path)
        pixel_array = dcm.pixel_array

        slope = extract_dicom_value(getattr(dcm, 'RescaleSlope', 1.0))
        intercept = extract_dicom_value(getattr(dcm, 'RescaleIntercept', 0.0))
        meta['rescale_slope'] = slope
        meta['rescale_intercept'] = intercept

        # Convert to true Hounsfield Units
        hu_array = convert_to_hounsfield_units(pixel_array, slope, intercept)

        # Extract Physical Spacing
        if hasattr(dcm, 'PixelSpacing') and dcm.PixelSpacing is not None:
            spacing = list(dcm.PixelSpacing)
            meta['pixel_spacing'] = (float(spacing[0]), float(spacing[1]))

        if hasattr(dcm, 'SliceThickness') and dcm.SliceThickness is not None:
            meta['slice_thickness'] = extract_dicom_value(dcm.SliceThickness)

        meta['patient_id'] = str(getattr(dcm, 'PatientID', Path(dicom_path).stem))
        meta['study_date'] = str(getattr(dcm, 'StudyDate', 'N/A'))
        meta['modality'] = str(getattr(dcm, 'Modality', 'CT'))

        # Check for window metadata, fallback to standard lung window
        if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
            center = extract_dicom_value(dcm.WindowCenter)
            width = extract_dicom_value(dcm.WindowWidth)
            if width > 0:
                display_uint8 = apply_window_level(hu_array, center, width)
            else:
                display_uint8 = apply_lung_window(hu_array)
        else:
            display_uint8 = apply_lung_window(hu_array)

        meta['hu_slice'] = hu_array  # true HU, for model input / measurements

        # Convert to 3-channel BGR for display and GUI rendering
        if display_uint8.ndim == 2:
            display_bgr = cv2.cvtColor(display_uint8, cv2.COLOR_GRAY2BGR)
        else:
            display_bgr = display_uint8

        logger.info(
            "Loaded DICOM: %s | Spacing: %s | HU Range: [%.1f, %.1f]",
            dicom_path, meta['pixel_spacing'], hu_array.min(), hu_array.max()
        )
        return display_bgr, meta

    except Exception as exc:
        logger.error("Failed to parse DICOM '%s': %s", dicom_path, exc)
        return None, meta


def load_image_any(file_path: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Unified loader for both standard image formats (.png, .jpg) and DICOM (.dcm).
    """
    p = Path(file_path)
    if p.suffix.lower() == '.dcm':
        return load_dicom_image(file_path)

    # Standard image loading
    img = cv2.imread(file_path)
    meta: Dict[str, Any] = {
        'pixel_spacing': None,
        'slice_thickness': None,
        'patient_id': p.stem,
        'study_date': 'N/A',
        'modality': 'Image'
    }
    return img, meta
