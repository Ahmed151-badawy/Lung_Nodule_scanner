"""
Clinical Lung Nodule AI - Deep Learning Model & Inference Pipeline
Medical AI Systems Architecture - Production Grade
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, Any

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Clinical Safety Flags
DEMO_MODE_ENV = "NODULE_DEMO"


def is_demo_mode_active() -> bool:
    """Return True if demo mode is explicitly enabled by the environment."""
    return os.getenv(DEMO_MODE_ENV, "0").strip().lower() in ("1", "true", "yes")


class ConvBlock(nn.Module):
    """Standard Conv-BN-ReLU block with explicit feature scaling."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NoduleCNN(nn.Module):
    """
    Lightweight, resolution-agnostic CNN for lung nodule classification.
    Replaces 33.5M dense layer with Global Average Pooling (GAP) + Dropout (~0.4M params).
    Returns raw logits for numerical stability and calibration.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1, dropout_rate: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Raw logits [B, num_classes]
        """
        feats = self.features(x)
        logits = self.head(feats)
        return logits


# Alias for backward compatibility
LungNoduleCNN = NoduleCNN


def load_model(
    model_path: str = "model/best_model.pth",
    device: Optional[torch.device] = None,
    num_classes: int = 1
) -> Optional[NoduleCNN]:
    """
    Safely load trained PyTorch model weights using weights_only=True.

    Args:
        model_path: Path to checkpoint (.pth / .pt)
        device: Target execution device (CPU or CUDA)
        num_classes: Target prediction classes (default 1 for binary sigmoid logit)

    Returns:
        NoduleCNN model in eval mode, or None if file not found in demo mode.
    Raises:
        FileNotFoundError: If model missing and DEMO_MODE not explicitly enabled.
        RuntimeError: If checkpoint corrupted or weights mismatch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        if is_demo_mode_active():
            logger.warning(
                "Model weights not found at %s. Demo mode explicitly enabled via %s=1.",
                model_path, DEMO_MODE_ENV
            )
            return None
        raise FileNotFoundError(
            f"Production safety error: Checkpoint file '{model_path}' does not exist. "
            f"To enable simulation for non-clinical testing, set {DEMO_MODE_ENV}=1."
        )

    try:
        model = NoduleCNN(in_channels=1, num_classes=num_classes)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info("Loaded nodule classification model from %s onto %s", model_path, device)
        return model

    except Exception as exc:
        logger.critical("Failed to load model weights: %s", exc)
        raise RuntimeError(f"Model initialization failed: {exc}") from exc


def load_model_meta(model_path: str = "model/best_model.pth") -> Dict[str, Any]:
    """Sidecar metadata written by scripts/train_lidc.py (best_model.meta.json); {} for legacy weights.
    input_mode == 'crop2d' means the model expects a patch_mm x patch_mm crop around ONE nodule."""
    meta_path = Path(model_path).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception as exc:
        logger.warning("Could not read model metadata %s: %s", meta_path, exc)
        return {}


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    hu_slice: Optional[np.ndarray] = None,
    crop: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Convert a CT slice into a [1, 1, H, W] float32 tensor using EXACTLY the
    same mapping as training (utils.image_utils.preprocess_hu_slice).

    Priority:
      1. hu_slice (true Hounsfield units)  -> exact.
      2. uint8 image                       -> assumed standard lung window,
                                              converted back to approx HU (lossy).
      3. float image                       -> interpreted as HU.
    """
    from utils.image_utils import preprocess_hu_slice, lung_window_uint8_to_hu

    if hu_slice is not None:
        hu = np.asarray(hu_slice, dtype=np.float32)
        if hu.ndim == 3:
            hu = hu[..., 0]
    else:
        if image is None or image.size == 0:
            raise ValueError("Invalid input image: array is empty or None")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if gray.ndim != 2:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        if gray.dtype == np.uint8:
            hu = lung_window_uint8_to_hu(gray)
        else:
            hu = gray.astype(np.float32)

    if crop is not None:
        # Crop-trained model: identical extraction to scripts/build_lidc_crops.py
        from utils.crops import crop_patch_2d
        from utils.image_utils import normalize_fixed_hu
        patch = crop_patch_2d(hu, crop["center_xy"], crop["spacing_mm"],
                              crop.get("patch_mm", 64.0), crop.get("out_px", 64))
        unit = normalize_fixed_hu(patch)
    else:
        unit = preprocess_hu_slice(hu, target_size)
    return torch.from_numpy(unit).unsqueeze(0).unsqueeze(0)


def equivalent_diameter_mm(mask: np.ndarray, spacing_mm: Tuple[float, float]) -> float:
    """
    Calculate 2D equivalent spherical diameter in millimeters from binary mask and real DICOM spacing:
        d = 2 * sqrt(Area_mm2 / pi)
    """
    if mask is None or mask.sum() == 0:
        return 0.0
    pixel_area = float(mask.sum())
    area_mm2 = pixel_area * float(spacing_mm[0]) * float(spacing_mm[1])
    return float(2.0 * np.sqrt(area_mm2 / np.pi))


def extract_features(
    image: np.ndarray,
    spacing_mm: Optional[Tuple[float, float]] = None,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    nodule_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Extract quantifiable morphological and physical nodule characteristics.
    Uses real DICOM physical spacing (row_spacing, col_spacing) when available.

    nodule_mask: binary mask of ONE segmented nodule (preferred). When given, all
    shape features come from that mask. When omitted, the legacy whole-image
    Otsu threshold is used (it measures whatever is brightest, usually NOT a
    nodule) and features['source'] says so.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot extract features from empty image")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    source = "nodule_mask" if nodule_mask is not None else "whole_image_otsu"
    # If a nodule mask is provided use it directly; else ROI / Otsu (legacy)
    if nodule_mask is not None:
        mask = (np.asarray(nodule_mask) > 0).astype(np.uint8) * 255
    elif roi_bbox is not None:
        rx, ry, rw, rh = roi_bbox
        sub_img = gray[ry:ry + rh, rx:rx + rw]
        _, thresh = cv2.threshold(sub_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[ry:ry + rh, rx:rx + rw] = thresh
    else:
        # Adaptive thresholding on lung parenchymal region
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            'source': source,
            'size_mm': None if (spacing_mm is None or nodule_mask is not None) else 0.0,
            'spacing_available': spacing_mm is not None,
            'sphericity': 0.0,
            'margin': 'indeterminate',
            'texture': 'homogeneous',
            'spiculation': False,
            'center': (gray.shape[1] // 2, gray.shape[0] // 2),
            'radius_px': 0.0,
            'size_pixels': 0
        }

    largest_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_contour))
    perimeter = float(cv2.arcLength(largest_contour, True))

    sphericity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    sphericity = float(np.clip(sphericity, 0.0, 1.0))

    M = cv2.moments(largest_contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = gray.shape[1] // 2, gray.shape[0] // 2

    hull = cv2.convexHull(largest_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else 1.0

    if solidity > 0.90:
        margin = 'smooth'
    elif solidity > 0.75:
        margin = 'lobulated'
    else:
        margin = 'irregular'

    spiculation = bool(solidity < 0.80 and sphericity < 0.70)

    # Physical size in millimeters
    if spacing_mm is not None and spacing_mm[0] > 0 and spacing_mm[1] > 0:
        nodule_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(nodule_mask, [largest_contour], -1, 1, -1)
        size_mm = round(equivalent_diameter_mm(nodule_mask, spacing_mm), 2)
        spacing_available = True
    else:
        size_mm = None
        spacing_available = False

    # Texture analysis within contour
    contour_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
    pixels = gray[contour_mask == 255]
    if len(pixels) > 0:
        std_intensity = float(np.std(pixels))
        if std_intensity < 20.0:
            texture = 'homogeneous'
        elif std_intensity < 45.0:
            texture = 'heterogeneous'
        else:
            texture = 'complex'
    else:
        texture = 'homogeneous'

    return {
        'source': source,
        'size_mm': size_mm,
        'spacing_available': spacing_available,
        'sphericity': round(sphericity, 3),
        'margin': margin,
        'texture': texture,
        'spiculation': spiculation,
        'solidity': round(solidity, 3),
        'center': (cx, cy),
        'radius_px': round(float(np.sqrt(area / np.pi)), 1),
        'size_pixels': int(area)
    }


@torch.inference_mode()
def predict_nodule(
    model: Optional[torch.nn.Module],
    image: np.ndarray,
    spacing_mm: Optional[Tuple[float, float]] = None,
    temperature: float = 1.0,
    hu_slice: Optional[np.ndarray] = None,
    nodule_mask: Optional[np.ndarray] = None,
    crop: Optional[Dict[str, Any]] = None,
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Execute validated neural inference on lung CT slice.

    Args:
        model: Evaluated PyTorch NoduleCNN instance
        image: Preprocessed 2D slice
        spacing_mm: Real DICOM pixel spacing (row_mm, col_mm)
        temperature: Temperature scaling factor for calibration (default 1.0)

    Returns:
        (prediction, confidence, features)
        prediction: 0 for Benign, 1 for Malignant
        confidence: Calibrated probability in [0.0, 1.0]
        features: Quantifiable morphological parameters
    """
    if image is None:
        raise ValueError("Cannot predict on None image")

    # 1. Feature extraction
    features = extract_features(image, spacing_mm=spacing_mm, nodule_mask=nodule_mask)

    # 2. Strict Check: If model is None, only proceed if explicit demo opt-in
    if model is None:
        if is_demo_mode_active():
            logger.warning("DEMO OPT-IN ACTIVE: Returning simulated prediction flagged as DEMO.")
            return _demo_prediction(features)
        raise RuntimeError(
            "Model is not loaded. Silent simulation is strictly prohibited in clinical mode. "
            "Load valid weights or set NODULE_DEMO=1 for testing."
        )

    # 3. Model Inference
    device = next(model.parameters()).device
    input_tensor = preprocess_image(image, hu_slice=hu_slice, crop=crop).to(device)

    logits = model(input_tensor)

    # Support single-logit binary classification or 2-class softmax
    if logits.shape[1] == 1:
        raw_val = logits.squeeze().item()
        features["raw_logit"] = float(raw_val)
        scaled_logit = logits.squeeze() / max(temperature, 1e-6)
        prob_malignant = torch.sigmoid(scaled_logit).item()
        pred_class = 1 if prob_malignant >= 0.5 else 0
        confidence = prob_malignant if pred_class == 1 else (1.0 - prob_malignant)
    else:
        scaled_logits = logits / max(temperature, 1e-6)
        probs = F.softmax(scaled_logits, dim=1)
        conf_t, pred_t = torch.max(probs, dim=1)
        pred_class = int(pred_t.item())
        confidence = float(conf_t.item())
        features["raw_logit"] = float(logits[0, pred_class].item())

    return pred_class, confidence, features


def _demo_prediction(features: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
    """
    Deterministic rule-based mock for testing environments only.
    Explicitly tags returned dictionary with DEMO warnings.
    """
    score = 0.0
    size_mm = features.get('size_mm')
    if size_mm is not None:
        if size_mm > 15:
            score += 0.35
        elif size_mm > 8:
            score += 0.20
    if features.get('margin') == 'irregular':
        score += 0.25
    if features.get('spiculation', False):
        score += 0.25

    pred = 1 if score >= 0.50 else 0
    confidence = min(0.92, max(0.60, 0.50 + score))

    features_copy = features.copy()
    features_copy["is_simulated"] = True
    features_copy["disclaimer"] = "DEMO – SIMULATED DATA, NOT A DIAGNOSIS"
    return pred, confidence, features_copy


def export_to_onnx(
    model: torch.nn.Module,
    save_path: str = "nodule.onnx",
    input_shape: Tuple[int, int, int, int] = (1, 1, 256, 256)
) -> str:
    """
    Export NoduleCNN model to ONNX runtime format for low-latency CPU/Edge deployment.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )
    logger.info("Exported ONNX model successfully to %s", save_path)
    return save_path
