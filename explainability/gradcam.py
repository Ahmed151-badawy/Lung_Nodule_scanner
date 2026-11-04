"""
Grad-CAM & Grad-CAM++ Explainability Module
Device-Safe, Hook-Leak Proof Visual Saliency for Clinical CT Inspection
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEMO_MODE_ENV = "NODULE_DEMO"


def is_demo_mode_active() -> bool:
    return os.getenv(DEMO_MODE_ENV, "0").strip().lower() in ("1", "true", "yes")


class GradCAM:
    """
    Production-grade Grad-CAM context manager.
    Guarantees hook removal on exit to prevent memory and tensor leaks.
    Maintains all gradient and tensor calculations on the native device (CPU/CUDA).
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or self._find_target_layer()
        if self.target_layer is None:
            raise ValueError("Target layer for Grad-CAM could not be resolved in the model.")

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    def _find_target_layer(self) -> Optional[nn.Module]:
        """Find the final Conv2d layer in the model hierarchy."""
        target = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                target = module
        return target

    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        self.activations = output

    def _backward_hook(self, module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
        # grad_output[0] shape: [B, C, H, W]
        self.gradients = grad_output[0]

    def register_hooks(self):
        """Register forward and full-backward hooks."""
        self.remove_hooks()
        h_fwd = self.target_layer.register_forward_hook(self._forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(self._backward_hook)
        self._hook_handles.extend([h_fwd, h_bwd])

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map.
        Args:
            input_tensor: [1, C, H, W] tensor on model device
            target_class: Target index. If None, uses argmax of logits
        Returns:
            2D numpy array [H, W] normalized to [0, 1]
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        if logits.dim() == 1 or logits.shape[1] == 1:
            score = logits.squeeze()
        else:
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            score = logits[0, target_class]

        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Compute weights via Global Average Pooling over spatial dimensions
        # Keep computation on the native device of activations!
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze(0)  # [H, W] on device

        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / (cam_max + 1e-8)

        return cam.detach().cpu().numpy()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ formulation: Higher-order gradient weighting for complex, multi-focal textures.
    """

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        if logits.dim() == 1 or logits.shape[1] == 1:
            score = logits.squeeze()
        else:
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            score = logits[0, target_class]

        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM++ hooks failed to capture tensors.")

        grads = self.gradients  # [1, C, H, W]
        acts = self.activations  # [1, C, H, W]

        # 2nd and 3rd order gradients
        grads_power_2 = grads.pow(2)
        grads_power_3 = grads.pow(3)

        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        eps = 1e-8
        aij = grads_power_2 / (2.0 * grads_power_2 + sum_acts * grads_power_3 + eps)
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))

        weights = (aij * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1)).squeeze(0)

        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / (cam_max + 1e-8)

        return cam.detach().cpu().numpy()


def apply_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay saliency heatmap onto input CT slice.
    Args:
        image: Original CT image (2D or 3-channel BGR/RGB)
        cam: 2D numpy array [0, 1]
        alpha: Blending weight
        colormap: OpenCV colormap
    Returns:
        3-channel BGR uint8 image with superimposed heatmap
    """
    if image.ndim == 2:
        base_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        base_img = cv2.cvtColor(image.squeeze(-1), cv2.COLOR_GRAY2BGR)
    else:
        base_img = image.copy()

    h, w = base_img.shape[:2]
    # Resize CAM to match original image dimensions
    heatmap = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Superimpose
    overlay = cv2.addWeighted(heatmap_color, alpha, base_img, 1.0 - alpha, 0)
    return overlay


def generate_gradcam(
    model: Optional[torch.nn.Module],
    image: np.ndarray,
    target_class: Optional[int] = None,
    use_plus_plus: bool = False,
    hu_slice: Optional[np.ndarray] = None,
    crop: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Produce validated Grad-CAM saliency overlay.
    Fails loudly with clear diagnostics. Never produces fabricated heatmaps in clinical mode.
    """
    if image is None:
        raise ValueError("Cannot compute Grad-CAM on empty image")

    if model is None:
        if is_demo_mode_active():
            logger.warning("DEMO OPT-IN ACTIVE: Returning synthetic heatmap stamped for DEMO.")
            return _generate_demo_gradcam(image)
        raise RuntimeError(
            "Grad-CAM generation failed: Model is not loaded. "
            "To enable synthetic visualizations for UI smoke testing, set NODULE_DEMO=1."
        )

    from model.model import preprocess_image

    device = next(model.parameters()).device
    input_tensor = preprocess_image(image, hu_slice=hu_slice, crop=crop).to(device)

    cam_cls = GradCAMPlusPlus if use_plus_plus else GradCAM
    with cam_cls(model) as explainer:
        cam_map = explainer.generate_cam(input_tensor, target_class=target_class)

    base = image
    if crop is not None:
        # Overlay on the patch the model actually saw, not on the full slice
        from utils.crops import crop_patch_2d
        from utils.image_utils import apply_lung_window, lung_window_uint8_to_hu
        hu = hu_slice if hu_slice is not None else lung_window_uint8_to_hu(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image)
        patch = crop_patch_2d(hu, crop["center_xy"], crop["spacing_mm"],
                              crop.get("patch_mm", 64.0), crop.get("out_px", 64))
        g = apply_lung_window(patch)
        base = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g.ndim == 2 else g
        base = cv2.resize(base, (256, 256), interpolation=cv2.INTER_LINEAR)

    visualization = apply_cam_on_image(base, cam_map)
    return visualization


def _generate_demo_gradcam(image: np.ndarray) -> np.ndarray:
    """
    Deterministic synthetic Gaussian heatmap strictly for demonstration UI smoke testing.
    Stamps an overt simulated watermark to prevent clinical confusion.
    """
    h, w = image.shape[:2]
    x = np.linspace(-2, 2, w)
    y = np.linspace(-2, 2, h)
    xx, yy = np.meshgrid(x, y)
    simulated_cam = np.exp(-(xx**2 + yy**2) / 0.5)
    simulated_cam = (simulated_cam - simulated_cam.min()) / (simulated_cam.max() - simulated_cam.min() + 1e-8)

    overlay = apply_cam_on_image(image, simulated_cam, alpha=0.45)
    # Visible clinical watermark
    cv2.putText(
        overlay,
        "DEMO - SIMULATED HEATMAP",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )
    return overlay
