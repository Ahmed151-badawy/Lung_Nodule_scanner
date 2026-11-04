"""
Tier 2: Model Calibration (Temperature Scaling / ECE) & High-Resolution Explainability
Axiomatic Attributions (Integrated Gradients) & Perturbation Saliency (Score-CAM)
"""

import os
import logging
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. TEMPERATURE SCALING & ECE
# ═══════════════════════════════════════════════════════════════

class ModelCalibrator(nn.Module):
    """
    Post-hoc Temperature Scaling to calibrate confidence probabilities.
    Learns single scalar T > 0 on validation set logits.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * float(temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature.clamp(min=1e-3)

    @torch.no_grad()
    def calibrate_probability(self, logit: torch.Tensor) -> float:
        scaled = self.forward(logit)
        return float(torch.sigmoid(scaled).item())

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.1, max_iter: int = 200):  # 0.01/50 can stop before converging
        """Fit optimal temperature using L-BFGS to minimize NLL on validation set."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        val_labels = val_labels.float()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = F.binary_cross_entropy_with_logits(scaled_logits.squeeze(), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        logger.info("Calibrated temperature parameter T = %.4f", float(self.temperature.item()))

    @classmethod
    def load_calibrator(cls, path: str = "model/calibrator.pth") -> "ModelCalibrator":
        """Load trained temperature weights from disk if available."""
        calibrator = cls()
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
                calibrator.load_state_dict(state)
                logger.info("Loaded calibrator from %s (T = %.3f)", path, calibrator.temperature.item())
            except Exception as exc:
                logger.warning("Could not load calibrator from %s: %s. Using default T=1.0", path, exc)
        else:
            logger.warning("Calibrator weights not found at %s. Using default T=1.0", path)
        return calibrator

    @staticmethod
    def calculate_ece(
        probabilities: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute Expected Calibration Error (ECE).
        ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        bin_details = []

        for i in range(num_bins):
            in_bin = (probabilities > bin_boundaries[i]) & (probabilities <= bin_boundaries[i + 1])
            prop_in_bin = float(np.mean(in_bin))
            if prop_in_bin > 0:
                acc_in_bin = float(np.mean(labels[in_bin]))
                avg_conf = float(np.mean(probabilities[in_bin]))
                ece += np.abs(acc_in_bin - avg_conf) * prop_in_bin
                bin_details.append({
                    "bin": (float(bin_boundaries[i]), float(bin_boundaries[i+1])),
                    "confidence": avg_conf,
                    "accuracy": acc_in_bin,
                    "weight": prop_in_bin
                })

        return float(ece), {"bins": bin_details}


# ═══════════════════════════════════════════════════════════════
# 2. SCORE-CAM (Gradient-Free Perturbation Activation Mapping)
# ═══════════════════════════════════════════════════════════════

class ScoreCAM:
    """
    Score-CAM: Fast, gradient-free visual explainability.
    Eliminates gradient noise and saturation artifacts.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or self._find_target_layer()
        if self.target_layer is None:
            raise ValueError("ScoreCAM target layer could not be resolved.")
        self.activations: Optional[torch.Tensor] = None

    def _find_target_layer(self) -> Optional[nn.Module]:
        target = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                target = module
        return target

    def _fwd_hook(self, module, inp, outp):
        self.activations = outp

    @torch.no_grad()
    def generate_cam(self, input_tensor: torch.Tensor, max_channels: int = 32) -> np.ndarray:
        """
        Generate score-weighted attribution map.
        Args:
            input_tensor: [1, C, H, W]
            max_channels: Top activation channels to evaluate for efficiency
        """
        self.model.eval()
        handle = self.target_layer.register_forward_hook(self._fwd_hook)

        # Baseline forward pass
        _ = self.model(input_tensor)
        handle.remove()

        if self.activations is None:
            raise RuntimeError("ScoreCAM hook failed to capture layer activations.")

        acts = self.activations[0]  # [Channels, Ha, Wa]
        c, ha, wa = acts.shape
        _, _, h, w = input_tensor.shape

        # Select top channels by activation variance to optimize compute
        variances = acts.view(c, -1).var(dim=1)
        top_k = min(c, max_channels)
        _, top_indices = torch.topk(variances, top_k)

        scores = []
        for idx in top_indices:
            channel_map = acts[idx:idx+1].unsqueeze(0)  # [1, 1, Ha, Wa]
            upsampled = F.interpolate(channel_map, size=(h, w), mode='bilinear', align_corners=False)
            norm_map = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)

            masked_input = input_tensor * norm_map
            score_logit = self.model(masked_input)
            scores.append(torch.sigmoid(score_logit.squeeze()).item())

        scores_t = torch.tensor(scores, device=input_tensor.device)
        weights = F.softmax(scores_t, dim=0).view(-1, 1, 1)

        selected_acts = acts[top_indices]
        cam = (weights * selected_acts).sum(dim=0)
        cam = F.relu(cam).cpu().numpy()

        cam_rescaled = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        norm_cam = (cam_rescaled - cam_rescaled.min()) / (cam_rescaled.max() - cam_rescaled.min() + 1e-8)
        return norm_cam


# ═══════════════════════════════════════════════════════════════
# 3. INTEGRATED GRADIENTS (Axiomatic Saliency)
# ═══════════════════════════════════════════════════════════════

class IntegratedGradients:
    """
    Computes path-integrated gradients from a baseline to satisfy Completeness and Implementation Invariance.
    Default baseline: Air intensity (-1000 HU mapped to 0.0).
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def attribute(
        self,
        input_tensor: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 30
    ) -> np.ndarray:
        self.model.eval()
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Linear path interpolation
        alphas = torch.linspace(0.0, 1.0, steps, device=input_tensor.device)
        delta = input_tensor - baseline

        interpolated_inputs = torch.cat([baseline + alpha * delta for alpha in alphas], dim=0)
        interpolated_inputs.requires_grad_(True)

        logits = self.model(interpolated_inputs)
        probs = torch.sigmoid(logits.squeeze())
        probs.sum().backward()

        grads = interpolated_inputs.grad  # [steps, C, H, W]
        avg_grads = grads.mean(dim=0, keepdim=True)  # [1, C, H, W]

        integrated_grad = (delta * avg_grads).squeeze(0)  # [C, H, W]
        saliency = integrated_grad.abs().sum(dim=0).detach().cpu().numpy()

        # Normalize to [0, 1]
        norm_saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return norm_saliency

