"""
Phase 1: Production Training & Calibration Harness
Trains NoduleCNN on Patient-Stratified Data, Fits Temperature Calibration,
and Exports Production Weights to model/best_model.pth
"""

import sys
import os
import time
import copy
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.model import NoduleCNN
from data.dataset import LungNoduleDataset, patient_level_split_3way
from utils.metrics import (auroc, sensitivity_at_specificity, expected_calibration_error,
                           brier, grouped_bootstrap_ci)
from explainability.calibration_and_saliency import ModelCalibrator


def synthesize_clinical_cohort(n_patients: int = 40, samples_per_patient: int = 4):
    """
    Synthesize realistic radiologically calibrated thoracic CT samples
    representing benign (smooth, high sphericity, small) and malignant (spiculated, irregular, large) nodules.
    """
    samples = []
    rng = np.random.RandomState(1337)

    for p_idx in range(n_patients):
        pid = f"LIDC_CASE_{p_idx+1:04d}"
        # Patient-level baseline malignancy tendency
        patient_malignant = rng.rand() > 0.5

        for s_idx in range(samples_per_patient):
            is_mal = patient_malignant if rng.rand() > 0.15 else not patient_malignant

            # Slices [z-1, z, z+1]
            slices = []
            diam = rng.uniform(10.0, 24.0) if is_mal else rng.uniform(4.0, 9.0)

            for z_offset in [-1, 0, 1]:
                sl = np.full((256, 256), -1000.0, dtype=np.float32)  # air
                cv2.ellipse(sl, (128, 128), (110, 90), 0, 0, 360, 40.0, -1)  # body
                cv2.ellipse(sl, (85, 125), (38, 60), 0, 0, 360, -800.0, -1)  # lung left
                cv2.ellipse(sl, (171, 125), (38, 60), 0, 0, 360, -800.0, -1)  # lung right

                # Nodule in lung
                rad = max(2, int((diam / 2.0) / 0.75) + z_offset)
                center = (80 + rng.randint(-3, 3), 120 + rng.randint(-3, 3))

                if is_mal:
                    # Irregular spiculated nodule
                    cv2.circle(sl, center, rad, 60.0, -1)
                    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                        px = int(center[0] + (rad + 4) * np.cos(np.radians(angle)))
                        py = int(center[1] + (rad + 4) * np.sin(np.radians(angle)))
                        cv2.line(sl, center, (px, py), 60.0, 2)
                else:
                    # Smooth round benign nodule
                    cv2.circle(sl, center, rad, 30.0, -1)

                # Sensor noise
                sl += rng.normal(0, 15.0, sl.shape).astype(np.float32)
                slices.append(sl)

            samples.append({
                "patient_id": pid,
                "slices": slices,
                "label": 1 if is_mal else 0,
                "diameter_mm": diam
            })

    return samples


def train_production_weights(
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_dir: str = "model"
):
    print("============================================================")
    print("PHASE 1: TRAINING BASELINE WEIGHTS (SYNTHETIC DATA - NOT CLINICAL)")
    print("============================================================")

    # 1. Generate & Split Cohort
    print("[1/6] Ingesting cohort and partitioning patient-level splits...")
    cohort = synthesize_clinical_cohort(n_patients=50, samples_per_patient=4)
    train_samples, val_samples, test_samples = patient_level_split_3way(
        cohort, val_ratio=0.2, test_ratio=0.2, seed=101)

    train_ds = LungNoduleDataset(train_samples, is_25d=False)
    val_ds = LungNoduleDataset(val_samples, is_25d=False)
    test_ds = LungNoduleDataset(test_samples, is_25d=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Architecture & Optimizer
    print("[2/6] Initializing NoduleCNN architecture & optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoduleCNN(in_channels=1, num_classes=1, dropout_rate=0.3).to(device)

    # Plain BCE (no class weighting). Add pos_weight=... if classes are imbalanced.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training Loop
    print(f"[3/6] Training for {epochs} epochs on {device}...")
    best_val_loss = float("inf")
    best_weights = None

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(x)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(y.squeeze(-1).cpu().numpy())

        val_loss /= len(val_ds)
        val_acc = np.mean((np.array(val_preds) >= 0.5) == np.array(val_targets))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

        print(f"      Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1%}")

    # 4. Save Best Production Weights
    print("[4/6] Saving validated checkpoint to model/best_model.pth...")
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pth"

    torch.save(best_weights, best_path)
    file_size_kb = best_path.stat().st_size / 1024.0
    print(f"      [OK] Saved production weights: {best_path} ({file_size_kb:.1f} KB)")

    # 5. Fit & Save Model Calibration
    print("[5/6] Fitting post-hoc Temperature Calibration on validation logits...")
    model.load_state_dict(best_weights)
    model.eval()

    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device)
            logits = model(x)
            val_logits_list.append(logits.squeeze(-1).cpu())
            val_labels_list.append(y.squeeze(-1).cpu())

    all_logits = torch.cat(val_logits_list, dim=0)
    all_labels = torch.cat(val_labels_list, dim=0)

    calibrator = ModelCalibrator(temperature=1.0)
    calibrator.fit(all_logits, all_labels)
    cal_path = out_dir / "calibrator.pth"
    torch.save(calibrator.state_dict(), cal_path)
    print(f"      [OK] Calibrated Temperature T = {calibrator.temperature.item():.3f} saved to {cal_path}")

    # Validation-set ECE is optimistic (T was fit on this very data) - informational only
    with torch.no_grad():
        val_probs = torch.sigmoid(calibrator(all_logits)).cpu().numpy()
    print(f"      [info] ECE on validation set (used to fit T, optimistic): {expected_calibration_error(all_labels.numpy(), val_probs):.2%}")

    # 6. HELD-OUT TEST: patient-disjoint, never used for training, checkpoint choice or T
    print("[6/6] Evaluating on held-out test patients...")
    test_logits, test_labels = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            test_logits.append(model(x.to(device)).squeeze(-1).cpu())
            test_labels.append(y.squeeze(-1).cpu())
    t_logits = torch.cat(test_logits)
    t_y = torch.cat(test_labels).numpy().astype(int)
    with torch.no_grad():
        t_p = torch.sigmoid(calibrator(t_logits)).cpu().numpy()
    t_groups = np.array([sm["patient_id"] for sm in test_samples])   # loader order == sample order (shuffle=False)

    def _line(name, fn):
        lo, hi = grouped_bootstrap_ci(fn, t_y, t_p, t_groups, n_boot=500)
        print(f"      {name:<22} {fn(t_y, t_p):.3f}  (95% CI {lo:.3f}-{hi:.3f}, patient bootstrap)")

    print(f"      Test set: {len(set(t_groups))} patients, {len(t_y)} nodules, {int(t_y.sum())} malignant")
    _line("AUROC", auroc)
    _line("Sens @ 90% spec", sensitivity_at_specificity)
    _line("ECE (10 bins)", expected_calibration_error)
    _line("Brier score", brier)

    print("============================================================")
    print("PHASE 1 COMPLETE (SYNTHETIC DATA): pipeline smoke test only.")
    print("These weights and the calibration/ECE above say NOTHING about clinical performance.")
    print("============================================================")


if __name__ == "__main__":
    train_production_weights()
