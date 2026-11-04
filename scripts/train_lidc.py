"""
Train NoduleCNN on LIDC patches (from scripts/build_lidc_crops.py).

Protocol: patient-disjoint train/val/test. Val picks the checkpoint (best AUROC) and fits
temperature scaling; test is evaluated once at the end with patient-bootstrap CIs.
Also reports a diameter-only baseline: if the CNN cannot beat "bigger = more malignant",
it has learned nothing beyond size.
Outputs: model/best_model.pth, model/calibrator.pth, model/best_model.meta.json, model/splits.json

*** Not run against real data by the author of this patch; expect to tune hyper-parameters. ***
"""
import argparse, copy, json, random, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.model import NoduleCNN
from data.dataset import patient_level_split_3way
from data.crop_dataset import NoduleCropDataset, read_manifest
from explainability.calibration_and_saliency import ModelCalibrator
from utils.metrics import (auroc, sensitivity_at_specificity, expected_calibration_error,
                           brier, grouped_bootstrap_ci)


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    logits, ys = [], []
    for x, y, _ in loader:
        logits.append(model(x.to(device)).squeeze(-1).cpu())
        ys.append(y.squeeze(-1).cpu())
    return torch.cat(logits), torch.cat(ys).numpy().astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/lidc_crops")
    ap.add_argument("--out", default="model")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info_path = Path(args.crops) / "build_info.json"
    build_info = json.loads(info_path.read_text())
    rows = read_manifest(args.crops)
    tr, va, te = patient_level_split_3way(rows, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
    for name, g in (("train", tr), ("val", va), ("test", te)):
        print(f"{name:<5}: {len({r['patient_id'] for r in g}):4d} patients, {len(g):5d} nodules, "
              f"{sum(r['label'] for r in g):4d} malignant")

    mk = lambda ds, shuf: DataLoader(ds, batch_size=args.batch, shuffle=shuf, num_workers=args.workers)
    tr_loader = mk(NoduleCropDataset(tr, augment=True), True)
    va_loader = mk(NoduleCropDataset(va), False)
    te_loader = mk(NoduleCropDataset(te), False)

    n_pos = max(1, sum(r["label"] for r in tr)); n_neg = max(1, len(tr) - n_pos)
    model = NoduleCNN(in_channels=1, num_classes=1, dropout_rate=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=device))
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_auc, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = criterion(model(x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item() * len(x)
        sched.step()
        v_logits, v_y = collect(model, va_loader, device)
        v_auc = auroc(v_y, torch.sigmoid(v_logits).numpy())
        print(f"epoch {ep:02d}  train loss {tl / len(tr):.4f}  val AUROC {v_auc:.3f}")
        if v_auc > best_auc:
            best_auc, best_ep, bad = v_auc, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping"); break

    model.load_state_dict(best_state)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "best_model.pth")

    v_logits, v_y = collect(model, va_loader, device)
    cal = ModelCalibrator(temperature=1.0); cal.fit(v_logits, torch.from_numpy(v_y).float())
    torch.save(cal.state_dict(), out / "calibrator.pth")
    print(f"best epoch {best_ep} (val AUROC {best_auc:.3f}); temperature T = {cal.temperature.item():.3f}")

    t_logits, t_y = collect(model, te_loader, device)
    with torch.no_grad():
        t_p = torch.sigmoid(cal(t_logits)).numpy()
    groups = np.array([r["patient_id"] for r in te])
    diam = np.array([r["diameter_mm"] for r in te])

    metrics = {}
    def report(name, fn, scores):
        lo, hi = grouped_bootstrap_ci(fn, t_y, scores, groups, n_boot=1000)
        metrics[name] = {"value": fn(t_y, scores), "ci95": [lo, hi]}
        print(f"  {name:<28} {metrics[name]['value']:.3f}  (95% CI {lo:.3f}-{hi:.3f})")

    print(f"TEST ({len(set(groups))} patients, {len(t_y)} nodules, {int(t_y.sum())} malignant):")
    report("AUROC (CNN)", auroc, t_p)
    report("AUROC (diameter only)", auroc, diam)
    report("Sens @ 90% spec (CNN)", sensitivity_at_specificity, t_p)
    report("ECE (calibrated)", expected_calibration_error, t_p)
    report("Brier", brier, t_p)
    if metrics["AUROC (CNN)"]["value"] <= metrics["AUROC (diameter only)"]["value"]:
        print("  WARNING: the CNN does not beat a diameter-only baseline on this test set.")

    meta = {
        "input_mode": "crop2d", "patch_mm": build_info["patch_mm"], "out_px": build_info["out_px"],
        "in_channels": 1, "hu_norm_range": [-1000.0, 400.0],
        "data": "LIDC-IDRI via pylidc", "label_definition": build_info["label_definition"],
        "n_train": len(tr), "n_val": len(va), "n_test": len(te), "best_epoch": best_ep, "seed": args.seed,
        "temperature": float(cal.temperature.item()), "test_metrics": metrics,
        "git_sha": _git_sha(), "torch": torch.__version__,
        "created": datetime.now(timezone.utc).isoformat(),
        "intended_use": "RESEARCH ONLY. Not validated for clinical use.",
    }
    (out / "best_model.meta.json").write_text(json.dumps(meta, indent=2))
    (out / "splits.json").write_text(json.dumps({
        "train": sorted({r["patient_id"] for r in tr}), "val": sorted({r["patient_id"] for r in va}),
        "test": sorted({r["patient_id"] for r in te})}, indent=2))
    print(f"saved weights, calibrator, meta and splits to {out}/")


if __name__ == "__main__":
    main()
