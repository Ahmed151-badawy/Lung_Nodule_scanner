"""
Build nodule patches + manifest from LIDC-IDRI via pylidc.

*** NOT TESTED AGAINST REAL LIDC DATA (written without access to it). ***
First run:   python scripts/build_lidc_crops.py --limit 3 --preview 12
then LOOK at <out>/preview.png: nodules centred, ~1 mm/pixel, labels plausible.

Setup:  pip install pylidc ;  create ~/.pylidcrc (Windows: %USERPROFILE%\\pylidc.conf) containing
            [dicom]
            path = /path/to/LIDC-IDRI
Output: <out>/patches/*.npy (int16 HU, out_px x out_px, patch_mm wide), <out>/manifest.csv, <out>/build_info.json
"""
import argparse, csv, json, sys, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2

from data.lidc_labels import consensus_label
from utils.crops import crop_patch_2d
from utils.image_utils import apply_lung_window

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_lidc_crops")

FIELDS = ["patient_id", "nodule_id", "series_uid", "label", "median_malignancy", "mean_malignancy",
          "n_annotators", "diameter_mm", "median_spiculation", "pixel_spacing_mm", "slice_idx",
          "x", "y", "patch_path"]


def load_hu_volume(scan):
    """HU volume [rows, cols, slices], computed from raw DICOM with rescale slope/intercept."""
    images = scan.load_all_dicom_images(verbose=False)       # sorted by slice position
    vol = np.stack(
        [im.pixel_array.astype(np.float32) * float(getattr(im, "RescaleSlope", 1.0))
         + float(getattr(im, "RescaleIntercept", 0.0)) for im in images],
        axis=-1,
    )
    return vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lidc_crops")
    ap.add_argument("--patch_mm", type=float, default=64.0)
    ap.add_argument("--out_px", type=int, default=64)
    ap.add_argument("--min_annotators", type=int, default=3)
    ap.add_argument("--min_diameter_mm", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0, help="only first N scans (0 = all)")
    ap.add_argument("--preview", type=int, default=0, help="write preview.png with first N patches")
    args = ap.parse_args()

    import pylidc as pl                                       # imported late: heavy / optional
    scans = sorted(pl.query(pl.Scan).all(), key=lambda s: (s.patient_id, s.series_instance_uid))
    if args.limit:
        scans = scans[:args.limit]

    out = Path(args.out); (out / "patches").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], {"few_annotators_or_ambiguous": 0, "too_small": 0, "bad_slice": 0, "scan_error": 0}

    for si, scan in enumerate(scans, 1):
        try:
            vol = load_hu_volume(scan)
            sp = float(scan.pixel_spacing)                    # pylidc: isotropic in-plane spacing (mm)
            clusters = scan.cluster_annotations(verbose=False)
        except Exception as exc:
            log.warning("scan %s failed: %s", scan.patient_id, exc); skipped["scan_error"] += 1; continue

        for ni, anns in enumerate(clusters):
            lab = consensus_label([a.malignancy for a in anns], args.min_annotators)
            if lab is None:
                skipped["few_annotators_or_ambiguous"] += 1; continue
            diam = float(np.mean([a.diameter for a in anns]))
            if diam < args.min_diameter_mm:
                skipped["too_small"] += 1; continue
            c = np.mean([np.asarray(a.centroid, dtype=float) for a in anns], axis=0)   # (row, col, slice)
            row_i, col_j, k = float(c[0]), float(c[1]), int(round(c[2]))
            if not (0 <= k < vol.shape[2]):
                skipped["bad_slice"] += 1; continue

            patch = crop_patch_2d(vol[:, :, k], (col_j, row_i), (sp, sp), args.patch_mm, args.out_px)
            name = f"{scan.patient_id}_n{ni:02d}.npy"
            np.save(out / "patches" / name, np.clip(patch, -1024, 3071).astype(np.int16))
            rows.append({
                "patient_id": scan.patient_id, "nodule_id": f"n{ni:02d}", "series_uid": scan.series_instance_uid,
                "label": lab["label"], "median_malignancy": lab["median_malignancy"],
                "mean_malignancy": round(lab["mean_malignancy"], 3), "n_annotators": lab["n_annotators"],
                "diameter_mm": round(diam, 2),
                "median_spiculation": float(np.median([a.spiculation for a in anns])),
                "pixel_spacing_mm": sp, "slice_idx": k, "x": round(col_j, 1), "y": round(row_i, 1),
                "patch_path": f"patches/{name}",
            })
        log.info("[%d/%d] %s -> %d nodules so far", si, len(scans), scan.patient_id, len(rows))

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

    n_mal = sum(r["label"] for r in rows)
    info = {"patch_mm": args.patch_mm, "out_px": args.out_px, "min_annotators": args.min_annotators,
            "min_diameter_mm": args.min_diameter_mm, "n_nodules": len(rows), "n_malignant": n_mal,
            "n_patients": len({r["patient_id"] for r in rows}), "skipped": skipped,
            "label_definition": "median radiologist malignancy rating: <3 benign, >3 malignant, ==3 excluded "
                                "(perceived malignancy, NOT pathology)"}
    (out / "build_info.json").write_text(json.dumps(info, indent=2))
    log.info("done: %s", info)

    if args.preview and rows:
        tiles = []
        for r in rows[:args.preview]:
            p = np.load(out / r["patch_path"]).astype(np.float32)
            g = apply_lung_window(p)
            g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g.ndim == 2 else g
            g = cv2.resize(g, (192, 192), interpolation=cv2.INTER_NEAREST)
            cv2.putText(g, f"{'MAL' if r['label'] else 'BEN'} {r['diameter_mm']}mm", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(g)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(out / "preview.png"), grid)
        log.info("wrote %s", out / "preview.png")


if __name__ == "__main__":
    main()
