"""
End-to-End System Verification Script
Executes all components: OOD Quality Gate, Neural Inference, Grad-CAM, 
Fleischner 2017 Guidelines, Lineage Stamping, and Clinical PDF Report Generation.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import cv2
import torch

from model.model import NoduleCNN, load_model, predict_nodule
from explainability.gradcam import generate_gradcam
from expert_system.rules import apply_expert_rules
from preprocessing.volume_ood import CTQualityAndOODValidator
from reports.pdf_adapter import generate_pdf_report
from reports.lineage import ClinicalLineageTracker

def run_verification():
    print("============================================================")
    print("STARTING END-TO-END CLINICAL PIPELINE VERIFICATION")
    print("============================================================")

    # 1. Synthesize realistic thoracic CT slice in Hounsfield Units
    print("[1/5] Synthesizing calibrated thoracic CT slice...")
    slice_hu = np.full((512, 512), -1000.0, dtype=np.float32)  # ambient air (-1000 HU)
    # Body/chest wall envelope (+40 HU)
    cv2.ellipse(slice_hu, (256, 256), (220, 180), 0, 0, 360, 40.0, -1)
    # Ribs/spine bones (+400 HU)
    cv2.circle(slice_hu, (256, 400), 25, 400.0, -1)
    # Lung fields (-800 HU)
    cv2.ellipse(slice_hu, (170, 250), (75, 120), 0, 0, 360, -800.0, -1)
    cv2.ellipse(slice_hu, (342, 250), (75, 120), 0, 0, 360, -800.0, -1)
    # Incidental solitary nodule in right upper lobe (+60 HU)
    cv2.circle(slice_hu, (160, 230), 12, 60.0, -1)

    # 2. Check OOD / Input Quality Gate
    is_valid, ood_reasons = CTQualityAndOODValidator.validate_slice(slice_hu, {"modality": "CT"})
    print(f"      Quality Gate: Valid={is_valid} | Warnings={ood_reasons}")
    assert is_valid, f"Quality gate failed: {ood_reasons}"

    # Standard lung window representation for neural model & display
    from utils.image_utils import apply_lung_window
    windowed_uint8 = apply_lung_window(slice_hu)
    img = cv2.cvtColor(windowed_uint8, cv2.COLOR_GRAY2BGR)

    # 3. Model Inference & Morphometrics
    print("[2/5] Running Neural Inference & Morphometrics...")
    # Load the REAL checkpoint (falls back to random init only with an explicit warning)
    try:
        model = load_model("model/best_model.pth")
        print("      Loaded model/best_model.pth")
    except Exception as exc:
        print(f"      WARNING: could not load best_model.pth ({exc}); using RANDOM weights - outputs are meaningless")
        model = NoduleCNN(in_channels=1, num_classes=1)
        model.eval()
    spacing = (0.75, 0.75)  # 0.75 mm/pixel DICOM spacing
    pred_class, conf, features = predict_nodule(model, img, spacing_mm=spacing, hu_slice=slice_hu)
    equiv_diam = features.get("size_mm")
    print(f"      Prediction: {'MALIGNANT' if pred_class == 1 else 'BENIGN'}")
    print(f"      Calibrated Confidence: {conf:.1%}")
    print(f"      Equivalent Diameter: {equiv_diam} mm")
    print(f"      Margin Definition: {features.get('margin')}")

    # 4. Grad-CAM Visualization
    print("[3/5] Generating Grad-CAM Saliency Map...")
    cam_overlay = generate_gradcam(model, img, target_class=pred_class, hu_slice=slice_hu)
    print(f"      CAM Overlay Generated: shape={cam_overlay.shape}, dtype={cam_overlay.dtype}")

    # 5. Fleischner Society 2017 Decision Support
    print("[4/5] Applying Fleischner Society 2017 Guidelines...")
    expert = apply_expert_rules(pred_class, conf, features, is_high_risk=True)
    tier = expert.get("fleischner_details", {}).get("tier", "N/A")
    rec = expert.get("recommendation", "N/A")
    interval = expert.get("follow_up_interval", "N/A")
    print(f"      Category Tier: {tier}")
    print(f"      Surveillance Cadence: {interval}")
    print(f"      Actionable Recommendation: {rec}")

    # 6. Anatomical Nodule Segmentation (RECIST 1.1) & Brock PanCan Risk
    print("[5/7] Computing RECIST 1.1 Dual Axes & Brock Malignancy Risk...")
    from preprocessing.nodule_segmenter import AnatomicalNoduleSegmenter
    from expert_system.brock_model import BrockUniversityRiskCalculator
    segmenter = AnatomicalNoduleSegmenter()
    mask = segmenter.segment_candidate_roi(slice_hu, seed_point=(160, 230), spacing_mm=spacing)
    recist = segmenter.measure_recist_11(mask, slice_hu, spacing)
    # Ground truth: drawn nodule radius 12 px * 0.75 mm -> ~18 mm diameter
    assert recist is not None, "Segmenter found no nodule"
    assert 15.0 <= recist.longest_diameter_mm <= 21.0, \
        f"Implausible nodule size {recist.longest_diameter_mm} mm (expected ~18 mm)"
    assert not recist.warnings, f"Measurement warnings: {recist.warnings}"
    if recist:
        print(f"      RECIST Axes: {recist.longest_diameter_mm} x {recist.perpendicular_diameter_mm} mm (mean: {recist.mean_recist_diameter_mm} mm)")
        print(f"      Consolidation Ratio (CTR): {recist.consolidation_to_tumor_ratio} ({recist.attenuation_type})")

    brock = BrockUniversityRiskCalculator.calculate_risk(
        size_mm=recist.longest_diameter_mm,
        age=64,
        is_female=True,
        family_history_lung_cancer=False,
        emphysema_present=False,
        nodule_type=recist.attenuation_type if recist else "solid",
        is_upper_lobe=True,
        nodule_count=1,
        is_spiculated=bool(features.get('spiculation', False))
    )
    print(f"      Brock Malignancy Risk: {brock.risk_percentage}% ({brock.risk_tier})")

    # 7. Clinical PDF Report Generation with Lineage
    print("[6/7] Exporting Clinical PDF Report with Lineage Stamping...")
    lineage = ClinicalLineageTracker.generate_provenance_stamp(
        dicom_metadata={"patient_id": "LIDC_0042", "series_instance_uid": "1.2.840.113619.2.55.3.60468838"}
    )
    analysis_result = {
        "prediction": "MALIGNANT" if pred_class == 1 else "BENIGN",
        "confidence": conf,
        "features": features,
        "expert_analysis": expert,
        "gradcam": cam_overlay,
        "recist": recist,
        "brock": brock
    }
    patient_info = {
        "id": "LIDC_0042",
        "name": "Jane Doe",
        "age": 64,
        "gender": "F",
        "scan_date": "2026-11-03",
        "date": "2026-11-03",
        "lineage": lineage
    }
    out_dir = Path("dist")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = str(out_dir / "e2e_verification_report.pdf")

    success = generate_pdf_report(pdf_path, analysis_result, img, patient_info)
    assert success, "PDF generation failed"
    pdf_size_kb = Path(pdf_path).stat().st_size / 1024.0
    print(f"      PDF Report Saved: {pdf_path} ({pdf_size_kb:.1f} KB)")
    print(f"      Lineage Signature: {lineage.get('audit_hash')}")

    # 8. DICOM Structured Reporting Part 10 Generation
    print("[7/7] Generating DICOM Basic Text SR...")
    from reports.dicom_sr import create_dicom_sr_report
    sr_path = str(out_dir / "e2e_dicom_sr.dcm")
    create_dicom_sr_report(sr_path, analysis_result, {"patient_id": "LIDC_0042", "patient_name": "Jane Doe"})
    sr_size_kb = Path(sr_path).stat().st_size / 1024.0
    print(f"      DICOM SR Saved: {sr_path} ({sr_size_kb:.1f} KB)")

    print("============================================================")
    print("E2E SMOKE TEST PASSED (synthetic slice; not clinical validation)")
    print("============================================================")

if __name__ == "__main__":
    run_verification()
