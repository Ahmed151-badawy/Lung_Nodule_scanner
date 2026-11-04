"""
Tier 3: Production Medical Microservice (FastAPI)
Exposing /predict, /explain, and /health Endpoints
Response schemas and provenance stamping (research prototype)
"""

import os
import io
import time
import uuid
import base64
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from model.model import load_model, load_model_meta, predict_nodule, is_demo_mode_active
from explainability.gradcam import generate_gradcam
from explainability.calibration_and_saliency import ModelCalibrator, ScoreCAM
from expert_system.rules import apply_expert_rules, CLINICAL_DISCLAIMER
from expert_system.brock_model import BrockUniversityRiskCalculator, BrockRiskResult
from preprocessing.volume_ood import CTQualityAndOODValidator
from preprocessing.nodule_segmenter import AnatomicalNoduleSegmenter, RECISTMeasurement
from reports.lineage import ClinicalLineageTracker
from utils.image_utils import convert_to_hounsfield_units

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PulmonaryCADxService")

@asynccontextmanager
async def lifespan(_app):
    startup_event()   # defined below; resolved at call time
    yield


app = FastAPI(
    title="Pulmonary Nodule Research Service",
    description="RESEARCH PROTOTYPE - not for diagnostic use. Nodule measurement, model output, "
                "Brock estimate and Fleischner-2017 rule lookup.",
    version="2.6.0",
    lifespan=lifespan
)

# Global Microservice State
SERVICE_STATE: Dict[str, Any] = {
    "model": None,
    "model_meta": {},
    "calibrator": ModelCalibrator(temperature=1.0),
    "model_path": "model/best_model.pth",
    "calibrator_path": "model/calibrator.pth",
    "ready": False
}


def startup_event():
    """Initialize neural models, load temperature calibrator, and compute immutable SHA hashes."""
    try:
        SERVICE_STATE["model"] = load_model(SERVICE_STATE["model_path"])
        SERVICE_STATE["model_meta"] = load_model_meta(SERVICE_STATE["model_path"])
        SERVICE_STATE["calibrator"] = ModelCalibrator.load_calibrator(SERVICE_STATE["calibrator_path"])
        SERVICE_STATE["ready"] = True
        logger.info("Model & calibrator loaded successfully into microservice.")
    except Exception as exc:
        if is_demo_mode_active():
            logger.warning("Starting service in DEMO_MODE: %s", exc)
        else:
            logger.error("Failed to load model on startup: %s", exc)


# ═══════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════

class FindingMeasurement(BaseModel):
    equivalent_diameter_mm: Optional[float] = Field(None, description="2D equivalent spherical diameter in mm")
    sphericity_index: float = Field(..., description="Circularity index in [0, 1]")
    margin_classification: str = Field(..., description="smooth, lobulated, or irregular")
    spiculation_present: bool = Field(..., description="True if spiculation detected")
    internal_texture: str = Field(..., description="homogeneous, heterogeneous, or complex")


class RECISTMeasurementSR(BaseModel):
    longest_diameter_mm: Optional[float] = Field(None, description="RECIST 1.1 Longest Axial Diameter (d_long)")
    perpendicular_diameter_mm: Optional[float] = Field(None, description="RECIST 1.1 Perpendicular Axis (d_perp)")
    mean_recist_diameter_mm: Optional[float] = Field(None, description="Mean axial diameter (d_long + d_perp) / 2")
    equivalent_diameter_mm: Optional[float] = Field(None, description="2D equivalent spherical diameter")
    consolidation_to_tumor_ratio: Optional[float] = Field(None, description="CTR ratio (solid core / total)")
    attenuation_type: str = Field("solid", description="solid, part_solid, or ground_glass")
    estimated_volume_mm3: Optional[float] = Field(None, description="Spheroid volume approximation")
    warnings: List[str] = Field(default_factory=list, description="Measurement plausibility warnings")


class BrockRiskSR(BaseModel):
    model_version: str = "PanCan / Brock Full Model (NEJM 2013)"
    malignancy_probability: float = Field(..., description="Malignancy probability in [0, 1]")
    risk_percentage: float = Field(..., description="Malignancy risk percentage [0, 100]%")
    risk_tier: str = Field(..., description="VERY_LOW, LOW, INTERMEDIATE, or HIGH")
    clinical_guidance: str = Field(..., description="Evidence-based guidance")


class FleischnerGuidelineSR(BaseModel):
    guideline_name: str = "Fleischner Society 2017"
    category_tier: str
    surveillance_interval: str
    actionable_recommendation: str


class DiagnosticPayload(BaseModel):
    request_id: str
    timestamp: str
    lineage: Dict[str, str]
    prediction: str
    confidence: float
    calibrated_probability: float
    measurements: FindingMeasurement
    recist_measurements: Optional[RECISTMeasurementSR] = None
    brock_risk: Optional[BrockRiskSR] = None
    expert_guidelines: FleischnerGuidelineSR
    ood_validation: Dict[str, Any]
    regulatory_disclaimer: str


class ExplainResponse(BaseModel):
    request_id: str
    target_class: int
    method: str
    saliency_png_base64: str
    lineage: Dict[str, str]


# ═══════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """Readiness probe: HTTP 503 when the model is not loaded (so orchestrators stop routing traffic)."""
    lineage = ClinicalLineageTracker.generate_provenance_stamp(
        model_path=SERVICE_STATE["model_path"]
    )
    ready = bool(SERVICE_STATE["ready"])
    body = {
        "status": "HEALTHY" if ready else "DEGRADED",
        "model_loaded": SERVICE_STATE["model"] is not None,
        "demo_mode": is_demo_mode_active(),
        "lineage": lineage,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)


@app.post("/predict", response_model=DiagnosticPayload)
async def predict_nodule_endpoint(
    file: UploadFile = File(...),
    pixel_spacing_row: Optional[float] = Form(None),
    pixel_spacing_col: Optional[float] = Form(None),
    patient_age: Optional[int] = Form(60),
    is_female: bool = Form(False),
    family_history_lung_cancer: bool = Form(False),
    emphysema_present: bool = Form(False),
    is_upper_lobe: bool = Form(True),
    nodule_count: int = Form(1),
    is_high_risk_patient: bool = Form(False),
    rescale_intercept: Optional[float] = Form(None),
    seed_x: Optional[int] = Form(None),
    seed_y: Optional[int] = Form(None),
    is_spiculated: Optional[bool] = Form(None)
):
    """
    Execute validated neural inference on uploaded CT slice or DICOM file.
    Calculates calibrated probabilities, RECIST 1.1 morphometry, Brock risk, and Fleischner 2017 guidelines.
    """
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    contents = await file.read()
    filename = file.filename or "scan.png"
    spacing = (pixel_spacing_row, pixel_spacing_col) if (pixel_spacing_row and pixel_spacing_col) else None

    # 1. Ingestion: DICOM vs Standard Image format
    if filename.lower().endswith(".dcm"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        from utils.image_utils import load_dicom_image
        image, dcm_meta = load_dicom_image(tmp_path)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        if image is None:
            raise HTTPException(status_code=400, detail="Cannot decode DICOM file.")
        if dcm_meta.get("pixel_spacing"):
            spacing = dcm_meta["pixel_spacing"]
        hu_slice = convert_to_hounsfield_units(
            image[:, :, 0] if image.ndim == 3 else image,
            dcm_meta.get("rescale_slope", 1.0),
            dcm_meta.get("rescale_intercept", 0.0)
        )
    else:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise HTTPException(status_code=400, detail="Uploaded file cannot be decoded as an image.")
        if image.dtype == np.uint8:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            hu_slice = (gray.astype(np.float32) / 255.0) * 1400.0 - 1000.0
        else:
            intercept_val = float(rescale_intercept) if rescale_intercept is not None else 0.0
            hu_slice = convert_to_hounsfield_units(image, rescale_slope=1.0, rescale_intercept=intercept_val)

    # 2. Quality and OOD Gate
    is_valid, ood_reasons = CTQualityAndOODValidator.validate_slice(hu_slice, {"modality": "CT"})
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail={"error": "Input Quality Gate Rejection", "reasons": ood_reasons}
        )

    # 3. Model check
    if SERVICE_STATE["model"] is None and not is_demo_mode_active():
        raise HTTPException(
            status_code=503,
            detail="Neural model weights not mounted. Set NODULE_DEMO=1 to enable UI demonstration."
        )

    # 3b. Nodule segmentation (BEFORE inference so shape features come from the nodule) & axial diameters (seed-guided when seed_x/seed_y given)
    recist_sr = None
    nodule_mask = None
    seed = (seed_x, seed_y) if (seed_x is not None and seed_y is not None) else None
    try:
        segmenter = AnatomicalNoduleSegmenter()
        nodule_mask = segmenter.segment_candidate_roi(hu_slice, seed_point=seed, spacing_mm=spacing)
        if spacing is not None and nodule_mask.sum() > 0:
            recist_meas = segmenter.measure_recist_11(nodule_mask, hu_slice, spacing)
            if recist_meas is not None:
                recist_sr = RECISTMeasurementSR(
                    longest_diameter_mm=recist_meas.longest_diameter_mm,
                    perpendicular_diameter_mm=recist_meas.perpendicular_diameter_mm,
                    mean_recist_diameter_mm=recist_meas.mean_recist_diameter_mm,
                    equivalent_diameter_mm=recist_meas.equivalent_diameter_mm,
                    consolidation_to_tumor_ratio=recist_meas.consolidation_to_tumor_ratio,
                    attenuation_type=recist_meas.attenuation_type,
                    estimated_volume_mm3=recist_meas.volume_mm3_estimated,
                    warnings=recist_meas.warnings
                )
    except Exception as exc:
        logger.warning("Nodule measurement error: %s", exc)
        nodule_mask = np.zeros(hu_slice.shape[:2], dtype=np.uint8)  # never fall back to whole-slice features

    # 3c. Crop-trained models classify ONE nodule: locate its centre, or refuse
    crop = None
    model_meta = SERVICE_STATE.get("model_meta") or {}
    if model_meta.get("input_mode") == "crop2d":
        center = None
        if nodule_mask is not None and int(nodule_mask.sum()) > 0:
            mom = cv2.moments(nodule_mask.astype(np.uint8), binaryImage=True)
            center = (mom["m10"] / mom["m00"], mom["m01"] / mom["m00"])
        elif seed is not None:
            center = (float(seed[0]), float(seed[1]))
        if center is None or spacing is None:
            raise HTTPException(
                status_code=422,
                detail="Model needs a nodule location and pixel spacing: send a DICOM with spacing "
                       "and seed_x/seed_y (or a slice where a nodule candidate can be segmented)."
            )
        crop = {"center_xy": center, "spacing_mm": spacing,
                "patch_mm": model_meta.get("patch_mm", 64.0), "out_px": model_meta.get("out_px", 64)}

    pred_class, raw_conf, features = predict_nodule(
        SERVICE_STATE["model"],
        image,
        spacing_mm=spacing,
        hu_slice=hu_slice,
        nodule_mask=nodule_mask,
        crop=crop,
    )

    # 4. Temperature Calibration
    raw_logit = features.get("raw_logit", 0.0)
    calibrated_prob = SERVICE_STATE["calibrator"].calibrate_probability(torch.tensor([raw_logit]))

    # 6. Brock / PanCan model: ONLY from a validated nodule measurement.
    #    (Brock expects the maximum diameter.) No measurement -> no risk score;
    #    never fall back to whole-slice features.
    brock_sr = None
    if recist_sr is not None and recist_sr.longest_diameter_mm:
        brock_res = BrockUniversityRiskCalculator.calculate_risk(
            size_mm=recist_sr.longest_diameter_mm,
            age=patient_age or 60,
            is_female=is_female,
            family_history_lung_cancer=family_history_lung_cancer,
            emphysema_present=emphysema_present,
            nodule_type=recist_sr.attenuation_type,
            is_upper_lobe=is_upper_lobe,
            nodule_count=nodule_count,
            # Brock needs a reader-assessed spiculation finding; the shape heuristic is not
            # used here because it is unvalidated. Default False when not supplied.
            is_spiculated=bool(is_spiculated)
        )
        brock_sr = BrockRiskSR(
            model_version=brock_res.model_version,
            malignancy_probability=brock_res.malignancy_probability,
            risk_percentage=brock_res.risk_percentage,
            risk_tier=brock_res.risk_tier,
            clinical_guidance=brock_res.clinical_guidance
        )

    # 7. Fleischner Society 2017 Rules
    effective_fleischner_size = (
        recist_sr.mean_recist_diameter_mm if recist_sr and recist_sr.mean_recist_diameter_mm
        else None   # whole-slice features["size_mm"] is not a nodule measurement
    )
    rule_features = dict(features)
    rule_features["size_mm"] = effective_fleischner_size

    expert = apply_expert_rules(
        prediction=pred_class,
        confidence=calibrated_prob,
        features=rule_features,
        is_high_risk=is_high_risk_patient
    )

    lineage = ClinicalLineageTracker.generate_provenance_stamp(
        model_path=SERVICE_STATE["model_path"]
    )
    lineage["latency_ms"] = f"{(time.perf_counter() - t0) * 1000.0:.2f}"

    return DiagnosticPayload(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        lineage=lineage,
        prediction="MALIGNANT" if pred_class == 1 else "BENIGN",
        confidence=round(raw_conf, 4),
        calibrated_probability=round(calibrated_prob, 4),
        measurements=FindingMeasurement(
            equivalent_diameter_mm=recist_sr.equivalent_diameter_mm if recist_sr else None,
            sphericity_index=features.get("sphericity", 0.0),
            margin_classification=features.get("margin", "smooth"),
            spiculation_present=features.get("spiculation", False),
            internal_texture=features.get("texture", "homogeneous")
        ),
        recist_measurements=recist_sr,
        brock_risk=brock_sr,
        expert_guidelines=FleischnerGuidelineSR(
            category_tier=expert.get("fleischner_details", {}).get("tier", "N/A"),
            surveillance_interval=expert.get("follow_up_interval", "N/A"),
            actionable_recommendation=expert.get("recommendation", "N/A")
        ),
        ood_validation={"status": "PASSED", "quality_checks": "Thoracic HU profile nominal"},
        regulatory_disclaimer=CLINICAL_DISCLAIMER
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain_nodule_endpoint(
    file: UploadFile = File(...),
    method: str = Form("gradcam++")  # 'gradcam++' or 'gradcam'
):
    """Generate visual saliency heatmap encoded as Base64 PNG."""
    request_id = str(uuid.uuid4())
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise HTTPException(status_code=400, detail="Cannot decode uploaded image.")

    if SERVICE_STATE["model"] is None and not is_demo_mode_active():
        raise HTTPException(status_code=503, detail="Model weights not available.")
    if (SERVICE_STATE.get("model_meta") or {}).get("input_mode") == "crop2d":
        raise HTTPException(
            status_code=422,
            detail="This model classifies a single-nodule crop; /explain does not support it yet "
                   "(it needs a nodule location and pixel spacing). Use the GUI or extend this endpoint."
        )
    m = method.lower()
    if m not in ("gradcam", "gradcam++"):
        raise HTTPException(status_code=400, detail="Unsupported method. Use 'gradcam' or 'gradcam++'.")
    # NOTE: this endpoint accepts 8-bit images only; they are mapped to HU assuming the
    # standard lung window (approximate). Use /predict with DICOM for exact preprocessing.
    overlay = generate_gradcam(SERVICE_STATE["model"], image, use_plus_plus=(m == "gradcam++"))

    # Encode to PNG Base64
    success, buffer = cv2.imencode(".png", overlay)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode saliency map.")

    b64_str = base64.b64encode(buffer).decode("utf-8")
    lineage = ClinicalLineageTracker.generate_provenance_stamp(SERVICE_STATE["model_path"])

    return ExplainResponse(
        request_id=request_id,
        target_class=1,
        method=method,
        saliency_png_base64=b64_str,
        lineage=lineage
    )

