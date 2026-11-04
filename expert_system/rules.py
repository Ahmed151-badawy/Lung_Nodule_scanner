"""
Clinical Expert System - Fleischner Society 2017 Guidelines & Heuristic Risk Analysis
Medical Decision Support for Incidental Pulmonary Nodules
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

CLINICAL_DISCLAIMER = (
    "RESEARCH USE ONLY – NOT FOR DIAGNOSTIC USE. "
    "Algorithmic outputs are investigational aids and must be interpreted by a qualified radiologist."
)


class FleischnerSocietyGuidelines2017:
    """
    Fleischner Society 2017 Guidelines for incidental pulmonary nodules in adults >= 35 years old.
    Exclusion criteria:
      - Patients < 35 years of age
      - Patients with known primary malignancy
      - Immunocompromised patients
      - Lung cancer screening cohorts (requires Lung-RADS)
    """

    EXCLUSION_CRITERIA = [
        "Patient age < 35 years",
        "Known history of active primary malignancy",
        "Immunocompromised or severe immunosuppression",
        "Screening setting (Lung-RADS applies instead of Fleischner)"
    ]

    @staticmethod
    def evaluate(
        size_mm: Optional[float],
        is_multiple: bool = False,
        is_high_risk_patient: bool = False,
        nodule_type: str = "solid"
    ) -> Dict[str, Any]:
        """
        Evaluate recommendation according to Fleischner 2017 published criteria.

        Args:
            size_mm: Average diameter in mm (from DICOM PixelSpacing)
            is_multiple: True if multiple nodules present
            is_high_risk_patient: True for smokers, emphysema, family history, etc.
            nodule_type: 'solid', 'ground_glass', or 'part_solid'
        """
        if size_mm is None or size_mm <= 0:
            return {
                "guideline": "Fleischner 2017",
                "tier": "Indeterminate",
                "recommendation": (
                    "Physical size in millimeters is unavailable (missing DICOM PixelSpacing). "
                    "Calibrate spatial resolution before applying Fleischner guideline thresholds."
                ),
                "follow_up_interval": "Requires spatial calibration",
                "exclusions": FleischnerSocietyGuidelines2017.EXCLUSION_CRITERIA
            }

        # 1. Subsolid Nodules
        if nodule_type == "ground_glass":
            if size_mm < 6.0:
                rec = "No routine CT follow-up required."
                interval = "None"
            else:
                rec = "CT follow-up at 6–12 months to confirm persistence, then CT every 2 years until 5 years."
                interval = "6-12 months, then every 2 years"
            return {
                "guideline": "Fleischner 2017 (Ground-glass)",
                "size_mm": size_mm,
                "tier": "< 6 mm" if size_mm < 6.0 else ">= 6 mm",
                "recommendation": rec,
                "follow_up_interval": interval,
                "exclusions": FleischnerSocietyGuidelines2017.EXCLUSION_CRITERIA
            }

        if nodule_type == "part_solid":
            if size_mm < 6.0:
                rec = "No routine CT follow-up required."
                interval = "None"
            else:
                rec = (
                    "CT follow-up at 3–6 months to confirm persistence. "
                    "If persistent and solid component >= 6 mm, consider PET/CT or tissue sampling."
                )
                interval = "3-6 months"
            return {
                "guideline": "Fleischner 2017 (Part-solid)",
                "size_mm": size_mm,
                "tier": "< 6 mm" if size_mm < 6.0 else ">= 6 mm",
                "recommendation": rec,
                "follow_up_interval": interval,
                "exclusions": FleischnerSocietyGuidelines2017.EXCLUSION_CRITERIA
            }

        # 2. Solid Nodules - Single vs Multiple
        if not is_multiple:
            if size_mm < 6.0:
                if is_high_risk_patient:
                    rec = "Optional CT at 12 months depending on clinical risk profile."
                    interval = "Optional 12 months"
                else:
                    rec = "No routine follow-up needed for low-risk clinical profile."
                    interval = "None"
                tier = "< 6 mm"

            elif 6.0 <= size_mm <= 8.0:
                if is_high_risk_patient:
                    rec = "CT follow-up at 6–12 months, then consider follow-up CT at 18–24 months."
                    interval = "6-12 months, then 18-24 months"
                else:
                    rec = "CT follow-up at 6–12 months; consider follow-up CT at 18–24 months if stable."
                    interval = "6-12 months"
                tier = "6-8 mm"

            else:  # > 8.0 mm
                rec = (
                    "Consider follow-up CT at 3 months, functional imaging (PET/CT), "
                    "or tissue sampling/biopsy depending on clinical suspicion."
                )
                interval = "3 months / PET-CT / Biopsy consideration"
                tier = "> 8 mm"

        else:  # Multiple solid nodules
            if size_mm < 6.0:
                if is_high_risk_patient:
                    rec = "Optional CT follow-up at 12 months."
                    interval = "Optional 12 months"
                else:
                    rec = "No routine follow-up needed."
                    interval = "None"
                tier = "< 6 mm (Multiple)"

            elif 6.0 <= size_mm <= 8.0:
                rec = "CT follow-up at 3–6 months, then subsequent CT at 18–24 months if stable."
                interval = "3-6 months, then 18-24 months"
                tier = "6-8 mm (Multiple)"

            else:  # > 8.0 mm
                rec = (
                    "CT follow-up at 3–6 months, then at 18–24 months; "
                    "consider PET/CT or biopsy for dominant suspicious nodule."
                )
                interval = "3-6 months / PET-CT"
                tier = "> 8 mm (Multiple)"

        return {
            "guideline": "Fleischner 2017 (Solid)",
            "size_mm": size_mm,
            "tier": tier,
            "is_multiple": is_multiple,
            "patient_risk": "High" if is_high_risk_patient else "Low",
            "recommendation": rec,
            "follow_up_interval": interval,
            "exclusions": FleischnerSocietyGuidelines2017.EXCLUSION_CRITERIA
        }


class HeuristicRiskEngine:
    """
    Research-Grade Heuristic Composite Scoring Engine.
    Combines neural classification probability with morphological descriptors.
    Explicitly labeled as heuristic and unvalidated by multi-center prospective trials.
    """

    WEIGHTS = {
        'model_prob': 0.40,
        'size': 0.25,
        'margin': 0.20,
        'sphericity': 0.10,
        'spiculation': 0.05
    }

    @classmethod
    def calculate_heuristic_score(
        cls,
        prediction: int,
        confidence: float,
        features: Dict[str, Any]
    ) -> float:
        """
        Calculate weighted heuristic risk index in [0.0, 1.0].
        Weights:
            0.40 * Model Probability
            0.25 * Size (Calibrated or Pixel Normalized)
            0.20 * Margin Irregularity
            0.10 * Non-Sphericity
            0.05 * Spiculation
        """
        score = 0.0

        # 1. Neural Model Output
        prob_malignant = confidence if prediction == 1 else (1.0 - confidence)
        score += cls.WEIGHTS['model_prob'] * prob_malignant

        # 2. Size (mm or normalized pixel count)
        size_mm = features.get('size_mm')
        if size_mm is not None:
            if size_mm < 6.0:
                size_factor = 0.1
            elif size_mm <= 8.0:
                size_factor = 0.4
            elif size_mm <= 15.0:
                size_factor = 0.75
            else:
                size_factor = 1.0
        else:
            # Fallback to relative pixel count if uncalibrated
            px = features.get('size_pixels', 0)
            size_factor = min(1.0, px / 1500.0)
        score += cls.WEIGHTS['size'] * size_factor

        # 3. Margin Irregularity
        margin = str(features.get('margin', 'smooth')).lower()
        margin_factors = {
            'smooth': 0.0,
            'lobulated': 0.4,
            'irregular': 0.8,
            'spiculated': 1.0,
            'indeterminate': 0.3
        }
        score += cls.WEIGHTS['margin'] * margin_factors.get(margin, 0.4)

        # 4. Sphericity (lower sphericity = higher irregularity)
        sphericity = float(features.get('sphericity', 0.8))
        shape_factor = float(np.clip(1.0 - sphericity, 0.0, 1.0))
        score += cls.WEIGHTS['sphericity'] * shape_factor

        # 5. Spiculation
        if bool(features.get('spiculation', False)):
            score += cls.WEIGHTS['spiculation']

        return round(float(max(0.0, min(1.0, score))), 3)

    @staticmethod
    def categorize_risk_tier(score: float) -> str:
        """Categorize into research risk tier."""
        if score < 0.30:
            return "LOW"
        elif score < 0.55:
            return "MODERATE"
        elif score < 0.75:
            return "ELEVATED"
        else:
            return "HIGH"


def apply_expert_rules(
    prediction: int,
    confidence: float,
    features: Dict[str, Any],
    is_high_risk_patient: bool = False,
    is_multiple: bool = False,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Synthesize complete clinical expert system review.
    Integrates genuine Fleischner 2017 guidelines with clearly designated heuristic score.
    """
    if "is_high_risk" in kwargs:
        is_high_risk_patient = bool(kwargs["is_high_risk"])
    heuristic_score = HeuristicRiskEngine.calculate_heuristic_score(
        prediction, confidence, features
    )
    risk_level = HeuristicRiskEngine.categorize_risk_tier(heuristic_score)

    # Fleischner evaluation
    size_mm = features.get('size_mm')
    texture = features.get('texture', 'homogeneous')
    nodule_type = "ground_glass" if texture == 'ground-glass' else "solid"

    fleischner_eval = FleischnerSocietyGuidelines2017.evaluate(
        size_mm=size_mm,
        is_multiple=is_multiple,
        is_high_risk_patient=is_high_risk_patient,
        nodule_type=nodule_type
    )

    # Key Contributing Risk Factors
    factors = []
    if prediction == 1:
        factors.append(f"AI classifier flagged malignant features (confidence: {confidence:.1%})")
    if size_mm is not None:
        if size_mm >= 8.0:
            factors.append(f"Dominant size {size_mm:.1f} mm exceeds 8 mm high-risk threshold")
        elif size_mm >= 6.0:
            factors.append(f"Size {size_mm:.1f} mm requires surveillance follow-up")
    else:
        factors.append("DICOM spatial pixel spacing uncalibrated; size evaluated in pixels")

    margin = features.get('margin', 'smooth')
    if margin in ('irregular', 'spiculated', 'lobulated'):
        factors.append(f"Non-smooth margin detected ({margin})")

    if features.get('spiculation', False):
        factors.append("Peritumoral spiculation present")

    if not factors:
        factors.append("Well-circumscribed margins with regular spherical geometry")

    # Responsible Clinical Summary
    if risk_level in ("HIGH", "ELEVATED"):
        clinical_impression = (
            "Findings suspicious for non-benign etiology based on morphological and neural indicators. "
            "Suggest multidisciplinary thoracic oncology consultation, correlation with prior chest imaging, "
            "and adherence to Fleischner follow-up protocol."
        )
    elif risk_level == "MODERATE":
        clinical_impression = (
            "Intermediate imaging features. Findings suggest surveillance protocol as detailed in "
            "Fleischner guidelines. Correlate with patient smoking history and prior imaging."
        )
    else:
        clinical_impression = (
            "Imaging features largely favor benign morphology. Routine surveillance or interval "
            "follow-up as indicated by clinical risk profile."
        )

    return {
        "risk_level": risk_level,
        "risk_score": heuristic_score,
        "score_type": "Heuristic Research Index (Unvalidated)",
        "recommendation": fleischner_eval["recommendation"],
        "follow_up_interval": fleischner_eval["follow_up_interval"],
        "fleischner_details": fleischner_eval,
        "factors": factors,
        "clinical_notes": clinical_impression,
        "disclaimer": CLINICAL_DISCLAIMER
    }
