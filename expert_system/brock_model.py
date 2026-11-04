"""
Brock University / PanCan malignancy model (full model with spiculation, "2b").
Reference: McWilliams A, et al. N Engl J Med 2013;369:910-919.

Intended population: nodules found on a *first screening CT* (diameters roughly
1-30 mm). Outside that range, or in non-screening settings, the estimate is
extrapolation and is flagged via `warnings`. The model is also known to be
miscalibrated in some external cohorts (e.g. NLST, Winter et al. Thorax 2019),
so treat the output as a research estimate, not a validated clinical risk.
"""

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class BrockRiskResult:
    malignancy_probability: float  # [0.0, 1.0]
    risk_percentage: float         # [0.0, 100.0]
    risk_tier: str                 # 'VERY_LOW', 'LOW', 'INTERMEDIATE', 'HIGH', 'NEGLIGIBLE'
    clinical_guidance: str
    model_version: str = "PanCan / Brock full model with spiculation (McWilliams 2013)"
    warnings: List[str] = field(default_factory=list)


class BrockUniversityRiskCalculator:
    # Published coefficients (full model with spiculation)
    INTERCEPT = -6.7892
    B_AGE = 0.0287             # per year, centered at 62
    B_FEMALE = 0.6011
    B_FAMILY_HISTORY = 0.2961
    B_EMPHYSEMA = 0.2953
    B_SIZE = -5.3854           # on (d/10)^-0.5 - 1.58113883
    B_PART_SOLID = 0.3770
    B_NON_SOLID = -0.1276
    B_UPPER_LOBE = 0.6581
    B_COUNT = -0.0824          # per nodule, centered at 4
    B_SPICULATION = 0.7729

    AGE_CENTER = 62.0
    SIZE_CENTER = 1.58113883   # (4/10) ** -0.5
    COUNT_CENTER = 4

    SIZE_RANGE_MM = (1.0, 30.0)

    @classmethod
    def calculate_risk(
        cls,
        size_mm: float,
        age: int,
        is_female: bool,
        family_history_lung_cancer: bool = False,
        emphysema_present: bool = False,
        nodule_type: str = "solid",   # 'solid', 'part_solid', 'ground_glass'
        is_upper_lobe: bool = True,
        nodule_count: int = 1,
        is_spiculated: bool = False,
    ) -> BrockRiskResult:
        if size_mm is None or size_mm <= 0:
            return BrockRiskResult(
                malignancy_probability=0.0,
                risk_percentage=0.0,
                risk_tier="NEGLIGIBLE",
                clinical_guidance="Nodule size is zero or uncalibrated; risk not computed.",
                warnings=["Invalid nodule size."],
            )

        warnings: List[str] = []
        lo, hi = cls.SIZE_RANGE_MM
        if not (lo <= size_mm <= hi):
            warnings.append(
                f"Nodule size {size_mm:.1f} mm is outside the model's development range "
                f"({lo:.0f}-{hi:.0f} mm); estimate is extrapolated and unreliable."
            )
        if nodule_count < 1:
            warnings.append("nodule_count < 1 is invalid; treated as 1.")
            nodule_count = 1

        x = cls.INTERCEPT
        x += cls.B_AGE * (age - cls.AGE_CENTER)
        x += cls.B_FEMALE * float(is_female)
        x += cls.B_FAMILY_HISTORY * float(family_history_lung_cancer)
        x += cls.B_EMPHYSEMA * float(emphysema_present)
        x += cls.B_SIZE * (((size_mm / 10.0) ** -0.5) - cls.SIZE_CENTER)

        nt = nodule_type.lower()
        if nt in ("ground_glass", "non_solid", "nonsolid"):
            x += cls.B_NON_SOLID
        elif nt in ("part_solid", "sub_solid", "partsolid"):
            x += cls.B_PART_SOLID

        x += cls.B_UPPER_LOBE * float(is_upper_lobe)
        x += cls.B_COUNT * (nodule_count - cls.COUNT_CENTER)
        x += cls.B_SPICULATION * float(is_spiculated)

        prob = 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0
        pct = round(prob * 100.0, 2)

        # Tier cut-offs unchanged from the original file; verify against the
        # guideline you intend to cite (BTS 2015 uses <10% / 10-70% / >70%).
        if pct < 5.0:
            tier, guidance = "VERY_LOW", "Very low estimated risk (<5%)."
        elif pct <= 10.0:
            tier, guidance = "LOW", "Low estimated risk (5-10%)."
        elif pct <= 65.0:
            tier, guidance = "INTERMEDIATE", "Intermediate estimated risk (10-65%)."
        else:
            tier, guidance = "HIGH", "High estimated risk (>65%)."
        guidance += " Research estimate only; clinical decisions rest with the treating team."

        return BrockRiskResult(
            malignancy_probability=round(prob, 5),
            risk_percentage=pct,
            risk_tier=tier,
            clinical_guidance=guidance,
            warnings=warnings,
        )
