"""
Phase 2: Anatomical Nodule Segmentation & RECIST 1.1 Morphometry
Calculates Longest Axial Diameter, Shortest Perpendicular Axis, and Consolidation-to-Tumor Ratio (CTR)
"""

import logging
from typing import Tuple, Dict, Optional, Any, List
from dataclasses import dataclass, field

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class RECISTMeasurement:
    longest_diameter_mm: float
    perpendicular_diameter_mm: float
    mean_recist_diameter_mm: float
    equivalent_diameter_mm: float
    consolidation_to_tumor_ratio: float  # CTR = solid_core_diam / total_diam
    attenuation_type: str  # 'solid', 'part_solid', 'ground_glass'
    area_mm2: float
    volume_mm3_estimated: float  # Spheroid approximation V = (pi/6) * d_long * d_perp^2
    warnings: List[str] = field(default_factory=list)


class AnatomicalNoduleSegmenter:
    """
    Candidate-level anatomical segmenter.
    Isolates parenchymal nodules from vascular tree and chest wall.
    Calculates standard RECIST 1.1 dual-axis orthogonal diameters.
    """

    # Plausibility limits (a "nodule" per Fleischner/Brock is <= 30 mm; larger = mass)
    MAX_NODULE_DIAMETER_MM = 40.0
    MIN_NODULE_DIAMETER_MM = 2.0

    def __init__(
        self,
        hu_lung_min: float = -950.0,
        hu_lung_max: float = -400.0,
        nodule_hu_min: float = -400.0,   # lower to ~-700 for sub-solid nodules (seed-guided use only)
        nodule_hu_max: float = 1500.0,   # includes calcified nodules
        lung_closing_px: int = 15,       # closes juxtapleural notches in the lung mask
    ):
        self.hu_lung_min = hu_lung_min
        self.hu_lung_max = hu_lung_max
        self.nodule_hu_min = nodule_hu_min
        self.nodule_hu_max = nodule_hu_max
        self.lung_closing_px = lung_closing_px

    # ------------------------------------------------------------------
    def compute_lung_mask(self, slice_hu: np.ndarray) -> np.ndarray:
        """
        Filled lung field (uint8 0/1). Air connected to an image corner is
        treated as outside-the-body and removed; remaining low-HU components
        are the lungs; holes (nodules, vessels) are filled so that soft tissue
        INSIDE the lungs can be told apart from chest wall / mediastinum.
        """
        h, w = slice_hu.shape[:2]
        air = (slice_hu < -400.0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(air, connectivity=8)
        outside = {int(lab[0, 0]), int(lab[0, w - 1]), int(lab[h - 1, 0]), int(lab[h - 1, w - 1])} - {0}
        min_area = 0.005 * h * w
        lung = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, n):
            if i in outside or stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            lung[lab == i] = 1
        if lung.sum() == 0:
            return lung
        k = max(1, int(self.lung_closing_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        lung = cv2.morphologyEx(lung, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(lung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(lung)
        cv2.drawContours(filled, contours, -1, 1, -1)
        return filled

    def segment_candidate_roi(
        self,
        slice_hu: np.ndarray,
        seed_point: Optional[Tuple[int, int]] = None,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        spacing_mm: Optional[Tuple[float, float]] = None,
        max_seed_distance_mm: float = 10.0,
    ) -> np.ndarray:
        """
        Segment ONE nodule candidate inside the lungs. Returns uint8 mask (0/1),
        or an all-zero mask when nothing plausible is found (never the body).

        seed_point (x, y) is strongly recommended. Without it a heuristic picks
        the most compact lung-internal component of plausible size; with several
        nodules on the slice that choice is arbitrary.
        spacing_mm (row, col) enables size limits in mm; otherwise limits are
        applied as a fraction of the image.
        """
        h, w = slice_hu.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)

        lung = self.compute_lung_mask(slice_hu)
        if lung.sum() == 0:
            logger.warning("No lung field found; refusing to segment a nodule.")
            return full_mask

        cand = ((slice_hu >= self.nodule_hu_min) & (slice_hu <= self.nodule_hu_max) & (lung == 1)).astype(np.uint8)

        if roi_bbox is not None:
            rx, ry, rw, rh = roi_bbox
            roi = np.zeros_like(cand)
            roi[max(0, ry):ry + rh, max(0, rx):rx + rw] = 1
            cand &= roi

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel)
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel)

        n, lab, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
        if n <= 1:
            return full_mask

        mm_per_px = float(np.sqrt(spacing_mm[0] * spacing_mm[1])) if spacing_mm else None

        def eq_diam_mm(area_px: float) -> Optional[float]:
            return 2.0 * np.sqrt(area_px / np.pi) * mm_per_px if mm_per_px else None

        def plausible(i: int) -> bool:
            area = float(stats[i, cv2.CC_STAT_AREA])
            d = eq_diam_mm(area)
            if d is not None:
                return self.MIN_NODULE_DIAMETER_MM <= d <= self.MAX_NODULE_DIAMETER_MM
            return 4.0 <= area <= 0.02 * h * w   # no spacing: <= 2% of the slice

        chosen = 0
        if seed_point is not None:
            sx, sy = int(seed_point[0]), int(seed_point[1])
            if 0 <= sx < w and 0 <= sy < h:
                if lab[sy, sx] > 0:
                    chosen = int(lab[sy, sx])
                else:
                    r = int(round(max_seed_distance_mm / mm_per_px)) if mm_per_px else 15
                    y0, y1, x0, x1 = max(0, sy - r), min(h, sy + r + 1), max(0, sx - r), min(w, sx + r + 1)
                    ys, xs = np.nonzero(lab[y0:y1, x0:x1])
                    if len(ys):
                        d2 = (ys + y0 - sy) ** 2 + (xs + x0 - sx) ** 2
                        j = int(np.argmin(d2))
                        if d2[j] <= r * r:
                            chosen = int(lab[ys[j] + y0, xs[j] + x0])
            if chosen == 0:
                logger.warning("Seed point does not fall on/near lung-internal soft tissue.")
                return full_mask
        else:
            best_score = -1.0
            for i in range(1, n):
                if not plausible(i):
                    continue
                comp = (lab == i).astype(np.uint8)
                cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                area = float(cv2.contourArea(cnts[0]))
                per = float(cv2.arcLength(cnts[0], True))
                circ = (4.0 * np.pi * area / per ** 2) if per > 0 else 0.0
                if circ > best_score:
                    best_score, chosen = circ, i
            if chosen == 0:
                return full_mask

        if not plausible(chosen):
            logger.warning(
                "Rejected candidate: equivalent diameter %s mm outside [%.0f, %.0f] mm "
                "(vessel, mass or consolidation, not a measurable nodule).",
                None if eq_diam_mm(float(stats[chosen, cv2.CC_STAT_AREA])) is None
                else round(eq_diam_mm(float(stats[chosen, cv2.CC_STAT_AREA])), 1),
                self.MIN_NODULE_DIAMETER_MM, self.MAX_NODULE_DIAMETER_MM,
            )
            return full_mask

        full_mask[lab == chosen] = 1
        return full_mask

    def measure_recist_11(
        self,
        nodule_mask: np.ndarray,
        slice_hu: np.ndarray,
        spacing_mm: Tuple[float, float]
    ) -> Optional[RECISTMeasurement]:
        """
        Compute true dual-axis orthogonal diameters complying with RECIST 1.1 guidelines.
        """
        if nodule_mask.sum() == 0:
            return None

        contours, _ = cv2.findContours(nodule_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        pts = contour.squeeze(1).astype(np.float32)  # [N, 2] (x, y)

        if len(pts) < 5:
            # Fallback to minimum enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            diam = 2.0 * radius * float(spacing_mm[0])
            return RECISTMeasurement(
                longest_diameter_mm=diam,
                perpendicular_diameter_mm=diam,
                mean_recist_diameter_mm=diam,
                equivalent_diameter_mm=diam,
                consolidation_to_tumor_ratio=1.0,
                attenuation_type="solid",
                area_mm2=float(nodule_mask.sum()) * spacing_mm[0] * spacing_mm[1],
                volume_mm3_estimated=(np.pi / 6.0) * (diam ** 3)
            )

        # Scale coordinates into physical millimeter space
        pts_mm = pts * np.array([spacing_mm[1], spacing_mm[0]], dtype=np.float32)

        # 1. Compute Longest Axial Diameter (d_long)
        # Pairwise distance matrix in mm
        diff = pts_mm[:, np.newaxis, :] - pts_mm[np.newaxis, :, :]
        dist_sq = (diff ** 2).sum(axis=-1)
        max_idx = np.unravel_index(np.argmax(dist_sq), dist_sq.shape)
        pt_a = pts_mm[max_idx[0]]
        pt_b = pts_mm[max_idx[1]]
        d_long = float(np.sqrt(dist_sq[max_idx]))

        # Unit vector along longest axis
        axis_vec = (pt_b - pt_a) / (d_long + 1e-8)
        perp_vec = np.array([-axis_vec[1], axis_vec[0]], dtype=np.float32)

        # 2. Compute Shortest Perpendicular Diameter (d_perp)
        # Project all points onto perpendicular axis
        proj_perp = np.dot(pts_mm, perp_vec)
        d_perp = float(proj_perp.max() - proj_perp.min())

        d_mean = (d_long + d_perp) / 2.0

        # Equivalent spherical diameter
        pixel_count = float(nodule_mask.sum())
        area_mm2 = pixel_count * float(spacing_mm[0]) * float(spacing_mm[1])
        d_equiv = 2.0 * np.sqrt(area_mm2 / np.pi)

        # Estimated volume: Prolate spheroid approximation
        volume_mm3 = (np.pi / 6.0) * d_long * (d_perp ** 2)

        # 3. Consolidation-to-Tumor Ratio (CTR) & Attenuation Type
        nodule_hu = slice_hu[nodule_mask == 1]
        solid_pixels = nodule_hu >= -150.0  # soft tissue solid core threshold
        solid_ratio = float(solid_pixels.sum()) / float(max(1, len(nodule_hu)))
        ctr = float(np.sqrt(solid_ratio))  # 1D linear ratio

        if ctr >= 0.95:
            attenuation = "solid"
        elif ctr <= 0.05:
            attenuation = "ground_glass"
        else:
            attenuation = "part_solid"

        warns: List[str] = []
        h_img, w_img = nodule_mask.shape[:2]
        ys, xs = np.nonzero(nodule_mask)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == h_img - 1 or xs.max() == w_img - 1:
            warns.append("Mask touches the image border; measurement may be truncated.")
        if d_long > self.MAX_NODULE_DIAMETER_MM:
            warns.append(f"Longest diameter {d_long:.1f} mm exceeds {self.MAX_NODULE_DIAMETER_MM:.0f} mm: not a nodule-scale object.")
        if d_long < self.MIN_NODULE_DIAMETER_MM:
            warns.append(f"Longest diameter {d_long:.1f} mm is below measurement reliability.")

        return RECISTMeasurement(
            longest_diameter_mm=round(d_long, 2),
            perpendicular_diameter_mm=round(d_perp, 2),
            mean_recist_diameter_mm=round(d_mean, 2),
            equivalent_diameter_mm=round(d_equiv, 2),
            consolidation_to_tumor_ratio=round(ctr, 3),
            attenuation_type=attenuation,
            area_mm2=round(area_mm2, 2),
            volume_mm3_estimated=round(volume_mm3, 2),
            warnings=warns
        )

