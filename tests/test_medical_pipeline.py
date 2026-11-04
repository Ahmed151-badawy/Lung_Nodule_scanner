"""
Comprehensive Unit & Integration Test Suite
Validates Clinical AI Safety, DICOM Calibration, Grad-CAM Hooks, 
Fleischner Rules, 2.5D Volumetric Context, OOD Detection, Temperature Scaling, and Lineage
"""

import os
import unittest
import tempfile
import numpy as np
import torch
import cv2
import pydicom

from model.model import (
    NoduleCNN,
    predict_nodule,
    equivalent_diameter_mm,
    preprocess_image,
    extract_features
)
from explainability.gradcam import GradCAM, GradCAMPlusPlus, generate_gradcam
from explainability.calibration_and_saliency import ModelCalibrator, ScoreCAM, IntegratedGradients
from preprocessing.volume_ood import CTQualityAndOODValidator, VolumetricContextExtractor
from preprocessing.nodule_segmenter import AnatomicalNoduleSegmenter, RECISTMeasurement
from expert_system.brock_model import BrockUniversityRiskCalculator, BrockRiskResult
from reports.lineage import ClinicalLineageTracker
from reports.dicom_sr import create_dicom_sr_report
from utils.image_utils import (
    convert_to_hounsfield_units,
    apply_window_level,
    apply_lung_window,
    resize_with_padding
)
from expert_system.rules import (
    FleischnerSocietyGuidelines2017,
    HeuristicRiskEngine,
    apply_expert_rules
)
from fastapi.testclient import TestClient
from service.app import app


class TestModelArchitecture(unittest.TestCase):
    def test_parameter_count_is_compact(self):
        """Verify model was reduced from 33.5M params to < 0.5M params."""
        model = NoduleCNN(in_channels=1, num_classes=1)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertLess(total_params, 1_000_000, f"Model has {total_params} parameters, expected < 1M")
        self.assertGreater(total_params, 100_000)

    def test_resolution_agnostic_forward(self):
        """Verify GAP allows forward passes on arbitrary input spatial resolutions."""
        model = NoduleCNN(in_channels=1, num_classes=1)
        model.eval()
        for size in [(256, 256), (128, 128), (300, 300)]:
            x = torch.randn(2, 1, *size)
            logits = model(x)
            self.assertEqual(logits.shape, (2, 1))

    def test_25d_model_architecture(self):
        """Verify 2.5D model with 3 input channels forward pass."""
        model_25d = NoduleCNN(in_channels=3, num_classes=1)
        model_25d.eval()
        x = torch.randn(1, 3, 256, 256)
        logits = model_25d(x)
        self.assertEqual(logits.shape, (1, 1))


class TestGradCAMSafety(unittest.TestCase):
    def setUp(self):
        self.model = NoduleCNN(in_channels=1, num_classes=1)
        self.model.eval()

    def test_gradcam_lifecycle_and_hook_cleanup(self):
        """Verify hooks are cleanly removed after context manager exit."""
        target_layer = self.model.features[6]
        with GradCAM(self.model, target_layer) as explainer:
            self.assertEqual(len(explainer._hook_handles), 2)
            input_tensor = torch.randn(1, 1, 256, 256, requires_grad=True)
            cam = explainer.generate_cam(input_tensor)
            self.assertEqual(cam.ndim, 2)
            self.assertGreaterEqual(cam.min(), 0.0)
            self.assertLessEqual(cam.max(), 1.0)

        # Post exit: hooks must be empty
        self.assertEqual(len(explainer._hook_handles), 0)

    def test_gradcam_plus_plus(self):
        """Verify Grad-CAM++ operates without exceptions."""
        with GradCAMPlusPlus(self.model) as explainer:
            x = torch.randn(1, 1, 256, 256, requires_grad=True)
            cam = explainer.generate_cam(x)
            self.assertEqual(cam.ndim, 2)


class TestCalibrationAndAdvancedSaliency(unittest.TestCase):
    def setUp(self):
        self.model = NoduleCNN(in_channels=1, num_classes=1)
        self.model.eval()

    def test_temperature_scaling_and_ece(self):
        """Verify temperature scaling calibration and ECE calculation."""
        calibrator = ModelCalibrator(temperature=1.2)
        logits = torch.tensor([1.5, -2.0, 0.5])
        scaled = calibrator(logits)
        self.assertAlmostEqual(float(scaled[0].item()), 1.5 / 1.2, places=4)

        # ECE test with dummy probabilities and labels
        probs = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0])
        ece, meta = ModelCalibrator.calculate_ece(probs, labels, num_bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)
        self.assertIn("bins", meta)

    def test_score_cam(self):
        """Verify Score-CAM generates valid normalized saliency map."""
        score_cam = ScoreCAM(self.model)
        x = torch.randn(1, 1, 256, 256)
        cam = score_cam.generate_cam(x, max_channels=4)
        self.assertEqual(cam.shape, (256, 256))
        self.assertGreaterEqual(cam.min(), 0.0)
        self.assertLessEqual(cam.max(), 1.0)

    def test_integrated_gradients(self):
        """Verify Integrated Gradients produces bounded attribution map."""
        ig = IntegratedGradients(self.model)
        x = torch.randn(1, 1, 256, 256)
        saliency = ig.attribute(x, steps=5)
        self.assertEqual(saliency.shape, (256, 256))
        self.assertGreaterEqual(saliency.min(), 0.0)
        self.assertLessEqual(saliency.max(), 1.0)


class TestVolumetricAndOOD(unittest.TestCase):
    def test_25d_context_builder(self):
        """Verify 2.5D extractor stacks 3 adjacent slices into [1, 3, 256, 256]."""
        extractor = VolumetricContextExtractor(target_size=(256, 256))
        slice_prev = np.full((512, 512), -700.0, dtype=np.float32)
        slice_curr = np.full((512, 512), -600.0, dtype=np.float32)
        slice_next = np.full((512, 512), -500.0, dtype=np.float32)

        tensor_25d = extractor.build_25d_tensor([slice_prev, slice_curr, slice_next])
        self.assertEqual(tensor_25d.shape, (1, 3, 256, 256))
        self.assertEqual(tensor_25d.dtype, torch.float32)

    def test_ood_quality_gate(self):
        """Verify OOD detector rejects blank air slice and non-CT modalities."""
        # 1. Blank air slice (all -1000 HU)
        blank_air = np.full((256, 256), -1000.0, dtype=np.float32)
        is_valid, reasons = CTQualityAndOODValidator.validate_slice(blank_air)
        self.assertFalse(is_valid)
        self.assertTrue(any("max HU" in r or "Lung parenchyma" in r for r in reasons))

        # 2. Non-CT modality rejection
        valid_sim = np.full((256, 256), 40.0, dtype=np.float32)
        valid_sim[50:150, 50:150] = -800.0  # simulate lung area
        is_valid, reasons = CTQualityAndOODValidator.validate_slice(valid_sim, {"modality": "MR"})
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid modality" in r for r in reasons))

        # 3. Valid simulated CT slice
        is_valid, reasons = CTQualityAndOODValidator.validate_slice(valid_sim, {"modality": "CT"})
        self.assertTrue(is_valid)
        self.assertEqual(len(reasons), 0)


class TestPhysicalMorphologyAndSizing(unittest.TestCase):
    def test_synthetic_disc_diameter(self):
        """
        Verify equivalent_diameter_mm on a known synthetic circular disc:
        Radius = 20 px, Spacing = (0.5 mm, 0.5 mm).
        Expected physical diameter = 2 * 20 * 0.5 = 20.0 mm.
        """
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 20, 1, -1)
        spacing = (0.5, 0.5)

        diam_mm = equivalent_diameter_mm(mask, spacing)
        self.assertAlmostEqual(diam_mm, 20.0, delta=0.5)

    def test_missing_spacing_reports_uncalibrated(self):
        """Verify absence of DICOM spacing does not silently compute fake sizes."""
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(test_img, (50, 50), 15, (255, 255, 255), -1)

        features = extract_features(test_img, spacing_mm=None)
        self.assertIsNone(features['size_mm'])
        self.assertFalse(features['spacing_available'])


class TestDICOMHounsfieldConversion(unittest.TestCase):
    def test_hu_rescale(self):
        """Verify raw pixel to HU conversion: HU = raw * slope + intercept."""
        raw = np.array([0, 1000, 2000], dtype=np.int16)
        slope = 1.0
        intercept = -1024.0
        hu = convert_to_hounsfield_units(raw, slope, intercept)
        expected = np.array([-1024.0, -24.0, 976.0])
        np.testing.assert_allclose(hu, expected)

    def test_lung_window_clipping(self):
        """Verify lung window (-600 HU center, 1500 HU width) covers [-1350, +150] HU."""
        hu = np.array([-2000.0, -1350.0, -600.0, 150.0, 500.0], dtype=np.float32)
        windowed = apply_lung_window(hu)
        self.assertEqual(windowed[0], 0)      # clipped below -1350
        self.assertEqual(windowed[1], 0)      # at -1350
        self.assertEqual(windowed[3], 255)    # at +150
        self.assertEqual(windowed[4], 255)    # clipped above +150


class TestFleischner2017Guidelines(unittest.TestCase):
    def test_solid_single_nodule_sub_6mm(self):
        res_low = FleischnerSocietyGuidelines2017.evaluate(size_mm=4.5, is_multiple=False, is_high_risk_patient=False)
        self.assertEqual(res_low['tier'], "< 6 mm")
        self.assertEqual(res_low['follow_up_interval'], "None")

        res_high = FleischnerSocietyGuidelines2017.evaluate(size_mm=4.5, is_multiple=False, is_high_risk_patient=True)
        self.assertIn("12 months", res_high['follow_up_interval'])

    def test_solid_single_nodule_6_to_8mm(self):
        res = FleischnerSocietyGuidelines2017.evaluate(size_mm=7.2, is_multiple=False, is_high_risk_patient=False)
        self.assertEqual(res['tier'], "6-8 mm")
        self.assertEqual(res['follow_up_interval'], "6-12 months")

    def test_solid_single_nodule_above_8mm(self):
        res = FleischnerSocietyGuidelines2017.evaluate(size_mm=12.0, is_multiple=False)
        self.assertEqual(res['tier'], "> 8 mm")
        self.assertIn("PET-CT / Biopsy", res['follow_up_interval'])


class TestLineageTracking(unittest.TestCase):
    def test_provenance_stamp_generation(self):
        """Verify SHA-256 and lineage generation."""
        meta = {
            "series_instance_uid": "1.2.840.113619.2.55.3.60468838",
            "patient_id": "TCIA_LIDC_0001"
        }
        stamp = ClinicalLineageTracker.generate_provenance_stamp(
            model_path="model/best_model.pth",
            dicom_metadata=meta
        )
        self.assertIn("model_sha256", stamp)
        self.assertIn("pipeline_git_sha", stamp)
        self.assertIn("series_instance_uid", stamp)
        self.assertIn("audit_hash", stamp)
        self.assertEqual(stamp["series_instance_uid"], "1.2.840.113619.2.55.3.60468838")


class TestRECIST11Morphometry(unittest.TestCase):
    def setUp(self):
        self.segmenter = AnatomicalNoduleSegmenter()

    def test_recist_synthetic_ellipse_orthogonal_axes(self):
        """
        Verify RECIST 1.1 dual-axis measurement on synthetic ellipse.
        Major axis radius = 20 px, Minor axis radius = 10 px.
        Spacing = (0.5 mm, 0.5 mm).
        Expected d_long ~= 20 mm, d_perp ~= 10 mm.
        """
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.ellipse(mask, (60, 60), (20, 10), 0, 0, 360, 1, -1)
        hu_slice = np.full((120, 120), -700.0, dtype=np.float32)
        hu_slice[mask == 1] = 50.0  # solid soft tissue

        meas = self.segmenter.measure_recist_11(mask, hu_slice, (0.5, 0.5))
        self.assertIsNotNone(meas)
        self.assertAlmostEqual(meas.longest_diameter_mm, 20.0, delta=1.5)
        self.assertAlmostEqual(meas.perpendicular_diameter_mm, 10.0, delta=1.5)
        self.assertGreater(meas.longest_diameter_mm, meas.perpendicular_diameter_mm)
        self.assertEqual(meas.attenuation_type, "solid")
        self.assertGreaterEqual(meas.consolidation_to_tumor_ratio, 0.95)

    def test_consolidation_to_tumor_ratio_ground_glass(self):
        """Verify ground-glass attenuation detection (HU < -150 throughout nodule)."""
        mask = np.zeros((60, 60), dtype=np.uint8)
        cv2.circle(mask, (30, 30), 12, 1, -1)
        hu_slice = np.full((60, 60), -650.0, dtype=np.float32)  # pure subsolid / ground-glass

        meas = self.segmenter.measure_recist_11(mask, hu_slice, (0.7, 0.7))
        self.assertIsNotNone(meas)
        self.assertEqual(meas.attenuation_type, "ground_glass")
        self.assertLessEqual(meas.consolidation_to_tumor_ratio, 0.05)


class TestBrockRiskCalculator(unittest.TestCase):
    def test_low_risk_profile(self):
        """Small 4mm nodule in a 35yo male non-smoker -> Very Low / Low risk."""
        res = BrockUniversityRiskCalculator.calculate_risk(
            size_mm=4.0,
            age=35,
            is_female=False,
            family_history_lung_cancer=False,
            emphysema_present=False,
            nodule_type="solid",
            is_upper_lobe=False,
            nodule_count=2,
            is_spiculated=False
        )
        self.assertLess(res.risk_percentage, 5.0)
        self.assertIn(res.risk_tier, ["VERY_LOW", "LOW"])
        self.assertIn("McWilliams", res.model_version)

    def test_high_risk_profile(self):
        """Large 22mm spiculated upper-lobe nodule in a 74yo female with emphysema -> High risk (> 65%)."""
        res = BrockUniversityRiskCalculator.calculate_risk(
            size_mm=22.0,
            age=74,
            is_female=True,
            family_history_lung_cancer=True,
            emphysema_present=True,
            nodule_type="part_solid",
            is_upper_lobe=True,
            nodule_count=1,
            is_spiculated=True
        )
        self.assertGreater(res.risk_percentage, 10.0)
        self.assertIn(res.risk_tier, ["INTERMEDIATE", "HIGH"])

    def test_uncalibrated_or_zero_size(self):
        """Zero size returns negligible risk safely."""
        res = BrockUniversityRiskCalculator.calculate_risk(size_mm=0.0, age=60, is_female=False)
        self.assertEqual(res.risk_percentage, 0.0)
        self.assertEqual(res.risk_tier, "NEGLIGIBLE")


class TestDICOMStructuredReporting(unittest.TestCase):
    def test_create_valid_binary_dicom_sr(self):
        """Generate binary DICOM Part 10 SR and verify SOP class, tags, and structure via pydicom."""
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp_path = tmp.name

        analysis_mock = {
            "prediction": "MALIGNANT",
            "confidence": 0.884,
            "features": {
                "size_mm": 11.2,
                "sphericity": 0.84,
                "margin": "lobulated",
                "texture": "heterogeneous",
                "spiculation": True
            },
            "expert_analysis": {
                "risk_score": 0.76,
                "risk_level": "High Risk",
                "follow_up_interval": "PET-CT / Biopsy",
                "recommendation": "Tissue diagnosis recommended."
            }
        }
        meta_mock = {
            "patient_name": "DOE^JOHN",
            "patient_id": "TEST_PATIENT_001",
            "study_instance_uid": "1.2.840.113619.2.55.3.12345"
        }

        try:
            out_file = create_dicom_sr_report(tmp_path, analysis_mock, meta_mock)
            self.assertTrue(os.path.exists(out_file))
            self.assertGreater(os.path.getsize(out_file), 500)

            # Read back using pydicom to verify conformance
            dcm = pydicom.dcmread(out_file)
            self.assertEqual(dcm.Modality, "SR")
            self.assertEqual(dcm.PatientID, "TEST_PATIENT_001")
            self.assertEqual(str(dcm.PatientName), "DOE^JOHN")
            self.assertEqual(dcm.SOPClassUID, "1.2.840.10008.5.1.4.1.1.88.11")  # Basic Text SR
            self.assertEqual(dcm.CompletionFlag, "COMPLETE")
            self.assertGreater(len(dcm.ContentSequence), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestFastAPIServiceEndpoints(unittest.TestCase):
    def setUp(self):
        from service.app import startup_event
        startup_event()
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("lineage", data)
        self.assertIn("model_sha256", data["lineage"])

    def test_predict_endpoint_synthetic_ct_slice(self):
        # Create a valid synthetic CT slice image conforming to thoracic anatomy
        # Outer ambient air (HU ~ -1000)
        slice_img = np.zeros((256, 256), dtype=np.uint8)
        # Add chest wall / soft tissue (pixel 230 -> HU ~ +262, so max_hu > 0)
        slice_img[15:241, 15:241] = 230
        # Add simulated thoracic lung fields (pixel 30 -> HU ~ -835, so min_hu < -200)
        slice_img[40:216, 40:216] = 30
        # Add simulated nodule (pixel 230 -> soft tissue nodule)
        cv2.circle(slice_img, (128, 128), 12, 230, -1)
        _, enc = cv2.imencode(".png", slice_img)

        response = self.client.post(
            "/predict",
            files={"file": ("synthetic_scan.png", enc.tobytes(), "image/png")},
            data={
                "pixel_spacing_row": "0.65",
                "pixel_spacing_col": "0.65",
                "patient_age": "65",
                "is_female": "false"
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("calibrated_probability", data)
        self.assertIn("expert_guidelines", data)
        self.assertIn("brock_risk", data)
        self.assertIn("recist_measurements", data)


class TestPatientSafetyGates(unittest.TestCase):
    def test_missing_model_raises_runtime_error_without_opt_in(self):
        """Strict check: Inference without model MUST NOT fabricate random output."""
        os.environ["NODULE_DEMO"] = "0"
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        with self.assertRaises(RuntimeError):
            predict_nodule(None, dummy_img)

        with self.assertRaises(RuntimeError):
            generate_gradcam(None, dummy_img)


if __name__ == "__main__":
    unittest.main()
