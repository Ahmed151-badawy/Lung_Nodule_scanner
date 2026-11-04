"""
Professional Medical Desktop Application
Lung Nodule AI Analysis System - Medical PACS-Grade Interface
Refactored for Patient Safety, Thread-Safe Inference, and DICOM Spatial Calibration
"""

import os
import sys
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

# Clinical Core Imports - Fails loudly if misconfigured
from model.model import load_model, load_model_meta, predict_nodule, is_demo_mode_active
from explainability.gradcam import generate_gradcam
from expert_system.rules import apply_expert_rules, CLINICAL_DISCLAIMER
from expert_system.brock_model import BrockUniversityRiskCalculator
from preprocessing.nodule_segmenter import AnatomicalNoduleSegmenter
from reports.pdf_adapter import generate_pdf_report
from reports.dicom_sr import create_dicom_sr_report
from reports.lineage import ClinicalLineageTracker
from utils.image_utils import load_image_any, convert_to_hounsfield_units

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 🎨 COLOR PALETTE - Medical Professional (PACS standard)
# ═══════════════════════════════════════════════════════════════

class MedicalColors:
    PRIMARY = "#1A73E8"       # Medical Blue
    SECONDARY = "#4FC3F7"     # Soft Cyan
    SUCCESS = "#34A853"       # Clinical Green
    WARNING = "#FBBC04"       # Caution Amber
    DANGER = "#EA4335"        # Alert Crimson

    BG_MAIN = "#F1F3F4"       # Light Gray background
    BG_CARD = "#FFFFFF"       # Pure white cards
    BG_SIDEBAR = "#1E293B"    # Deep slate sidebar

    TEXT_PRIMARY = "#0F172A"  # Dark readable
    TEXT_SECONDARY = "#64748B"# Neutral slate
    TEXT_LIGHT = "#FFFFFF"    # Crisp white text

    BORDER = "#E2E8F0"        # Subtle card border
    HOVER = "#2563EB"         # Button hover state
    DEMO_BANNER = "#DC2626"   # Warning banner red


# ═══════════════════════════════════════════════════════════════
# 🏥 MAIN PACS APPLICATION
# ═══════════════════════════════════════════════════════════════

class MedicalLungAnalyzerGUI:
    """Professional PACS-style Medical GUI for Chest CT Nodule Evaluation"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.current_image_path: Optional[str] = None
        self.current_image: Optional[np.ndarray] = None
        self.image_metadata: Dict[str, Any] = {}
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.gradcam_photo: Optional[ImageTk.PhotoImage] = None
        self.analysis_result: Optional[Dict[str, Any]] = None
        self.model = None
        self.model_meta: Dict[str, Any] = {}
        self.view_mode = "original"  # "original" or "gradcam"
        self._is_analyzing = False

        self.demo_mode = is_demo_mode_active()

        self._setup_window()
        self._apply_styles()
        self._load_model_safe()
        self._create_interface()

    def _setup_window(self):
        """Configure main window dimensions and keybindings"""
        self.root.title("Clinical Lung Nodule AI Analysis System [Medical Edition]")
        
        # Apply clinical icon if available
        icon_path = Path("assets/nodule_ai.ico")
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except Exception:
                pass

        self.root.update_idletasks()

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        width = min(1440, int(screen_width * 0.90))
        height = min(900, int(screen_height * 0.90))

        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.configure(bg=MedicalColors.BG_MAIN)
        self.root.minsize(1100, 750)

        self.root.bind('<F11>', self._toggle_fullscreen)
        self.root.bind('<Escape>', self._exit_fullscreen)
        self.fullscreen_state = False

    def _apply_styles(self):
        """Configure ttk clinical styles"""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure(
            'Medical.TButton',
            font=('Inter', 10, 'bold'),
            padding=10,
            background=MedicalColors.PRIMARY,
            foreground=MedicalColors.TEXT_LIGHT
        )
        style.map(
            'Medical.TButton',
            background=[('active', MedicalColors.SECONDARY)]
        )
        style.configure('Card.TFrame', background=MedicalColors.BG_CARD, relief='flat')

    def _load_model_safe(self):
        """Safely attempt to load model checkpoint"""
        try:
            self.model = load_model("model/best_model.pth")
            self.model_meta = load_model_meta("model/best_model.pth")
            logger.info("Model loaded successfully into GUI.")
        except FileNotFoundError:
            if self.demo_mode:
                self.model = None
                logger.warning("GUI running in explicit DEMO_MODE (NODULE_DEMO=1). Model weights omitted.")
            else:
                self.model = None
                logger.warning("Model weights best_model.pth not found. Real inference will require model file.")
        except Exception as exc:
            logger.error("Error loading model: %s", exc)
            self.model = None

    def _create_interface(self):
        """Construct PACS desktop layout"""
        main_container = tk.Frame(self.root, bg=MedicalColors.BG_MAIN)
        main_container.pack(fill=tk.BOTH, expand=True)

        # 1. Top Safety/Demo Banner (if in demo mode)
        if self.demo_mode:
            self._create_demo_banner(main_container)

        # 2. PACS Header
        self._create_header(main_container)

        # 3. Content Area
        content_area = tk.Frame(main_container, bg=MedicalColors.BG_MAIN)
        content_area.pack(fill=tk.BOTH, expand=True)

        # Left Sidebar Navigation
        self._create_sidebar(content_area)

        # Main Workspace
        self._create_workspace(content_area)

        # Bottom PACS Status Bar
        self._create_statusbar(main_container)

    def _create_demo_banner(self, parent: tk.Widget):
        """Prominent clinical safety warning banner for demo/testing mode"""
        banner = tk.Frame(parent, bg=MedicalColors.DEMO_BANNER, height=30)
        banner.pack(fill=tk.X, side=tk.TOP)
        banner.pack_propagate(False)

        tk.Label(
            banner,
            text="⚠️ DEMO MODE ACTIVE – SIMULATED DATA FOR UI TESTING ONLY (NOT A CLINICAL DIAGNOSIS)",
            font=('Inter', 10, 'bold'),
            bg=MedicalColors.DEMO_BANNER,
            fg=MedicalColors.TEXT_LIGHT
        ).pack(fill=tk.BOTH, expand=True)

    def _create_header(self, parent: tk.Widget):
        """Header bar with title and institutional metadata"""
        header = tk.Frame(parent, bg=MedicalColors.PRIMARY, height=65)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title_frame = tk.Frame(header, bg=MedicalColors.PRIMARY)
        title_frame.pack(side=tk.LEFT, padx=25, pady=12)

        tk.Label(
            title_frame,
            text="🫁 Pulmonary Nodule AI PACS",
            font=('Inter', 18, 'bold'),
            bg=MedicalColors.PRIMARY,
            fg=MedicalColors.TEXT_LIGHT
        ).pack(side=tk.LEFT)

        tk.Label(
            title_frame,
            text="Clinical Assist v2.1",
            font=('Inter', 9),
            bg=MedicalColors.PRIMARY,
            fg=MedicalColors.SECONDARY
        ).pack(side=tk.LEFT, padx=(10, 0), pady=(4, 0))

        # Status badge on right
        status_frame = tk.Frame(header, bg=MedicalColors.PRIMARY)
        status_frame.pack(side=tk.RIGHT, padx=25, pady=12)

        status_text = "🟢 Model Active" if self.model else ("🟡 Demo Mode" if self.demo_mode else "🔴 Model Missing")
        tk.Label(
            status_frame,
            text=status_text,
            font=('Inter', 10, 'bold'),
            bg=MedicalColors.PRIMARY,
            fg=MedicalColors.TEXT_LIGHT
        ).pack(side=tk.RIGHT)

    def _create_sidebar(self, parent: tk.Widget):
        """Left PACS command panel"""
        sidebar = tk.Frame(parent, bg=MedicalColors.BG_SIDEBAR, width=260)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="CLINICAL WORKFLOW",
            font=('Inter', 10, 'bold'),
            bg=MedicalColors.BG_SIDEBAR,
            fg=MedicalColors.TEXT_SECONDARY,
            anchor='w',
            padx=20,
            pady=15
        ).pack(fill=tk.X)

        self.btn_load = self._make_sidebar_button(sidebar, "📂 Open CT Scan / DICOM", self.load_image)
        self.btn_analyze = self._make_sidebar_button(sidebar, "⚡ Run AI Inference", self.run_analysis)
        self.btn_explain = self._make_sidebar_button(sidebar, "🔥 Toggle Grad-CAM", self._toggle_view_mode)
        self.btn_report = self._make_sidebar_button(sidebar, "📄 Export Clinical PDF", self.generate_pdf)
        self.btn_dicom_sr = self._make_sidebar_button(sidebar, "💾 Export DICOM SR (Basic Text)", self.generate_dicom_sr)

        # Patient risk inputs for the Brock model (no defaults are assumed)
        self.var_age = tk.StringVar(value="")
        self.var_sex = tk.StringVar(value="")
        self.var_lobe = tk.StringVar(value="")
        self.var_count = tk.StringVar(value="1")
        self.var_fh = tk.BooleanVar(value=False)
        self.var_emph = tk.BooleanVar(value=False)
        self.var_spic = tk.BooleanVar(value=False)

        risk = tk.Frame(sidebar, bg=MedicalColors.BG_SIDEBAR)
        risk.pack(fill=tk.X, padx=15, pady=(18, 0))
        tk.Label(risk, text="PATIENT RISK INPUTS (Brock)", font=('Inter', 9, 'bold'),
                 bg=MedicalColors.BG_SIDEBAR, fg=MedicalColors.TEXT_SECONDARY).pack(anchor='w')

        def _row(label, widget_factory):
            row = tk.Frame(risk, bg=MedicalColors.BG_SIDEBAR)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, width=9, anchor='w', font=('Inter', 9),
                     bg=MedicalColors.BG_SIDEBAR, fg=MedicalColors.TEXT_LIGHT).pack(side=tk.LEFT)
            widget_factory(row).pack(side=tk.LEFT, fill=tk.X, expand=True)

        _row("Age", lambda r: tk.Entry(r, textvariable=self.var_age, width=6))
        _row("Sex", lambda r: tk.OptionMenu(r, self.var_sex, "Female", "Male"))
        _row("Lobe", lambda r: tk.OptionMenu(r, self.var_lobe, "Upper", "Middle/Lower"))
        _row("Nodules", lambda r: tk.Entry(r, textvariable=self.var_count, width=6))
        for text, var in (("Family hx lung cancer", self.var_fh),
                          ("Emphysema on CT", self.var_emph),
                          ("Spiculated (reader)", self.var_spic)):
            tk.Checkbutton(risk, text=text, variable=var, anchor='w', font=('Inter', 9),
                           bg=MedicalColors.BG_SIDEBAR, fg=MedicalColors.TEXT_LIGHT,
                           selectcolor=MedicalColors.BG_SIDEBAR,
                           activebackground=MedicalColors.BG_SIDEBAR).pack(fill=tk.X)
        tk.Label(risk, text="Risk score needs age, sex and lobe.", font=('Inter', 8),
                 bg=MedicalColors.BG_SIDEBAR, fg=MedicalColors.TEXT_SECONDARY).pack(anchor='w')

        # Metadata panel at bottom of sidebar
        self.meta_frame = tk.Frame(sidebar, bg=MedicalColors.BG_SIDEBAR)
        self.meta_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)

        tk.Label(
            self.meta_frame,
            text="DICOM CALIBRATION",
            font=('Inter', 9, 'bold'),
            bg=MedicalColors.BG_SIDEBAR,
            fg=MedicalColors.TEXT_SECONDARY
        ).pack(anchor='w')

        self.lbl_calibration = tk.Label(
            self.meta_frame,
            text="No CT scan loaded",
            font=('JetBrains Mono', 8),
            bg=MedicalColors.BG_SIDEBAR,
            fg=MedicalColors.TEXT_LIGHT,
            justify=tk.LEFT
        )
        self.lbl_calibration.pack(anchor='w', pady=(4, 0))

    def _make_sidebar_button(self, parent, text, command):
        btn = tk.Button(
            parent,
            text=text,
            font=('Inter', 10),
            bg=MedicalColors.BG_SIDEBAR,
            fg=MedicalColors.TEXT_LIGHT,
            activebackground=MedicalColors.PRIMARY,
            activeforeground=MedicalColors.TEXT_LIGHT,
            relief='flat',
            anchor='w',
            padx=20,
            pady=12,
            cursor='hand2',
            command=command
        )
        btn.pack(fill=tk.X)
        btn.bind('<Enter>', lambda e: btn.configure(bg=MedicalColors.HOVER))
        btn.bind('<Leave>', lambda e: btn.configure(bg=MedicalColors.BG_SIDEBAR))
        return btn

    def _create_workspace(self, parent: tk.Widget):
        """Construct central CT viewer and right-hand diagnostic results panel"""
        workspace = tk.Frame(parent, bg=MedicalColors.BG_MAIN)
        workspace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Center - CT Viewer Card
        viewer_card = tk.Frame(workspace, bg=MedicalColors.BG_CARD, relief='flat', bd=1)
        viewer_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # View mode selector bar
        view_bar = tk.Frame(viewer_card, bg=MedicalColors.BG_CARD)
        view_bar.pack(fill=tk.X, padx=15, pady=10)

        self.btn_original = tk.Button(
            view_bar,
            text="Original CT Slice",
            font=('Inter', 9, 'bold'),
            bg=MedicalColors.PRIMARY,
            fg=MedicalColors.TEXT_LIGHT,
            relief='flat',
            padx=12,
            pady=6,
            cursor='hand2',
            command=lambda: self._set_view_mode('original')
        )
        self.btn_original.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_gradcam = tk.Button(
            view_bar,
            text="Grad-CAM Saliency",
            font=('Inter', 9, 'bold'),
            bg=MedicalColors.BG_MAIN,
            fg=MedicalColors.TEXT_PRIMARY,
            relief='flat',
            padx=12,
            pady=6,
            cursor='hand2',
            command=lambda: self._set_view_mode('gradcam')
        )
        self.btn_gradcam.pack(side=tk.LEFT)

        # Image Canvas / Label
        self.viewer_frame = tk.Frame(viewer_card, bg="#000000")
        self.viewer_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        self.image_label = tk.Label(
            self.viewer_frame,
            text="No CT Scan Loaded\n\nClick 'Open CT Scan / DICOM' to begin.",
            font=('Inter', 12),
            bg="#000000",
            fg="#94A3B8"
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Right Panel - Structured Clinical Findings
        self.right_panel = tk.Frame(workspace, bg=MedicalColors.BG_CARD, width=440, bd=1)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.right_panel.pack_propagate(False)

        # Scrollable container for results
        self.results_canvas = tk.Canvas(self.right_panel, bg=MedicalColors.BG_CARD, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.right_panel, orient="vertical", command=self.results_canvas.yview)
        self.results_container = tk.Frame(self.results_canvas, bg=MedicalColors.BG_CARD)

        self.results_container.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        self.results_canvas.create_window((0, 0), window=self.results_container, anchor="nw", width=420)
        self.results_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._render_empty_results_state()

    def _create_statusbar(self, parent: tk.Widget):
        """Status bar at bottom of window"""
        self.statusbar = tk.Frame(parent, bg=MedicalColors.BG_SIDEBAR, height=28)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)
        self.statusbar.pack_propagate(False)

        self.lbl_status = tk.Label(
            self.statusbar,
            text="Ready. Awaiting scan input.",
            font=('Inter', 9),
            bg=MedicalColors.BG_SIDEBAR,
            fg=MedicalColors.TEXT_LIGHT,
            padx=15
        )
        self.lbl_status.pack(side=tk.LEFT)

    def update_status(self, message: str, level: str = "info"):
        color_map = {
            "info": MedicalColors.TEXT_LIGHT,
            "success": MedicalColors.SUCCESS,
            "warning": MedicalColors.WARNING,
            "error": MedicalColors.DANGER
        }
        self.lbl_status.config(text=message, fg=color_map.get(level, MedicalColors.TEXT_LIGHT))

    # ═══════════════════════════════════════════════════════════
    # 📂 CLINICAL FILE INGESTION
    # ═══════════════════════════════════════════════════════════

    def load_image(self):
        """Ingest CT scan (DICOM or standard image format)"""
        path = filedialog.askopenfilename(
            title="Open CT Scan / DICOM File",
            filetypes=[
                ("All Supported CT Files", "*.dcm *.png *.jpg *.jpeg *.tif"),
                ("DICOM Volumes (*.dcm)", "*.dcm"),
                ("Standard Image Slices (*.png, *.jpg)", "*.png *.jpg *.jpeg"),
                ("All Files", "*.*")
            ]
        )
        if not path:
            return

        try:
            self.update_status(f"Loading {Path(path).name}...", "info")
            img, meta = load_image_any(path)

            if img is None:
                raise ValueError("Could not decode image or parse DICOM headers.")

            self.current_image = img
            self.current_image_path = path
            self.image_metadata = meta
            self.analysis_result = None
            self.gradcam_photo = None

            # Update calibration UI
            spacing = meta.get('pixel_spacing')
            if spacing is not None:
                calib_str = f"Spacing: {spacing[0]:.2f} x {spacing[1]:.2f} mm\nPatient: {meta.get('patient_id')}\nModality: {meta.get('modality')}"
            else:
                calib_str = "Spacing: Uncalibrated (2D Image)\nSize in mm requires DICOM\nPatient: Anonymous"
            self.lbl_calibration.config(text=calib_str)

            self._display_image(self.current_image, "original")
            self._set_view_mode('original')
            self._render_empty_results_state()

            self.update_status(f"✓ Loaded {Path(path).name} successfully.", "success")

        except Exception as exc:
            logger.error("Failed to load scan: %s", exc, exc_info=True)
            messagebox.showerror("File Error", f"Unable to open medical image:\n{exc}")
            self.update_status("Failed to load image.", "error")

    # ═══════════════════════════════════════════════════════════
    # ⚡ THREAD-SAFE AI INFERENCE PIPELINE
    # ═══════════════════════════════════════════════════════════

    def run_analysis(self):
        """Trigger asynchronous AI prediction in a background worker thread"""
        if self.current_image is None:
            messagebox.showwarning("Input Required", "Please open a CT scan or DICOM slice first.")
            return

        if self._is_analyzing:
            return

        # Patient Safety Gate: Check if model exists or if demo mode explicitly opted in
        if self.model is None and not self.demo_mode:
            messagebox.showerror(
                "Model Weights Not Found",
                "Model checkpoint 'best_model.pth' is missing.\n\n"
                "PATIENT SAFETY POLICY:\n"
                "Inference cannot proceed without verified neural weights.\n\n"
                "To run simulated evaluation for UI demonstration, set the environment variable:\n"
                "NODULE_DEMO=1"
            )
            return

        self._is_analyzing = True
        self.btn_analyze.config(state=tk.DISABLED)
        self.update_status("Running AI neural inference & explainability...", "info")
        self._render_analyzing_state()

        # Snapshot Tk variables on the main thread (Tk is not thread-safe)
        self._risk_inputs = self._read_risk_inputs()

        # Execute in background thread to keep Tkinter GUI responsive
        worker = threading.Thread(target=self._analysis_worker, daemon=True)
        worker.start()

    def _read_risk_inputs(self) -> dict:
        """Validate patient inputs. Returns {} when Brock inputs are incomplete."""
        try:
            age = int(self.var_age.get().strip())
            count = int(self.var_count.get().strip() or "1")
        except ValueError:
            return {}
        sex, lobe = self.var_sex.get(), self.var_lobe.get()
        if not (18 <= age <= 100) or count < 1 or sex not in ("Female", "Male") or lobe not in ("Upper", "Middle/Lower"):
            return {}
        return {
            "age": age, "is_female": sex == "Female", "is_upper_lobe": lobe == "Upper",
            "nodule_count": count, "family_history_lung_cancer": self.var_fh.get(),
            "emphysema_present": self.var_emph.get(), "is_spiculated": self.var_spic.get(),
        }

    def _analysis_worker(self):
        """Background thread executing neural model and expert system"""
        try:
            spacing = self.image_metadata.get('pixel_spacing')

            # 1. Nodule segmentation & axial diameters (before inference so that
            #    shape features come from the nodule, not the whole slice)
            recist_meas = None
            nodule_mask = None
            hu_slice = self.image_metadata.get('hu_slice')
            try:
                if hu_slice is None:
                    if self.current_image.dtype == np.uint8:
                        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if self.current_image.ndim == 3 else self.current_image
                        hu_slice = (gray.astype(np.float32) / 255.0) * 1400.0 - 1000.0
                    else:
                        slope = float(self.image_metadata.get('rescale_slope', 1.0))
                        intercept = float(self.image_metadata.get('rescale_intercept', 0.0))
                        hu_slice = convert_to_hounsfield_units(self.current_image, slope, intercept)

                if spacing is not None:
                    segmenter = AnatomicalNoduleSegmenter()
                    nodule_mask = segmenter.segment_candidate_roi(hu_slice, spacing_mm=spacing)
                    if nodule_mask.sum() > 0:
                        recist_meas = segmenter.measure_recist_11(nodule_mask, hu_slice, spacing)
            except Exception as exc:
                logger.warning("Nodule measurement in GUI: %s", exc)

            # Crop-trained models classify ONE nodule: need its centre and the pixel spacing
            crop = None
            if self.model_meta.get("input_mode") == "crop2d":
                if nodule_mask is None or int(nodule_mask.sum()) == 0 or spacing is None:
                    raise ValueError("No nodule candidate located (or no pixel spacing); "
                                     "this model classifies a single nodule and cannot run on the whole slice.")
                mom = cv2.moments(nodule_mask.astype(np.uint8), binaryImage=True)
                crop = {"center_xy": (mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]), "spacing_mm": spacing,
                        "patch_mm": self.model_meta.get("patch_mm", 64.0), "out_px": self.model_meta.get("out_px", 64)}

            # 2. Prediction & Morphometrics
            pred_class, confidence, features = predict_nodule(
                self.model,
                self.current_image,
                spacing_mm=spacing,
                hu_slice=self.image_metadata.get('hu_slice'),
                nodule_mask=nodule_mask,
                crop=crop
            )

            # 3. Grad-CAM Visualization
            gradcam_overlay = generate_gradcam(
                self.model,
                self.current_image,
                target_class=pred_class,
                hu_slice=hu_slice,
                crop=crop
            )

            # 4. Brock University / PanCan Malignancy Model (needs real patient inputs
            #    AND a validated nodule measurement; otherwise not computed)
            brock_res = None
            risk_inputs = getattr(self, "_risk_inputs", {})
            if recist_meas is not None and risk_inputs:
                brock_res = BrockUniversityRiskCalculator.calculate_risk(
                    size_mm=recist_meas.longest_diameter_mm,   # Brock expects the maximum diameter
                    nodule_type=recist_meas.attenuation_type,
                    **risk_inputs
                )

            # 5. Fleischner Guidelines & Heuristic Expert Analysis
            rule_features = dict(features)
            if recist_meas and recist_meas.mean_recist_diameter_mm:
                rule_features["size_mm"] = recist_meas.mean_recist_diameter_mm

            expert_analysis = apply_expert_rules(
                prediction=pred_class,
                confidence=confidence,
                features=rule_features
            )

            result = {
                "prediction": "MALIGNANT" if pred_class == 1 else "BENIGN",
                "pred_int": pred_class,
                "confidence": confidence,
                "features": features,
                "recist": recist_meas,
                "brock": brock_res,
                "expert_analysis": expert_analysis,
                "image_path": self.current_image_path,
                "gradcam": gradcam_overlay,
                "metadata": self.image_metadata,
                "is_demo": self.demo_mode
            }

            # Post back to Tkinter event loop safely
            self.root.after(0, self._on_analysis_success, result)

        except Exception as exc:
            logger.error("Analysis worker error: %s", exc, exc_info=True)
            self.root.after(0, self._on_analysis_failure, str(exc))

    def _on_analysis_success(self, result: Dict[str, Any]):
        """Handler for successful background inference"""
        self._is_analyzing = False
        self.btn_analyze.config(state=tk.NORMAL)
        self.analysis_result = result

        # Cache Grad-CAM photo
        self.gradcam_photo = self._create_photo(result['gradcam'])

        self._display_results()
        self.update_status("✓ Analysis completed successfully.", "success")

    def _on_analysis_failure(self, error_message: str):
        """Handler for failed background inference"""
        self._is_analyzing = False
        self.btn_analyze.config(state=tk.NORMAL)
        self._render_empty_results_state()
        messagebox.showerror("Inference Error", f"AI analysis failed:\n{error_message}")
        self.update_status("Inference failed.", "error")

    # ═══════════════════════════════════════════════════════════
    # 📄 REPORT GENERATION
    # ═══════════════════════════════════════════════════════════

    def generate_pdf(self):
        """Export comprehensive clinical PDF report"""
        if self.analysis_result is None:
            messagebox.showwarning("Warning", "Run AI analysis prior to generating a report.")
            return

        if self.current_image is None:
            messagebox.showwarning("Warning", "Original image slice is not available.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Export Clinical PDF Report",
            defaultextension=".pdf",
            filetypes=[("PDF Documents (*.pdf)", "*.pdf")],
            initialfile=f"Lung_AI_Report_{Path(self.current_image_path).stem}.pdf"
        )
        if not save_path:
            return

        try:
            self.update_status("Generating PDF report...", "info")

            lineage_stamp = ClinicalLineageTracker.generate_provenance_stamp(
                model_path="model/best_model.pth",
                dicom_metadata=self.image_metadata
            )

            patient_info = {
                'id': self.image_metadata.get('patient_id', Path(self.current_image_path).stem),
                'name': 'Incidental Finding',
                'scan_date': self.image_metadata.get('study_date', datetime.now().strftime("%Y-%m-%d")),
                'date': datetime.now().strftime("%Y-%m-%d"),
                'lineage': lineage_stamp
            }

            success = generate_pdf_report(
                save_path=save_path,
                analysis_result=self.analysis_result,
                original_image=self.current_image,
                patient_info=patient_info
            )

            if success and Path(save_path).exists() and Path(save_path).stat().st_size > 0:
                messagebox.showinfo("Report Exported", f"Clinical PDF generated successfully:\n{save_path}")
                self.update_status("✓ PDF report generated.", "success")
            else:
                raise RuntimeError("Report generation adapter did not create a valid PDF.")

        except Exception as exc:
            logger.error("PDF generation failure: %s", exc, exc_info=True)
            messagebox.showerror("Report Error", f"Failed to generate clinical PDF:\n{exc}")
            self.update_status("Report export failed.", "error")

    def generate_dicom_sr(self):
        """Export a binary DICOM Basic Text Structured Report (.dcm)"""
        if self.analysis_result is None:
            messagebox.showwarning("Warning", "Run AI analysis prior to exporting DICOM SR.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Export DICOM Structured Report (Part 10)",
            defaultextension=".dcm",
            filetypes=[("DICOM Structured Report (*.dcm)", "*.dcm")],
            initialfile=f"SR_{Path(self.current_image_path).stem}.dcm"
        )
        if not save_path:
            return

        try:
            self.update_status("Generating binary DICOM SR...", "info")
            out_file = create_dicom_sr_report(
                save_path=save_path,
                analysis_result=self.analysis_result,
                dicom_metadata=self.image_metadata
            )
            if Path(out_file).exists() and Path(out_file).stat().st_size > 0:
                messagebox.showinfo("DICOM SR Exported", f"DICOM Basic Text SR saved:\n{out_file}")
                self.update_status("✓ DICOM SR exported.", "success")
            else:
                raise RuntimeError("DICOM SR file was not written.")
        except Exception as exc:
            logger.error("DICOM SR export error: %s", exc, exc_info=True)
            messagebox.showerror("Export Error", f"Failed to generate DICOM SR:\n{exc}")
            self.update_status("DICOM SR export failed.", "error")

    # ═══════════════════════════════════════════════════════════
    # 🎨 RENDERING & VIEW MODES
    # ═══════════════════════════════════════════════════════════

    def _display_image(self, img_bgr: np.ndarray, mode: str):
        """Render CT slice on screen"""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((850, 680), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)

        if mode == "original":
            self.current_photo = photo

        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

    def _create_photo(self, img_bgr: np.ndarray) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((850, 680), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_img)

    def _set_view_mode(self, mode: str):
        self.view_mode = mode
        if mode == "original":
            self.btn_original.config(bg=MedicalColors.PRIMARY, fg=MedicalColors.TEXT_LIGHT)
            self.btn_gradcam.config(bg=MedicalColors.BG_MAIN, fg=MedicalColors.TEXT_PRIMARY)
            if self.current_photo:
                self.image_label.config(image=self.current_photo)
                self.image_label.image = self.current_photo
        else:
            self.btn_gradcam.config(bg=MedicalColors.PRIMARY, fg=MedicalColors.TEXT_LIGHT)
            self.btn_original.config(bg=MedicalColors.BG_MAIN, fg=MedicalColors.TEXT_PRIMARY)
            if self.gradcam_photo:
                self.image_label.config(image=self.gradcam_photo)
                self.image_label.image = self.gradcam_photo
            else:
                messagebox.showinfo("Info", "Execute AI Inference first to compute Grad-CAM heatmaps.")
                self._set_view_mode('original')

    def _toggle_view_mode(self):
        new_mode = "gradcam" if self.view_mode == "original" else "original"
        self._set_view_mode(new_mode)

    def _render_empty_results_state(self):
        """Display placeholder when no analysis has run"""
        for w in self.results_container.winfo_children():
            w.destroy()

        card = tk.Frame(self.results_container, bg=MedicalColors.BG_CARD)
        card.pack(fill=tk.BOTH, expand=True, pady=40)

        tk.Label(
            card,
            text="Awaiting Analysis",
            font=('Inter', 14, 'bold'),
            bg=MedicalColors.BG_CARD,
            fg=MedicalColors.TEXT_SECONDARY
        ).pack(pady=(0, 8))

        tk.Label(
            card,
            text="Load a CT slice and click 'Run AI Inference'\nto extract physical features and guidelines.",
            font=('Inter', 10),
            bg=MedicalColors.BG_CARD,
            fg=MedicalColors.TEXT_SECONDARY,
            justify=tk.CENTER
        ).pack()

    def _render_analyzing_state(self):
        """Show progress spinner/notice"""
        for w in self.results_container.winfo_children():
            w.destroy()

        card = tk.Frame(self.results_container, bg=MedicalColors.BG_CARD)
        card.pack(fill=tk.BOTH, expand=True, pady=40)

        tk.Label(
            card,
            text="Processing Volume...",
            font=('Inter', 13, 'bold'),
            bg=MedicalColors.BG_CARD,
            fg=MedicalColors.PRIMARY
        ).pack(pady=(0, 10))

        prog = ttk.Progressbar(card, mode='indeterminate', length=200)
        prog.pack(pady=10)
        prog.start(15)

    def _display_results(self):
        """Render complete structured clinical results in right panel"""
        for w in self.results_container.winfo_children():
            w.destroy()

        res = self.analysis_result
        if not res:
            return

        expert = res.get('expert_analysis', {})
        features = res.get('features', {})
        is_demo = res.get('is_demo', False)

        # 1. Demo Banner Card (if demo)
        if is_demo:
            demo_card = tk.Frame(self.results_container, bg=MedicalColors.DEMO_BANNER, padx=10, pady=8)
            demo_card.pack(fill=tk.X, pady=(0, 10))
            tk.Label(
                demo_card,
                text="DEMO – SIMULATED DATA, NOT A DIAGNOSIS",
                font=('Inter', 9, 'bold'),
                bg=MedicalColors.DEMO_BANNER,
                fg=MedicalColors.TEXT_LIGHT
            ).pack()

        # 2. Classification Card
        class_card = tk.Frame(self.results_container, bg='#F8FAFC', relief='flat', bd=1, padx=15, pady=12)
        class_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(class_card, text="AI PREDICTION", font=('Inter', 9, 'bold'), bg='#F8FAFC', fg=MedicalColors.TEXT_SECONDARY).pack(anchor='w')

        pred_text = res['prediction']
        pred_color = MedicalColors.DANGER if pred_text == "MALIGNANT" else MedicalColors.SUCCESS

        tk.Label(
            class_card,
            text=pred_text,
            font=('Inter', 20, 'bold'),
            bg='#F8FAFC',
            fg=pred_color
        ).pack(anchor='w', pady=(2, 2))

        tk.Label(
            class_card,
            text=f"Confidence: {res['confidence']:.1%}",
            font=('Inter', 11),
            bg='#F8FAFC',
            fg=MedicalColors.TEXT_PRIMARY
        ).pack(anchor='w')

        # 3. Fleischner Guidelines & Recommendation Card
        fleisch_card = tk.Frame(self.results_container, bg='#F8FAFC', relief='flat', bd=1, padx=15, pady=12)
        fleisch_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(fleisch_card, text="FLEISCHNER 2017 GUIDELINES", font=('Inter', 9, 'bold'), bg='#F8FAFC', fg=MedicalColors.PRIMARY).pack(anchor='w')

        tk.Label(
            fleisch_card,
            text=f"Surveillance: {expert.get('follow_up_interval', 'N/A')}",
            font=('Inter', 11, 'bold'),
            bg='#F8FAFC',
            fg=MedicalColors.TEXT_PRIMARY
        ).pack(anchor='w', pady=(4, 4))

        tk.Label(
            fleisch_card,
            text=expert.get('recommendation', 'N/A'),
            font=('Inter', 9),
            bg='#F8FAFC',
            fg=MedicalColors.TEXT_SECONDARY,
            wraplength=380,
            justify=tk.LEFT
        ).pack(anchor='w')

        # 4. Quantitative Morphometric Features Card
        feat_card = tk.Frame(self.results_container, bg='#F8FAFC', relief='flat', bd=1, padx=15, pady=12)
        feat_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(feat_card, text="NODULE MORPHOLOGY & METRICS", font=('Inter', 9, 'bold'), bg='#F8FAFC', fg=MedicalColors.TEXT_SECONDARY).pack(anchor='w', pady=(0, 6))

        size_mm = features.get('size_mm')
        size_display = f"{size_mm:.1f} mm (Equiv. Diameter)" if size_mm is not None else "Uncalibrated (missing DICOM spacing)"

        recist = res.get('recist')
        brock = res.get('brock')

        metrics = [
            ("Physical Diameter", size_display),
            ("Sphericity Index", f"{features.get('sphericity', 0):.2f}"),
            ("Margin Definition", str(features.get('margin', 'N/A')).capitalize()),
            ("Internal Texture", str(features.get('texture', 'N/A')).capitalize()),
            ("Spiculation", "Present" if features.get('spiculation') else "Absent"),
            ("Heuristic Risk Index", f"{expert.get('risk_score', 0):.2f} ({expert.get('risk_level', 'N/A')})")
        ]

        if recist is not None:
            metrics.insert(1, ("RECIST 1.1 (Long × Perp)", f"{recist.longest_diameter_mm:.1f} × {recist.perpendicular_diameter_mm:.1f} mm"))
            metrics.insert(2, ("RECIST Mean Diameter", f"{recist.mean_recist_diameter_mm:.1f} mm"))
            metrics.insert(3, ("Consolidation (CTR)", f"{recist.consolidation_to_tumor_ratio:.2f} ({recist.attenuation_type})"))

        if brock is not None:
            metrics.append(("Brock Malignancy Risk", f"{brock.risk_percentage:.1f}% ({brock.risk_tier})"))

        for label, val in metrics:
            row = tk.Frame(feat_card, bg='#F8FAFC')
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, font=('Inter', 9), bg='#F8FAFC', fg=MedicalColors.TEXT_SECONDARY).pack(side=tk.LEFT)
            tk.Label(row, text=val, font=('Inter', 9, 'bold'), bg='#F8FAFC', fg=MedicalColors.TEXT_PRIMARY).pack(side=tk.RIGHT)

        # 5. Clinical Impression & Regulatory Disclaimer
        disc_card = tk.Frame(self.results_container, bg='#F1F5F9', padx=10, pady=8)
        disc_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            disc_card,
            text=CLINICAL_DISCLAIMER,
            font=('Inter', 7),
            bg='#F1F5F9',
            fg=MedicalColors.TEXT_SECONDARY,
            wraplength=380,
            justify=tk.LEFT
        ).pack()

    # Fullscreen helpers
    def _toggle_fullscreen(self, event=None):
        self.fullscreen_state = not self.fullscreen_state
        self.root.attributes("-fullscreen", self.fullscreen_state)

    def _exit_fullscreen(self, event=None):
        self.fullscreen_state = False
        self.root.attributes("-fullscreen", False)


def main():
    root = tk.Tk()
    app = MedicalLungAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
