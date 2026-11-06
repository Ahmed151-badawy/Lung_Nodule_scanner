"""
Nodule Analysis Workstation - research prototype (NOT a medical device).
Desktop UI for single-slice CT nodule measurement, Brock risk estimate, guideline lookup and
model visualisation. Built from ui/theme.py, ui/widgets.py and ui/viewer.py.
"""

import sys
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2

from model.model import load_model, load_model_meta, predict_nodule, is_demo_mode_active
from explainability.gradcam import generate_gradcam
from explainability.calibration_and_saliency import ModelCalibrator
from expert_system.rules import apply_expert_rules, CLINICAL_DISCLAIMER
from expert_system.brock_model import BrockUniversityRiskCalculator
from preprocessing.nodule_segmenter import AnatomicalNoduleSegmenter
from reports.pdf_adapter import generate_pdf_report
from reports.dicom_sr import create_dicom_sr_report
from reports.lineage import ClinicalLineageTracker
from utils.image_utils import load_image_any, convert_to_hounsfield_units

from ui.theme import T, Fonts, setup_ttk
from ui.widgets import FlatButton, Segmented, Card, Chip, kv_rows, BarGauge, Collapsible, Tooltip, ScrollFrame
from ui.viewer import ImageViewer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LEFT_W, RIGHT_W = 300, 410
TIER_KIND = {"VERY_LOW": "ok", "LOW": "ok", "INTERMEDIATE": "warn", "HIGH": "danger", "NEGLIGIBLE": "neutral"}

HINT = "Click: set seed  ·  Wheel: zoom  ·  Drag: pan  ·  Right-click: clear"


class NoduleWorkstation:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.current_image_path: Optional[str] = None
        self.current_image: Optional[np.ndarray] = None
        self.image_metadata: Dict[str, Any] = {}
        self.analysis_result: Optional[Dict[str, Any]] = None
        self.gradcam_img: Optional[np.ndarray] = None
        self.model = None
        self.model_meta: Dict[str, Any] = {}
        self.temperature: float = 1.0
        self.calibrated = False
        self.seed: Optional[Tuple[int, int]] = None
        self.view_mode = "ct"
        self._is_analyzing = False
        self._risk_inputs: Dict[str, Any] = {}
        self.demo_mode = is_demo_mode_active()
        self.fullscreen_state = False

        Fonts.init(root)
        self._setup_window()
        setup_ttk(root)
        self._load_model_safe()
        self._build()
        self._build_menu()
        self._bind_keys()
        self._render_empty_results()
        self._refresh_risk_hint()
        self._sync_buttons()

    # ═════════════════════════════════════════════════════════ setup
    def _setup_window(self):
        self.root.title("Nodule Analysis Workstation — Research Prototype")
        icon = Path("assets/nodule_ai.ico")
        if icon.exists():
            try:
                self.root.iconbitmap(str(icon))
            except Exception:
                pass
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = min(1480, int(sw * 0.92)), min(920, int(sh * 0.92))
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        self.root.minsize(1180, 720)
        self.root.configure(bg=T.APP_BG)

    def _load_model_safe(self):
        try:
            self.model = load_model("model/best_model.pth")
            self.model_meta = load_model_meta("model/best_model.pth")
            logger.info("Model loaded successfully into GUI.")
        except FileNotFoundError:
            self.model = None
            logger.warning("Model weights not found%s.", " (demo mode)" if self.demo_mode else "")
        except Exception as exc:
            logger.error("Error loading model: %s", exc)
            self.model = None
        try:   # temperature scaling fitted on validation data (identical to the service)
            cal = ModelCalibrator.load_calibrator("model/calibrator.pth")
            self.temperature, self.calibrated = float(cal.temperature.item()), True
        except Exception as exc:
            logger.warning("No calibrator loaded (%s); probabilities are uncalibrated.", exc)

    def _model_status(self) -> Tuple[str, str, str]:
        if self.model is None:
            return (("Demo mode — simulated output", "warn", "NODULE_DEMO=1: outputs are simulated.") if self.demo_mode
                    else ("No model weights", "danger", "model/best_model.pth not found; analysis is disabled."))
        if self.model_meta:
            auc = (self.model_meta.get("test_metrics", {}).get("AUROC (CNN)", {}) or {}).get("value")
            tip = f"{self.model_meta.get('data', '?')} · test AUROC {auc:.2f}" if auc else self.model_meta.get("data", "")
            return "Model: LIDC-trained (research)", "ok", tip
        return ("Model: prototype weights (synthetic)", "warn",
                "Trained on synthetic drawings. Outputs are not clinically meaningful.")

    # ═════════════════════════════════════════════════════════ layout
    def _build(self):
        r = self.root
        r.columnconfigure(0, weight=1)
        r.rowconfigure(2, weight=1)
        self._build_topbar(r).grid(row=0, column=0, sticky="ew")
        if self.demo_mode:
            tk.Label(r, text="DEMO MODE — simulated output for UI testing only", font=Fonts.small_b, bg=T.WARN_SOFT,
                     fg=T.WARN, pady=4).grid(row=1, column=0, sticky="ew")
        body = tk.Frame(r, bg=T.APP_BG)
        body.grid(row=2, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        self._build_left(body).grid(row=0, column=0, sticky="ns")
        self._build_center(body).grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        self._build_right(body).grid(row=0, column=2, sticky="ns")
        self._build_statusbar(r).grid(row=3, column=0, sticky="ew")

    def _build_topbar(self, parent):
        bar = tk.Frame(parent, bg=T.NAV_BG, height=52)
        bar.pack_propagate(False)
        logo = tk.Canvas(bar, width=30, height=30, bg=T.NAV_BG, highlightthickness=0)
        logo.pack(side=tk.LEFT, padx=(18, 10), pady=11)
        logo.create_rectangle(1, 1, 29, 29, fill=T.ACCENT, outline="")
        logo.create_oval(6, 7, 14, 23, fill="#FFFFFF", outline="")
        logo.create_oval(16, 7, 24, 23, fill="#FFFFFF", outline="")
        logo.create_oval(13, 12, 17, 16, fill=T.ACCENT, outline="")
        titles = tk.Frame(bar, bg=T.NAV_BG)
        titles.pack(side=tk.LEFT)
        tk.Label(titles, text="Nodule Analysis Workstation", font=Fonts.title, bg=T.NAV_BG, fg=T.NAV_FG).pack(anchor="w")
        tk.Label(titles, text="Research prototype · single-slice CT", font=Fonts.micro, bg=T.NAV_BG, fg=T.NAV_MUTED).pack(anchor="w")
        right = tk.Frame(bar, bg=T.NAV_BG)
        right.pack(side=tk.RIGHT, padx=18)
        text, kind, tip = self._model_status()
        self.chip_model = Chip(right, text, kind)
        self.chip_model.pack(side=tk.RIGHT, padx=(8, 0))
        Tooltip(self.chip_model, tip)
        Chip(right, "RESEARCH USE ONLY", "warn").pack(side=tk.RIGHT)
        return bar

    def _section(self, parent, title: str, first: bool = False):
        tk.Label(parent, text=title, font=Fonts.micro_b, bg=T.SURFACE, fg=T.MUTED, anchor="w").pack(
            fill=tk.X, padx=18, pady=(14 if first else 20, 8))
        f = tk.Frame(parent, bg=T.SURFACE)
        f.pack(fill=tk.X, padx=18)
        return f

    def _build_left(self, parent):
        wrap = tk.Frame(parent, bg=T.BORDER)
        left = tk.Frame(wrap, bg=T.SURFACE, width=LEFT_W)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        tk.Frame(wrap, bg=T.BORDER, width=1).pack(side=tk.RIGHT, fill=tk.Y)

        # --- case
        case = self._section(left, "CASE", first=True)
        self.btn_open = FlatButton(case, "Open CT slice…", self.load_image, variant="primary", pady=10)
        self.btn_open.pack(fill=tk.X)
        Tooltip(self.btn_open, "DICOM (.dcm) or image slice   ·   Ctrl+O")
        self.lbl_file = tk.Label(case, text="No file loaded", font=Fonts.body_b, bg=T.SURFACE, fg=T.TEXT, anchor="w",
                                 wraplength=LEFT_W - 40, justify="left")
        self.lbl_file.pack(fill=tk.X, pady=(12, 4))
        self.var_spacing = tk.StringVar(value="—")
        self.var_seed = tk.StringVar(value="not set")
        grid = tk.Frame(case, bg=T.SURFACE)
        grid.pack(fill=tk.X)
        grid.columnconfigure(1, weight=1)
        for i, (lab, var) in enumerate((("Pixel spacing", self.var_spacing), ("Nodule seed", self.var_seed))):
            tk.Label(grid, text=lab, font=Fonts.small, bg=T.SURFACE, fg=T.MUTED, anchor="w").grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(grid, textvariable=var, font=Fonts.small_b, bg=T.SURFACE, fg=T.TEXT, anchor="e").grid(row=i, column=1, sticky="e", pady=2)

        # --- patient details
        pat = self._section(left, "PATIENT DETAILS  ·  for Brock estimate")
        self.var_age, self.var_sex, self.var_lobe, self.var_count = (tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(value="1"))
        self.var_fh, self.var_emph, self.var_spic = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        form = tk.Frame(pat, bg=T.SURFACE)
        form.pack(fill=tk.X)
        form.columnconfigure(1, weight=1)

        def row(i, label, widget):
            tk.Label(form, text=label, font=Fonts.small, bg=T.SURFACE, fg=T.MUTED, anchor="w", width=9).grid(row=i, column=0, sticky="w", pady=3)
            widget.grid(row=i, column=1, sticky="ew", pady=3)

        self.ent_age = ttk.Entry(form, textvariable=self.var_age, width=8, font=Fonts.body)
        row(0, "Age (years)", self.ent_age)
        row(1, "Sex", ttk.Combobox(form, textvariable=self.var_sex, values=("Female", "Male"), state="readonly", font=Fonts.body, width=14))
        row(2, "Lobe", ttk.Combobox(form, textvariable=self.var_lobe, values=("Upper", "Middle / Lower"), state="readonly", font=Fonts.body, width=14))
        row(3, "Nodules", ttk.Spinbox(form, textvariable=self.var_count, from_=1, to=30, width=6, font=Fonts.body))
        for i, (txt, var) in enumerate((("Family history of lung cancer", self.var_fh), ("Emphysema on CT", self.var_emph),
                                        ("Spiculated (reader's assessment)", self.var_spic))):
            ttk.Checkbutton(pat, text=txt, variable=var, style="TCheckbutton").pack(anchor="w", pady=(6 if i == 0 else 1, 0))
        for v in (self.var_age, self.var_sex, self.var_lobe, self.var_count):
            v.trace_add("write", lambda *_: self._refresh_risk_hint())
        self.chip_risk = Chip(pat, "Needs age, sex, lobe", "neutral")
        self.chip_risk.pack(anchor="w", pady=(10, 0))

        # --- actions (bottom)
        actions = tk.Frame(left, bg=T.SURFACE)
        actions.pack(side=tk.BOTTOM, fill=tk.X, padx=18, pady=18)
        self.btn_run = FlatButton(actions, "Run analysis", self.run_analysis, variant="primary", pady=12)
        self.btn_run.pack(fill=tk.X)
        Tooltip(self.btn_run, "Ctrl+R")
        return wrap

    def _build_center(self, parent):
        card = tk.Frame(parent, bg=T.SURFACE, highlightthickness=1, highlightbackground=T.BORDER)
        bar = tk.Frame(card, bg=T.SURFACE)
        bar.pack(fill=tk.X, padx=14, pady=10)
        self.seg_view = Segmented(bar, (("ct", "CT slice"), ("saliency", "Model saliency")), self._set_view_mode)
        self.seg_view.pack(side=tk.LEFT)
        btn_fit = FlatButton(bar, "Fit", lambda: self.viewer.fit(), variant="ghost", pady=4, padx=10)
        btn_fit.pack(side=tk.RIGHT)
        Tooltip(btn_fit, "Fit to window   ·   F")
        self.viewer = ImageViewer(card, on_seed=self._set_seed, on_clear_seed=self._clear_seed, on_hover=self._on_hover)
        self.viewer.pack(fill=tk.BOTH, expand=True, padx=1)
        tk.Label(card, text=HINT, font=Fonts.micro, bg=T.SURFACE, fg=T.MUTED, pady=7).pack(fill=tk.X)
        return card

    def _build_right(self, parent):
        wrap = tk.Frame(parent, bg=T.BORDER)
        tk.Frame(wrap, bg=T.BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)
        panel = tk.Frame(wrap, bg=T.APP_BG, width=RIGHT_W)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)
        head = tk.Frame(panel, bg=T.APP_BG)
        head.pack(fill=tk.X, padx=16, pady=(14, 6))
        tk.Label(head, text="Findings", font=Fonts.title, bg=T.APP_BG, fg=T.TEXT).pack(side=tk.LEFT)
        self.chip_state = Chip(head, "Awaiting analysis", "neutral")
        self.chip_state.pack(side=tk.RIGHT)
        foot = tk.Frame(panel, bg=T.SURFACE_ALT, highlightthickness=1, highlightbackground=T.BORDER)
        foot.pack(side=tk.BOTTOM, fill=tk.X)
        exp = tk.Frame(panel, bg=T.APP_BG)
        exp.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(4, 12))
        exp.columnconfigure((0, 1), weight=1, uniform="ex")
        self.btn_pdf = FlatButton(exp, "Export PDF", self.generate_pdf, pady=8)
        self.btn_pdf.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn_sr = FlatButton(exp, "Export DICOM SR", self.generate_dicom_sr, pady=8)
        self.btn_sr.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        tk.Label(foot, text=CLINICAL_DISCLAIMER, font=Fonts.micro, bg=T.SURFACE_ALT, fg=T.MUTED, wraplength=RIGHT_W - 36,
                 justify="left", padx=14, pady=10).pack(fill=tk.X)
        self.results = ScrollFrame(panel, bg=T.APP_BG)
        self.results.pack(fill=tk.BOTH, expand=True, padx=(16, 10), pady=(0, 8))
        return wrap

    def _build_statusbar(self, parent):
        bar = tk.Frame(parent, bg=T.NAV_BG, height=28)
        bar.pack_propagate(False)
        self.dot = tk.Canvas(bar, width=10, height=10, bg=T.NAV_BG, highlightthickness=0)
        self.dot.pack(side=tk.LEFT, padx=(14, 6))
        self._dot_id = self.dot.create_oval(1, 1, 9, 9, fill=T.NAV_MUTED, outline="")
        self.lbl_status = tk.Label(bar, text="Ready.", font=Fonts.small, bg=T.NAV_BG, fg=T.NAV_FG)
        self.lbl_status.pack(side=tk.LEFT)
        self.lbl_cursor = tk.Label(bar, text="", font=Fonts.mono, bg=T.NAV_BG, fg=T.NAV_MUTED, padx=14)
        self.lbl_cursor.pack(side=tk.RIGHT)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=140, style="Accent.Horizontal.TProgressbar")
        return bar

    def _build_menu(self):
        m = tk.Menu(self.root)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Open CT slice…", accelerator="Ctrl+O", command=self.load_image)
        f.add_separator()
        f.add_command(label="Export PDF report…", command=self.generate_pdf)
        f.add_command(label="Export DICOM SR…", command=self.generate_dicom_sr)
        f.add_separator()
        f.add_command(label="Quit", command=self.root.destroy)
        v = tk.Menu(m, tearoff=0)
        v.add_command(label="Fit to window", accelerator="F", command=lambda: self.viewer.fit())
        v.add_command(label="Toggle saliency", accelerator="G", command=self._toggle_view_mode)
        v.add_command(label="Clear nodule seed", accelerator="Del", command=self._clear_seed)
        v.add_command(label="Full screen", accelerator="F11", command=self._toggle_fullscreen)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Intended use & limitations", command=self._show_limitations)
        m.add_cascade(label="File", menu=f)
        m.add_cascade(label="View", menu=v)
        m.add_cascade(label="Help", menu=h)
        self.root.config(menu=m)

    def _bind_keys(self):
        r = self.root

        def guarded(fn):
            def h(e):
                if e.widget.winfo_class() in ("TEntry", "TSpinbox", "TCombobox", "Entry"):
                    return
                fn()
            return h

        r.bind("<Control-o>", lambda e: self.load_image())
        r.bind("<Control-r>", lambda e: self.run_analysis())
        r.bind("<F11>", self._toggle_fullscreen)
        r.bind("<Escape>", self._exit_fullscreen)
        r.bind("<KeyPress-g>", guarded(self._toggle_view_mode))
        r.bind("<KeyPress-f>", guarded(lambda: self.viewer.fit()))
        r.bind("<Delete>", guarded(self._clear_seed))

    # ═════════════════════════════════════════════════════════ small helpers
    def update_status(self, message: str, level: str = "info"):
        colors = {"info": T.NAV_MUTED, "success": "#4ADE80", "warning": "#FBBF24", "error": "#F87171", "busy": T.OVERLAY}
        self.dot.itemconfigure(self._dot_id, fill=colors.get(level, T.NAV_MUTED))
        self.lbl_status.config(text=message)

    def _sync_buttons(self):
        have_img = self.current_image is not None
        self.btn_run.set_enabled(have_img and not self._is_analyzing)
        done = self.analysis_result is not None
        self.btn_pdf.set_enabled(done)
        self.btn_sr.set_enabled(done)
        self.seg_view.set_enabled("saliency", self.gradcam_img is not None)

    def _refresh_risk_hint(self):
        r = self._read_risk_inputs()
        if r:
            self.chip_risk.update_chip("Brock inputs complete", "ok")
        else:
            missing = [n for n, v in (("age", self.var_age.get().strip()), ("sex", self.var_sex.get()), ("lobe", self.var_lobe.get())) if not v]
            self.chip_risk.update_chip("Needs " + ", ".join(missing) if missing else "Check age (18–100) and nodule count", "neutral")

    def _read_risk_inputs(self) -> dict:
        """Validated patient inputs; {} if incomplete or invalid (Brock then isn't computed)."""
        try:
            age = int(self.var_age.get().strip())
            count = int(self.var_count.get().strip() or "1")
        except ValueError:
            return {}
        sex, lobe = self.var_sex.get(), self.var_lobe.get()
        if not (18 <= age <= 100) or count < 1 or sex not in ("Female", "Male") or lobe not in ("Upper", "Middle / Lower"):
            return {}
        return {"age": age, "is_female": sex == "Female", "is_upper_lobe": lobe == "Upper", "nodule_count": count,
                "family_history_lung_cancer": self.var_fh.get(), "emphysema_present": self.var_emph.get(),
                "is_spiculated": self.var_spic.get()}

    def _set_seed(self, x: int, y: int):
        self.seed = (x, y)
        self.var_seed.set(f"x {x}, y {y}")
        self.viewer.set_seed(self.seed)
        if self.view_mode != "ct":
            self._set_view_mode("ct")
        self.update_status("Seed placed. Run analysis (Ctrl+R) to measure this nodule.", "info")

    def _clear_seed(self):
        self.seed = None
        self.var_seed.set("not set")
        self.viewer.set_seed(None)

    def _on_hover(self, pos):
        if pos is None or self.current_image is None:
            self.lbl_cursor.config(text="")
            return
        x, y = pos
        hu = self.image_metadata.get("hu_slice")
        if hu is not None and 0 <= y < hu.shape[0] and 0 <= x < hu.shape[1]:
            self.lbl_cursor.config(text=f"x {x:>4}   y {y:>4}   HU {hu[y, x]:>7.0f}")
        else:
            self.lbl_cursor.config(text=f"x {x:>4}   y {y:>4}")

    def _set_view_mode(self, mode: str):
        if mode == "saliency" and self.gradcam_img is None:
            self.update_status("Run the analysis first to compute the saliency map.", "warning")
            mode = "ct"
        self.view_mode = mode
        self.seg_view.select(mode)
        if mode == "ct":
            self.viewer.set_image(self.current_image, spatial=True, keep_view=True)
        else:
            res = self.analysis_result or {}
            if res.get("gradcam_is_patch"):
                self.viewer.set_image(self.gradcam_img, spatial=False,
                                      caption=f"Saliency · {self.model_meta.get('patch_mm', 64):.0f} mm patch around nodule")
            else:
                self.viewer.set_image(self.gradcam_img, spatial=True, keep_view=True)

    def _toggle_view_mode(self):
        self._set_view_mode("saliency" if self.view_mode == "ct" else "ct")

    def _toggle_fullscreen(self, event=None):
        self.fullscreen_state = not self.fullscreen_state
        self.root.attributes("-fullscreen", self.fullscreen_state)

    def _exit_fullscreen(self, event=None):
        self.fullscreen_state = False
        self.root.attributes("-fullscreen", False)

    def _show_limitations(self):
        win = tk.Toplevel(self.root)
        win.title("Intended use & limitations")
        win.configure(bg=T.SURFACE)
        win.transient(self.root)
        win.resizable(False, False)
        meta = self.model_meta
        data = meta.get("data") if meta else "synthetic drawings (prototype weights)"
        lines = [
            ("Research prototype — not a medical device", True),
            ("Not validated and not for diagnosis, screening or treatment decisions. Every output must be reviewed by a qualified radiologist.", False),
            ("What you are looking at", True),
            (f"•  Measurements come from a threshold-based segmentation tested only on synthetic slices.\n"
             f"•  The classifier was trained on: {data}. "
             f"{'Labels are radiologist impressions, not pathology.' if meta else 'Its output is not clinically meaningful.'}\n"
             f"•  Brock: published model for screening-detected nodules of about 1–30 mm; uses the patient details you enter.\n"
             f"•  Fleischner 2017: lookup for incidental nodules in adults ≥35 years without known cancer or immunosuppression.\n"
             f"•  Single 2D slice only: no volumetry, no comparison with prior scans.", False),
        ]
        body = tk.Frame(win, bg=T.SURFACE, padx=24, pady=20)
        body.pack()
        for text, head in lines:
            tk.Label(body, text=text, font=Fonts.title if head else Fonts.body, bg=T.SURFACE, fg=T.TEXT if head else T.MUTED,
                     wraplength=520, justify="left", anchor="w").pack(fill=tk.X, pady=(10 if head else 2, 2))
        FlatButton(body, "Close", win.destroy, variant="primary", padx=24).pack(anchor="e", pady=(16, 0))
        win.update_idletasks()
        win.geometry(f"+{self.root.winfo_rootx() + 180}+{self.root.winfo_rooty() + 120}")

    # ═════════════════════════════════════════════════════════ file ingestion
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Open CT slice",
            filetypes=[("CT slices", "*.dcm *.png *.jpg *.jpeg *.tif"), ("DICOM", "*.dcm"),
                       ("Images", "*.png *.jpg *.jpeg *.tif"), ("All files", "*.*")])
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        try:
            self.update_status(f"Loading {Path(path).name}…", "busy")
            img, meta = load_image_any(path)
            if img is None:
                raise ValueError("Could not decode image or parse DICOM headers.")
            self.current_image, self.current_image_path, self.image_metadata = img, path, meta
            self.analysis_result, self.gradcam_img = None, None
            self._clear_seed()
            sp = meta.get("pixel_spacing")
            self.lbl_file.config(text=Path(path).name)
            self.var_spacing.set(f"{sp[0]:.2f} × {sp[1]:.2f} mm" if sp is not None else "uncalibrated")
            self.viewer.set_spacing(tuple(sp) if sp is not None else None)
            h, w = img.shape[:2]
            self.viewer.set_info([Path(path).name, f"{w} × {h} px", f"{meta.get('patient_id') or 'anonymous'} · {meta.get('modality') or '—'}"])
            self.viewer.set_overlay(None)
            self._set_view_mode("ct")
            self.viewer.set_image(self.current_image, spatial=True)
            self._render_empty_results(loaded=True)
            self.chip_state.update_chip("Ready to analyse", "info")
            self._sync_buttons()
            self.update_status(f"Loaded {Path(path).name}. Click the nodule to set a seed, then run the analysis.", "success")
        except Exception as exc:
            logger.error("Failed to load scan: %s", exc, exc_info=True)
            messagebox.showerror("Cannot open file", f"Unable to open medical image:\n{exc}")
            self.update_status("Failed to load image.", "error")

    # ═════════════════════════════════════════════════════════ analysis
    def run_analysis(self):
        if self._is_analyzing:
            return
        if self.current_image is None:
            messagebox.showwarning("Input required", "Open a CT slice first.")
            return
        if self.model is None and not self.demo_mode:
            messagebox.showerror(
                "Model weights not found",
                "model/best_model.pth is missing, so inference cannot run.\n\n"
                "Train a model (scripts/train_lidc.py) or set NODULE_DEMO=1 to explore the UI with simulated output.")
            return
        self._is_analyzing = True
        self._risk_inputs = self._read_risk_inputs()      # snapshot on the main thread (Tk is not thread-safe)
        self._seed_snapshot = self.seed
        self._sync_buttons()
        self.chip_state.update_chip("Analysing…", "accent")
        self.update_status("Analysing…", "busy")
        self.progress.pack(side=tk.RIGHT, padx=8)
        self.progress.start(12)
        self._render_analyzing()
        threading.Thread(target=self._analysis_worker, daemon=True).start()

    def _analysis_worker(self):
        try:
            spacing = self.image_metadata.get('pixel_spacing')
            recist_meas, nodule_mask = None, None
            hu_slice = self.image_metadata.get('hu_slice')
            try:
                if hu_slice is None:
                    if self.current_image.dtype == np.uint8:
                        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY) if self.current_image.ndim == 3 else self.current_image
                        hu_slice = (gray.astype(np.float32) / 255.0) * 1400.0 - 1000.0
                    else:
                        hu_slice = convert_to_hounsfield_units(self.current_image, float(self.image_metadata.get('rescale_slope', 1.0)),
                                                               float(self.image_metadata.get('rescale_intercept', 0.0)))
                if spacing is not None:
                    segmenter = AnatomicalNoduleSegmenter()
                    nodule_mask = segmenter.segment_candidate_roi(hu_slice, seed_point=self._seed_snapshot, spacing_mm=spacing)
                    if nodule_mask.sum() > 0:
                        recist_meas = segmenter.measure_recist_11(nodule_mask, hu_slice, spacing)
            except Exception as exc:
                logger.warning("Nodule measurement in GUI: %s", exc)

            crop = None
            if self.model_meta.get("input_mode") == "crop2d":
                if nodule_mask is None or int(nodule_mask.sum()) == 0 or spacing is None:
                    raise ValueError("No nodule located (or no pixel spacing). Click on the nodule to set a seed — "
                                     "this model classifies a single nodule, not the whole slice.")
                mom = cv2.moments(nodule_mask.astype(np.uint8), binaryImage=True)
                crop = {"center_xy": (mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]), "spacing_mm": spacing,
                        "patch_mm": self.model_meta.get("patch_mm", 64.0), "out_px": self.model_meta.get("out_px", 64)}

            pred_class, confidence, features = predict_nodule(
                self.model, self.current_image, spacing_mm=spacing, temperature=self.temperature,
                hu_slice=self.image_metadata.get('hu_slice'), nodule_mask=nodule_mask, crop=crop)

            gradcam_overlay = generate_gradcam(self.model, self.current_image, target_class=pred_class, hu_slice=hu_slice, crop=crop)

            brock_res = None
            if recist_meas is not None and self._risk_inputs:
                brock_res = BrockUniversityRiskCalculator.calculate_risk(
                    size_mm=recist_meas.longest_diameter_mm, nodule_type=recist_meas.attenuation_type, **self._risk_inputs)

            rule_features = dict(features)
            if recist_meas and recist_meas.mean_recist_diameter_mm:
                rule_features["size_mm"] = recist_meas.mean_recist_diameter_mm
            expert_analysis = apply_expert_rules(prediction=pred_class, confidence=confidence, features=rule_features)

            result = {
                "prediction": "MALIGNANT" if pred_class == 1 else "BENIGN", "pred_int": pred_class, "confidence": confidence,
                "features": features, "recist": recist_meas, "brock": brock_res, "expert_analysis": expert_analysis,
                "image_path": self.current_image_path, "gradcam": gradcam_overlay, "metadata": self.image_metadata,
                "is_demo": self.demo_mode, "nodule_mask": nodule_mask, "gradcam_is_patch": crop is not None,
                "temperature": self.temperature, "calibrated": self.calibrated, "seed": self._seed_snapshot,
                "had_patient_inputs": bool(self._risk_inputs),
            }
            self.root.after(0, self._on_analysis_success, result)
        except Exception as exc:
            logger.error("Analysis worker error: %s", exc, exc_info=True)
            self.root.after(0, self._on_analysis_failure, str(exc))

    def _end_busy(self):
        self._is_analyzing = False
        self.progress.stop()
        self.progress.pack_forget()

    def _on_analysis_success(self, result: Dict[str, Any]):
        self._end_busy()
        self.analysis_result = result
        self.gradcam_img = result["gradcam"]
        rec = result.get("recist")
        label = f"{rec.longest_diameter_mm:.1f} × {rec.perpendicular_diameter_mm:.1f} mm" if rec else None
        self.viewer.set_overlay(result.get("nodule_mask"), label)
        self._set_view_mode("ct")
        self._display_results()
        self.chip_state.update_chip("Demo output" if result["is_demo"] else "Research output", "warn")
        self._sync_buttons()
        self.update_status("Analysis complete." if rec else "Analysis complete — no nodule was measured.", "success" if rec else "warning")

    def _on_analysis_failure(self, error_message: str):
        self._end_busy()
        self._render_empty_results(loaded=self.current_image is not None)
        self.chip_state.update_chip("Analysis failed", "danger")
        self._sync_buttons()
        messagebox.showerror("Analysis failed", error_message)
        self.update_status("Analysis failed.", "error")

    # ═════════════════════════════════════════════════════════ exports
    def generate_pdf(self):
        if self.analysis_result is None or self.current_image is None:
            messagebox.showwarning("Nothing to export", "Run the analysis before exporting a report.")
            return
        save_path = filedialog.asksaveasfilename(
            title="Export PDF report", defaultextension=".pdf", filetypes=[("PDF documents", "*.pdf")],
            initialfile=f"Nodule_Report_{Path(self.current_image_path).stem}.pdf")
        if not save_path:
            return
        try:
            self.update_status("Generating PDF report…", "busy")
            lineage_stamp = ClinicalLineageTracker.generate_provenance_stamp(
                model_path="model/best_model.pth", dicom_metadata=self.image_metadata)
            patient_info = {
                'id': self.image_metadata.get('patient_id', Path(self.current_image_path).stem),
                'name': 'Incidental Finding',
                'scan_date': self.image_metadata.get('study_date', datetime.now().strftime("%Y-%m-%d")),
                'date': datetime.now().strftime("%Y-%m-%d"),
                'lineage': lineage_stamp,
            }
            ok = generate_pdf_report(save_path=save_path, analysis_result=self.analysis_result,
                                     original_image=self.current_image, patient_info=patient_info)
            if ok and Path(save_path).exists() and Path(save_path).stat().st_size > 0:
                messagebox.showinfo("Report exported", f"PDF saved:\n{save_path}")
                self.update_status("PDF report exported.", "success")
            else:
                raise RuntimeError("The report adapter did not create a valid PDF.")
        except Exception as exc:
            logger.error("PDF generation failure: %s", exc, exc_info=True)
            messagebox.showerror("Report error", f"Failed to generate the PDF:\n{exc}")
            self.update_status("Report export failed.", "error")

    def generate_dicom_sr(self):
        if self.analysis_result is None:
            messagebox.showwarning("Nothing to export", "Run the analysis before exporting a DICOM SR.")
            return
        save_path = filedialog.asksaveasfilename(
            title="Export DICOM Structured Report", defaultextension=".dcm",
            filetypes=[("DICOM Structured Report", "*.dcm")], initialfile=f"SR_{Path(self.current_image_path).stem}.dcm")
        if not save_path:
            return
        try:
            self.update_status("Writing DICOM SR…", "busy")
            out = create_dicom_sr_report(save_path=save_path, analysis_result=self.analysis_result,
                                         dicom_metadata=self.image_metadata)
            if Path(out).exists() and Path(out).stat().st_size > 0:
                messagebox.showinfo("DICOM SR exported", f"DICOM Basic Text SR saved:\n{out}")
                self.update_status("DICOM SR exported.", "success")
            else:
                raise RuntimeError("The DICOM SR file was not written.")
        except Exception as exc:
            logger.error("DICOM SR export error: %s", exc, exc_info=True)
            messagebox.showerror("Export error", f"Failed to write the DICOM SR:\n{exc}")
            self.update_status("DICOM SR export failed.", "error")

    # ═════════════════════════════════════════════════════════ results panel
    def _card(self, title: str, accent: Optional[str] = None) -> Card:
        c = Card(self.results.inner, title, accent=accent)
        c.pack(fill=tk.X, pady=(0, 10))
        return c

    def _render_empty_results(self, loaded: bool = False):
        self.results.clear()
        c = self._card("Getting started")
        steps = [("Open a CT slice", loaded), ("Click on the nodule to set a seed", self.seed is not None),
                 ("Add patient details (optional, for Brock)", bool(self._read_risk_inputs())), ("Run analysis", False)]
        for i, (txt, done) in enumerate(steps, 1):
            r = tk.Frame(c.body, bg=T.SURFACE)
            r.pack(fill=tk.X, pady=4)
            tk.Label(r, text="✓" if done else str(i), font=Fonts.small_b, width=2, bg=T.OK_SOFT if done else T.DIVIDER,
                     fg=T.OK if done else T.MUTED).pack(side=tk.LEFT)
            tk.Label(r, text=txt, font=Fonts.body, bg=T.SURFACE, fg=T.TEXT if not done else T.MUTED).pack(side=tk.LEFT, padx=10)
        tk.Label(c.body, text="Results appear here: size, risk estimate, guideline lookup, and model output.",
                 font=Fonts.small, bg=T.SURFACE, fg=T.MUTED, wraplength=RIGHT_W - 80, justify="left").pack(anchor="w", pady=(8, 0))

    def _render_analyzing(self):
        self.results.clear()
        c = self._card("Analysing")
        tk.Label(c.body, text="Segmenting the nodule, running the model and computing guideline lookups…", font=Fonts.body,
                 bg=T.SURFACE, fg=T.MUTED, wraplength=RIGHT_W - 80, justify="left").pack(anchor="w")

    def _display_results(self):
        self.results.clear()
        res = self.analysis_result
        if not res:
            return
        expert, features = res.get("expert_analysis", {}), res.get("features", {})
        rec, brock = res.get("recist"), res.get("brock")
        wrap = RIGHT_W - 84

        if res.get("is_demo"):
            tk.Label(self.results.inner, text="DEMO — simulated output, not a model result", font=Fonts.small_b,
                     bg=T.WARN_SOFT, fg=T.WARN, pady=6).pack(fill=tk.X, pady=(0, 10))

        # 1. measurement
        c = self._card("Nodule measurement", accent=T.ACCENT)
        if rec is not None:
            top = tk.Frame(c.body, bg=T.SURFACE)
            top.pack(fill=tk.X)
            tk.Label(top, text=f"{rec.longest_diameter_mm:.1f}", font=Fonts.num_xl, bg=T.SURFACE, fg=T.TEXT).pack(side=tk.LEFT)
            tk.Label(top, text=" mm", font=Fonts.title, bg=T.SURFACE, fg=T.MUTED).pack(side=tk.LEFT, anchor="s", pady=(0, 6))
            Chip(top, rec.attenuation_type.replace("_", " "), "info").pack(side=tk.RIGHT, anchor="n")
            tk.Label(c.body, text=f"longest axis  ·  {rec.perpendicular_diameter_mm:.1f} mm perpendicular", font=Fonts.small,
                     bg=T.SURFACE, fg=T.MUTED).pack(anchor="w", pady=(0, 10))
            kv_rows(c.body, [("Mean diameter", f"{rec.mean_recist_diameter_mm:.1f} mm"),
                             ("Equivalent diameter", f"{rec.equivalent_diameter_mm:.1f} mm"),
                             ("Estimated volume", f"{rec.volume_mm3_estimated / 1000:.2f} mL"),
                             ("Solid fraction (CTR)", f"{rec.consolidation_to_tumor_ratio:.2f}")])
            for w in rec.warnings:
                tk.Label(c.body, text=f"⚠  {w}", font=Fonts.small, bg=T.WARN_SOFT, fg=T.WARN, wraplength=wrap,
                         justify="left", padx=8, pady=5).pack(fill=tk.X, pady=(8, 0))
        else:
            why = ("Pixel spacing is missing — sizes need a DICOM with calibrated spacing." if self.image_metadata.get("pixel_spacing") is None
                   else "No nodule was found. Click directly on the nodule to set a seed, then run the analysis again.")
            tk.Label(c.body, text="No measurement", font=Fonts.h2, bg=T.SURFACE, fg=T.TEXT).pack(anchor="w")
            tk.Label(c.body, text=why, font=Fonts.small, bg=T.SURFACE, fg=T.MUTED, wraplength=wrap, justify="left").pack(anchor="w", pady=(4, 0))

        # 2. Brock
        c = self._card("Risk estimate · Brock (PanCan)")
        if brock is not None:
            top = tk.Frame(c.body, bg=T.SURFACE)
            top.pack(fill=tk.X)
            tk.Label(top, text=f"{brock.risk_percentage:.1f}%", font=Fonts.num_l, bg=T.SURFACE, fg=T.TEXT).pack(side=tk.LEFT)
            Chip(top, brock.risk_tier.replace("_", " "), TIER_KIND.get(brock.risk_tier, "neutral")).pack(side=tk.RIGHT, anchor="n")
            tk.Label(c.body, text="estimated probability of malignancy", font=Fonts.small, bg=T.SURFACE, fg=T.MUTED).pack(anchor="w")
            BarGauge(c.body, brock.risk_percentage, T.ACCENT, ticks=((5, "5"), (10, "10"), (65, "65"))).pack(fill=tk.X, pady=(8, 0))
            for w in getattr(brock, "warnings", []):
                tk.Label(c.body, text=f"⚠  {w}", font=Fonts.small, bg=T.WARN_SOFT, fg=T.WARN, wraplength=wrap, justify="left",
                         padx=8, pady=5).pack(fill=tk.X, pady=(8, 0))
            tk.Label(c.body, text="Published model for screening-detected nodules of about 1–30 mm; can be miscalibrated in other populations.",
                     font=Fonts.micro, bg=T.SURFACE, fg=T.MUTED, wraplength=wrap, justify="left").pack(anchor="w", pady=(8, 0))
        else:
            need = "a nodule measurement" if rec is None else "patient details"
            tk.Label(c.body, text="Not computed", font=Fonts.h2, bg=T.SURFACE, fg=T.TEXT).pack(anchor="w")
            tk.Label(c.body, text=f"Needs {need}. Age, sex and nodule location are entered on the left; nothing is assumed.",
                     font=Fonts.small, bg=T.SURFACE, fg=T.MUTED, wraplength=wrap, justify="left").pack(anchor="w", pady=(4, 6))
            if rec is not None:
                FlatButton(c.body, "Add patient details", lambda: self.ent_age.focus_set(), pady=5, padx=10).pack(anchor="w")

        # 3. guideline
        c = self._card("Follow-up lookup · Fleischner 2017")
        tk.Label(c.body, text=expert.get("follow_up_interval", "N/A"), font=Fonts.body_b, bg=T.SURFACE, fg=T.TEXT,
                 wraplength=wrap, justify="left").pack(anchor="w")
        tk.Label(c.body, text=expert.get("recommendation", "N/A"), font=Fonts.small, bg=T.SURFACE, fg=T.MUTED,
                 wraplength=wrap, justify="left").pack(anchor="w", pady=(4, 6))
        tk.Label(c.body, text="Applies to incidental nodules in adults ≥35 without known cancer or immunosuppression.",
                 font=Fonts.micro, bg=T.SURFACE, fg=T.FAINT, wraplength=wrap, justify="left").pack(anchor="w")

        # 4. model output (de-emphasised, neutral colours, with provenance)
        c = self._card("Model output · research")
        p_mal = res["confidence"] if res["pred_int"] == 1 else 1.0 - res["confidence"]
        top = tk.Frame(c.body, bg=T.SURFACE)
        top.pack(fill=tk.X)
        tk.Label(top, text=f"{p_mal:.0%}", font=Fonts.num_l, bg=T.SURFACE, fg=T.TEXT).pack(side=tk.LEFT)
        tk.Label(top, text="  malignant-class score", font=Fonts.small, bg=T.SURFACE, fg=T.MUTED).pack(side=tk.LEFT, anchor="s", pady=(0, 4))
        BarGauge(c.body, p_mal * 100, T.FAINT, ticks=((50, "50"),)).pack(fill=tk.X, pady=(6, 0))
        text, kind, tip = self._model_status()
        Chip(c.body, text, kind).pack(anchor="w", pady=(6, 0))
        tk.Label(c.body, text=(tip + ". " if tip else "") + (f"Calibrated (T = {res['temperature']:.2f})." if res.get("calibrated")
                                                              else "Uncalibrated probabilities."),
                 font=Fonts.micro, bg=T.SURFACE, fg=T.MUTED, wraplength=wrap, justify="left").pack(anchor="w", pady=(6, 0))

        # 5. heuristic shape features (collapsed)
        c = self._card(None)
        col = Collapsible(c.body, "Shape features (heuristic, unvalidated)")
        col.pack(fill=tk.X)
        kv_rows(col.body, [("Sphericity", f"{features.get('sphericity', 0):.2f}"),
                           ("Margin", str(features.get("margin", "N/A")).capitalize()),
                           ("Texture", str(features.get("texture", "N/A")).capitalize()),
                           ("Spiculation (heuristic)", "Present" if features.get("spiculation") else "Absent"),
                           ("Heuristic risk index", f"{expert.get('risk_score', 0):.2f}"),
                           ("Computed from", "nodule mask" if features.get("source") == "nodule_mask" else "whole image")])


MedicalLungAnalyzerGUI = NoduleWorkstation   # backwards-compatible name


def main():
    if sys.platform == "win32":      # crisp text on high-DPI Windows displays
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    NoduleWorkstation(root)
    root.mainloop()


if __name__ == "__main__":
    main()
