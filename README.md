# Pulmonary Nodule Diagnostic Intelligence (CADx) & Decision Support System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.15+-blueviolet.svg)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ed.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, clinically calibrated Computer-Aided Detection & Diagnosis (CADe/CADx) platform for thoracic Computed Tomography (CT) scans. Combines deep learning with evidence-based clinical guidelines (**Fleischner Society 2017** and **Brock University / PanCan Model**) and medical imaging standards (**DICOM Part 10 Structured Reporting**).

---

## 📋 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Clinical & Scientific Foundation](#-clinical--scientific-foundation)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Usage & Running](#-usage--running)
  - [1. PACS Desktop Application](#1-pacs-desktop-application)
  - [2. FastAPI REST Microservice](#2-fastapi-rest-microservice)
  - [3. Docker Container](#3-docker-container)
  - [4. End-to-End Pipeline Verification](#4-end-to-end-pipeline-verification)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Regulatory & Clinical Disclaimer](#-regulatory--clinical-disclaimer)

---

## 🚀 Key Features

* **Calibrated CT Radiodensity Ingestion:** Native 16-bit DICOM ingestion with automatic rescale slope and intercept mapping to Hounsfield Units (HU) and standard pulmonary parenchymal windowing ($L = -600, W = 1500 \implies [-1350, +150]\text{ HU}$).
* **OOD Input Quality Gate:** Real-time out-of-distribution (OOD) detector evaluating CT attenuation distributions to block invalid modalities (e.g., MRI, Ultrasound) and unreadable empty air scans ($<-900\text{ HU}$).
* **Compact, Resolution-Agnostic CNN:** Custom 4-stage convolutional backbone with Global Average Pooling (GAP) reducing parameters by **98.8%** (~388K params vs ~33.5M in dense architectures) with sub-50ms CPU inference via ONNX INT8 quantization.
* **Post-Hoc Probability Calibration:** Temperature Scaling fitted on validation set logits to minimize Expected Calibration Error (ECE) and eliminate neural overconfidence.
* **Leak-Free Explainability (XAI):** Context-managed gradient hooks for **Grad-CAM**, **Grad-CAM++**, **Score-CAM**, and **Integrated Gradients** ensuring zero PyTorch memory leaks during continuous PACS reading.
* **RECIST 1.1 Morphometry & CTR:** Semi-automated contour segmentation measuring longest diameter ($d_{\text{long}}$), perpendicular short axis ($d_{\text{perp}}$), and Consolidation-to-Tumor Ratio (CTR) to classify solid, part-solid, and pure ground-glass nodules (GGN).
* **Clinical Decision Support (CDSS):**
  * **Fleischner Society 2017 Guidelines:** Automated stratification based on nodule attenuation, size cutoffs (<6 mm, 6–8 mm, >8 mm), multiplicity, and patient risk history.
  * **Brock University (PanCan) Model:** Full multivariable logistic regression accounting for age, sex, cancer family history, emphysema, spiculation, and upper-lobe localization.
* **Audit Lineage & Interoperable Reporting:**
  * Cryptographic provenance stamping embedding Model SHA-256, Git Commit SHA, and DICOM Series Instance UID.
  * Native binary **DICOM Part 10 Structured Reporting (TID 1500 SR)** generation.
  * Publication-ready multi-page clinical PDF report generation via ReportLab.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A["Raw Thoracic CT Scan (DICOM)"] --> B["HU Conversion & Pulmonary Windowing\n(L: -600 HU, W: 1500 HU)"]
    B --> C{"Input Quality Gate & OOD Detector"}
    C -- "Invalid / Out of Bounds" --> C_Reject["Reject Slice / Flag Warning"]
    C -- "Valid Parenchyma" --> D["2.5D Axial Multi-Slice Stacking\n(z-1, z, z+1)"]
    
    D --> E["NoduleCNN Classifier\n(GAP Head, ~388K params)"]
    E --> F["Logits"]
    F --> G["Temperature Scaling Calibrator\n(Minimizes ECE)"]
    G --> H["Calibrated Malignancy Probability"]

    D --> I["Explainability Engine\n(Grad-CAM / Grad-CAM++ / Score-CAM)"]
    I --> J["Saliency Heatmap Overlay"]

    B --> K["Anatomical Nodule Segmenter"]
    K --> L["RECIST 1.1 Sizing (d_long x d_perp)\n& Consolidation Ratio (CTR)"]

    H & L & M["Patient Risk Profile"] --> N["Clinical Rules Engine"]
    N --> O["Fleischner Society 2017 Guidance"]
    N --> P["Brock (PanCan) Risk Calculation"]

    H & J & L & O & P --> Q["Clinical Lineage Stamping\n(SHA-256 + Git SHA + DICOM UID)"]
    Q --> R1["PACS Desktop Interface (Tkinter)"]
    Q --> R2["FastAPI REST Microservice"]
    Q --> R3["DICOM Part 10 SR (TID 1500)"]
    Q --> R4["Clinical PDF Diagnostic Report"]
```

---

## 🔬 Clinical & Scientific Foundation

1. **Fleischner Society 2017 Guidelines:** *Bankier AA, et al. Recommendations for Measuring Pulmonary Nodules at CT: A Statement from the Fleischner Society. Radiology 2017; 288(3):899-906.*
2. **Brock University / PanCan Risk Model:** *McWilliams A, et al. Probability of Cancer in Pulmonary Nodules Detected on First Screening CT. N Engl J Med 2013; 369:910-919.*
3. **RECIST 1.1 Criteria:** *Eisenhauer EA, et al. New response evaluation criteria in solid tumours: Revised RECIST guideline (version 1.1). Eur J Cancer 2009; 45(2):228-247.*
4. **LIDC-IDRI Cohort:** *Armato SG 3rd, et al. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI). Med Phys 2011; 38(2):915-931.*

---

## 📂 Project Directory Structure

```text
├── assets/                     # Application icons and GUI resources
├── data/                       # LIDC dataset ingestion and crop loaders
├── dist/                       # Output artifacts, ONNX models, and reports
│   └── onnx/                   # Exported FP32 and INT8 ONNX models
├── expert_system/              # Clinical guideline rules & risk calculators
│   ├── brock_model.py          # Brock University (PanCan) logistic model
│   └── rules.py                # Fleischner Society 2017 decision tree
├── explainability/             # Explainable AI & Uncertainty Calibration
│   ├── calibration_and_saliency.py # Temperature scaling, ECE, Score-CAM
│   └── gradcam.py              # Context-managed Grad-CAM & Grad-CAM++
├── model/                      # PyTorch neural network definitions & weights
│   ├── model.py                # NoduleCNN architecture & inference pipeline
│   ├── best_model.pth          # Calibrated checkpoint
│   └── calibrator.pth          # Learned temperature parameter
├── preprocessing/              # Medical image transformation & segmentation
│   ├── nodule_segmenter.py     # RECIST 1.1 morphometry & CTR calculation
│   └── volume_ood.py           # 2.5D context stacker & HU OOD quality gate
├── reports/                    # Clinical documentation and PACS outputs
│   ├── dicom_sr.py             # DICOM Part 10 Structured Reporting (SR)
│   ├── lineage.py              # Provenance & audit hashing
│   └── pdf_adapter.py          # Multipage ReportLab clinical PDF generator
├── scripts/                    # Verification, ONNX export, and training
│   ├── export_onnx.py          # PyTorch to ONNX quantization exporter
│   ├── train_lidc.py           # Patient-disjoint training pipeline
│   └── verify_e2e.py           # End-to-end integration smoke test
├── service/                    # Production REST Microservice
│   └── app.py                  # FastAPI server with /health, /predict, /explain
├── tests/                      # Automated unit and integration test suite
├── utils/                      # DICOM I/O, windowing, and metric utilities
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Container orchestration configuration
├── gui.py                      # Multi-threaded desktop PACS viewer
├── requirements.txt            # Python dependencies
└── start.bat                   # Interactive Windows launcher
```

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Recommended: Virtual Environment (`venv` or `conda`)

```bash
# Clone the repository
git clone https://github.com/Ahmed151-badawy/Lung_Nodule_scanner.git
cd Lung_Nodule_scanner

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Usage & Running

### 1. PACS Desktop Application
Launch the PACS desktop interface for interactive nodule inspection, zoom/pan navigation, seed-point segmentation, and one-click PDF/DICOM SR generation:

```bash
python gui.py
```
*(Or double-click `start.bat` on Windows and select Option `1`)*

### 2. FastAPI REST Microservice
Run the high-performance async microservice:

```bash
uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at:
* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

#### Example Endpoint Usage:
* `GET /health` - Checks model readiness, device, and service status.
* `POST /predict` - Uploads a CT slice (DICOM or image), returns calibrated probability, RECIST measurements, and Brock risk tier.
* `POST /explain` - Returns base64-encoded Grad-CAM saliency heatmaps with localization coordinates.

### 3. Docker Container
Build and start the containerized service:

```bash
# Build and run with Docker Compose
docker-compose up --build -d

# Check health probe
curl -f http://localhost:8000/health
```

### 4. End-to-End Pipeline Verification
Execute the full verification harness (synthesizes calibrated CT slice, tests quality gate, generates Grad-CAM, computes RECIST 1.1 & Brock risk, and produces PDF/DICOM SR reports in `dist/`):

```bash
python scripts/verify_e2e.py
```

---

## 🧪 Testing & Quality Assurance

The codebase includes an automated test suite comprising 56 unit and integration tests covering:
* Model parameter limits (<1M params) and spatial-resolution invariance.
* Memory-leak-free context lifecycle of Grad-CAM hooks.
* Temperature scaling numerical correctness and ECE calculations.
* DICOM slope/intercept conversion and HU window boundary clipping.
* Out-of-distribution quality gating (empty air scans & non-CT modality rejection).
* RECIST 1.1 dual-axis accuracy and Consolidation-to-Tumor Ratio (CTR).
* Brock University (PanCan) risk calculator boundaries.
* DICOM Part 10 SR tag generation and ReportLab PDF creation.

Run all tests via `unittest`:

```bash
python -m unittest discover tests
```

---

## ⚖️ Regulatory & Clinical Disclaimer

> [!CAUTION]
> **RESEARCH & INVESTIGATIONAL PROTOTYPE ONLY**  
> This software system is developed strictly for research, benchmarking, and educational purposes. It is **NOT** a certified Medical Device (SaMD) and has not received clearance or approval from the U.S. Food and Drug Administration (FDA), CE marking under the European Medical Device Regulation (EU MDR), or equivalent health authorities. It must **NOT** be used as a primary diagnostic tool or to guide direct patient management without qualified radiologist over-read. All clinical decisions remain the sole legal responsibility of the licensed treating medical team.
