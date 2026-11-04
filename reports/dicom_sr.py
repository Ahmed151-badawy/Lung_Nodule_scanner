"""
DICOM Basic Text Structured Report (SR) writer.

SOP Class: 1.2.840.10008.5.1.4.1.1.88.11 (Basic Text SR Storage).

SCOPE / HONESTY NOTE
  This is a *Basic Text* SR: a CONTAINER root holding free-text items. It is NOT a
  TID 1500 Measurement Report (that needs a Comprehensive/Enhanced SR with coded,
  numeric measurements and references to the source images) and has NOT been
  conformance-tested against any PACS. Treat it as a research export. Validate it
  with a DICOM validator (e.g. dciodvfy) and your PACS vendor before any clinical use.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from reports.lineage import ClinicalLineageTracker

BASIC_TEXT_SR_UID = "1.2.840.10008.5.1.4.1.1.88.11"


def _code(value: str, scheme: str, meaning: str) -> Dataset:
    c = Dataset()
    c.CodeValue = value
    c.CodingSchemeDesignator = scheme
    c.CodeMeaning = meaning
    return c


def _text_item(label: str, text: str) -> Dataset:
    """TEXT content item using DCM 121106 'Comment'; the label is the first words of the text."""
    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = "TEXT"
    item.ConceptNameCodeSequence = Sequence([_code("121106", "DCM", "Comment")])
    item.TextValue = f"{label}: {text}"
    return item


def _fmt(v: Any, spec: str = "") -> str:
    if v is None:
        return "N/A"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def create_dicom_sr_report(
    save_path: str,
    analysis_result: Dict[str, Any],
    dicom_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Write a Basic Text SR (.dcm). Unknown patient demographics are left empty, never invented."""
    meta = dicom_metadata or {}
    features = analysis_result.get("features") or {}
    expert = analysis_result.get("expert_analysis") or {}
    recist = analysis_result.get("recist")
    brock = analysis_result.get("brock")

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = BASIC_TEXT_SR_UID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.9.7133.1.1"

    ds = FileDataset(save_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SpecificCharacterSet = "ISO_IR 192"

    # Patient (Type 2: present, possibly empty)
    ds.PatientName = str(meta.get("patient_name", ""))
    ds.PatientID = str(meta.get("patient_id", ""))
    ds.PatientBirthDate = str(meta.get("patient_birth_date", ""))
    ds.PatientSex = str(meta.get("patient_sex", ""))

    # Study / Series
    now = datetime.now()
    ds.StudyInstanceUID = str(meta.get("study_instance_uid") or generate_uid())
    ds.StudyDate = str(meta.get("study_date") or now.strftime("%Y%m%d"))
    ds.StudyTime = now.strftime("%H%M%S")
    ds.ReferringPhysicianName = ""
    ds.StudyID = ""
    ds.AccessionNumber = ""
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 901
    ds.Modality = "SR"
    ds.SeriesDescription = "AI nodule analysis (research prototype)"

    # Equipment / SR document
    ds.Manufacturer = "Research Prototype"
    ds.InstanceNumber = 1
    ds.SOPClassUID = BASIC_TEXT_SR_UID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.CompletionFlag = "COMPLETE"
    ds.VerificationFlag = "UNVERIFIED"
    ds.PerformedProcedureCodeSequence = Sequence([])
    ds.CurrentRequestedProcedureEvidenceSequence = Sequence([])

    # Root content item: CONTAINER (required for any SR document)
    ds.ValueType = "CONTAINER"
    ds.ConceptNameCodeSequence = Sequence([_code("18748-4", "LN", "Diagnostic Imaging Report")])
    ds.ContinuityOfContent = "SEPARATE"

    items = [
        _text_item("Status", "RESEARCH USE ONLY - NOT FOR DIAGNOSTIC USE. Algorithm output, unverified by a radiologist."),
        _text_item("Model output", f"{analysis_result.get('prediction', 'UNKNOWN')} "
                                   f"(model confidence {_fmt(analysis_result.get('confidence'), '.1%')})"),
        _text_item("Shape features", f"equivalent diameter {_fmt(features.get('size_mm'))} mm; "
                                     f"sphericity {_fmt(features.get('sphericity'), '.2f')}; "
                                     f"margin {features.get('margin', 'N/A')}; "
                                     f"spiculation heuristic {features.get('spiculation', 'N/A')}; "
                                     f"feature source {features.get('source', 'N/A')}"),
    ]
    if recist is not None:
        items.append(_text_item("Axial diameters", f"longest {recist.longest_diameter_mm} mm; "
                                                   f"perpendicular {recist.perpendicular_diameter_mm} mm; "
                                                   f"attenuation {recist.attenuation_type}"))
    if brock is not None:
        items.append(_text_item("Brock estimate", f"{brock.risk_percentage}% ({brock.risk_tier}); {brock.model_version}"))
    items.append(_text_item("Follow-up guidance", f"{expert.get('follow_up_interval', 'N/A')} - {expert.get('recommendation', 'N/A')}"))

    lineage = ClinicalLineageTracker.generate_provenance_stamp(dicom_metadata=meta)
    items.append(_text_item("Algorithm provenance",
                            f"model SHA-256 {lineage.get('model_sha256')}; git {lineage.get('pipeline_git_sha')}; "
                            f"audit {lineage.get('audit_hash')} ({lineage.get('audit_hash_type', 'n/a')})"))
    ds.ContentSequence = Sequence(items)

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_file), write_like_original=False)
    return str(out_file)
