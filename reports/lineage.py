"""
Tier 5: Clinical Provenance, Lineage Stamping, and Audit Logging
Provenance stamping to support traceability. This is NOT a validated 21 CFR Part 11 audit system.
"""

import hashlib
import hmac
import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional


class ClinicalLineageTracker:
    """Computes and seals immutable provenance signatures for reports and APIs."""

    @staticmethod
    def get_git_commit_sha() -> str:
        """Git commit of the code: GIT_COMMIT_SHA env (set at image build) else `git rev-parse`."""
        env_sha = os.getenv("GIT_COMMIT_SHA", "").strip()
        if env_sha:
            return env_sha
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("ascii").strip()
            return sha
        except Exception:
            return "DEV_TREE_NO_COMMIT"

    @staticmethod
    def compute_file_sha256(filepath: str) -> str:
        """Compute SHA-256 checksum of model weights or artifacts."""
        p = Path(filepath)
        if not p.exists():
            return "UNAVAILABLE_WEIGHTS_MISSING"
        sha256 = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    @classmethod
    def generate_provenance_stamp(
        cls,
        model_path: str = "model/best_model.pth",
        dicom_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Produce auditable provenance stamp."""
        meta = dicom_metadata or {}
        model_sha = cls.compute_file_sha256(model_path)
        git_sha = cls.get_git_commit_sha()
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Unique reproducible audit signature
        audit_raw = f"{model_sha}_{git_sha}_{meta.get('patient_id', 'ANON')}_{now_iso}"
        key = os.getenv("AUDIT_HMAC_KEY", "")
        if key:
            audit_sig = hmac.new(key.encode("utf-8"), audit_raw.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
            sig_type = "hmac-sha256"
        else:
            # Unkeyed: anyone can recompute it, so it detects accidents, not tampering.
            audit_sig = hashlib.sha256(audit_raw.encode("utf-8")).hexdigest()[:16]
            sig_type = "sha256-unkeyed (set AUDIT_HMAC_KEY for a keyed signature)"

        return {
            "model_sha256": model_sha,
            "pipeline_git_sha": git_sha,
            "sop_instance_uid": str(meta.get("sop_instance_uid", "NOT_SPECIFIED")),
            "series_instance_uid": str(meta.get("series_instance_uid", "NOT_SPECIFIED")),
            "study_instance_uid": str(meta.get("study_instance_uid", "NOT_SPECIFIED")),
            "radiological_timestamp": now_iso,
            "audit_hash": audit_sig,
            "audit_hash_type": sig_type
        }

