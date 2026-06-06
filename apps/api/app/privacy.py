"""DICOM de-identification.

Implements a subset of the DICOM Basic Application Confidentiality Profile
(DICOM PS 3.15 Annex E, Basic Profile) sufficient for this assessment.
PNG/JPEG carry no DICOM PHI tags, so they are returned unchanged.
"""

from __future__ import annotations

import io
import uuid

import pydicom
from pydicom.dataset import Dataset

# Tags to remove or blank. Names mirror the DICOM standard.
_PHI_TAGS_TO_REMOVE: tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "EthnicGroup",
    "Occupation",
    "AccessionNumber",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName",
    "OperatorsName",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "StationName",
    "StudyDate",
    "StudyTime",
    "SeriesDate",
    "SeriesTime",
    "AcquisitionDate",
    "AcquisitionTime",
    "ContentDate",
    "ContentTime",
    "StudyDescription",
    "SeriesDescription",
    "RequestingPhysician",
    "OtherPatientIDs",
    "OtherPatientNames",
    "MedicalRecordLocator",
    "PatientMotherBirthName",
    "DeviceSerialNumber",
)

# UIDs need to be replaced (not just removed) to keep the DICOM file valid.
_UID_TAGS: tuple[str, ...] = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
)


def _new_uid() -> str:
    # 2.25.<uint128> is the de facto pattern for UUID-derived DICOM UIDs.
    return f"2.25.{uuid.uuid4().int}"


def strip_identifiers(data: bytes) -> bytes:
    """Return DICOM bytes with PHI removed. Non-DICOM input is returned as-is."""
    if len(data) < 132 or data[128:132] != b"DICM":
        return data
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    _scrub(ds)
    out = io.BytesIO()
    ds.save_as(out, enforce_file_format=True)
    return out.getvalue()


def _scrub(ds: Dataset) -> None:
    for tag in _PHI_TAGS_TO_REMOVE:
        if tag in ds:
            del ds[tag]
    for tag in _UID_TAGS:
        if tag in ds:
            ds[tag].value = _new_uid()
    # Remove every private tag block (vendor-specific PHI often hides here).
    ds.remove_private_tags()
    # Recurse into sequences (e.g., ReferencedStudySequence may carry PHI).
    for elem in ds.iterall():
        if elem.VR == "SQ":
            for item in elem.value or []:
                _scrub(item)
