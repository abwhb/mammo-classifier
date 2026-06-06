"""Verify DICOM PHI stripping leaves no identifiers behind."""

from __future__ import annotations

import io
from datetime import date, time

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from app.privacy import _PHI_TAGS_TO_REMOVE, _UID_TAGS, strip_identifiers


def _build_phi_laden_dicom() -> bytes:
    """Construct a minimal DICOM with every tag we promise to strip set to a known value."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # PHI fields — every one of these should be stripped.
    ds.PatientName = "Jane^Patient"
    ds.PatientID = "PHI-12345"
    ds.PatientBirthDate = date(1970, 1, 1).strftime("%Y%m%d")
    ds.PatientSex = "F"
    ds.AccessionNumber = "ACC-001"
    ds.ReferringPhysicianName = "Dr^Referrer"
    ds.PerformingPhysicianName = "Dr^Performer"
    ds.OperatorsName = "Tech^Operator"
    ds.InstitutionName = "PHI Hospital"
    ds.InstitutionAddress = "123 PHI St"
    ds.InstitutionalDepartmentName = "Radiology"
    ds.StationName = "MAMMO-01"
    ds.StudyDate = "20240101"
    ds.StudyTime = time(12, 0).strftime("%H%M%S")
    ds.SeriesDate = "20240101"
    ds.AcquisitionDate = "20240101"
    ds.ContentDate = "20240101"
    ds.StudyDescription = "Bilateral mammogram"
    ds.SeriesDescription = "MLO views"
    ds.OtherPatientIDs = "PHI-ALT"
    ds.DeviceSerialNumber = "DEV-001"

    # UIDs that should be replaced (not deleted) to keep the file valid.
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    # Trivial pixel data so this is a structurally-valid DICOM.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 2
    ds.Columns = 2
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = b"\x00\x10\x20\x30"

    # Add a private tag block to confirm we strip them.
    private_block = ds.private_block(0x000B, "FAHM TEST", create=True)
    private_block.add_new(0x01, "LO", "vendor-secret-PHI")

    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def test_strip_identifiers_removes_every_phi_tag() -> None:
    src = _build_phi_laden_dicom()
    cleaned = strip_identifiers(src)

    ds = pydicom.dcmread(io.BytesIO(cleaned), force=True)

    for tag in _PHI_TAGS_TO_REMOVE:
        assert tag not in ds, f"PHI tag {tag!r} survived de-identification"

    # UIDs should still be present (the file would be invalid without them)
    # but rotated to new values.
    src_ds = pydicom.dcmread(io.BytesIO(src), force=True)
    for tag in _UID_TAGS:
        assert tag in ds, f"UID {tag} should be replaced, not deleted"
        assert getattr(ds, tag) != getattr(src_ds, tag), f"UID {tag} was not rotated"

    # No private tags should remain.
    private_tags = [el for el in ds.iterall() if el.tag.is_private]
    assert not private_tags, f"Private tags survived: {private_tags}"


def test_strip_identifiers_passes_through_non_dicom_unchanged() -> None:
    png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    assert strip_identifiers(png_magic) == png_magic
