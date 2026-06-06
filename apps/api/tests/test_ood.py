"""Verify the OOD guard accepts real mammograms and rejects obvious non-mammograms."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.ood import is_plausible_mammogram

SAMPLES_ROOT = Path(__file__).resolve().parents[3] / "ml" / "data" / "cbis-ddsm-png" / "test"


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_real_mammograms_pass_when_available() -> None:
    """Sanity-check the heuristic on real test-set mammograms.

    Skipped automatically when the dataset isn't on disk (CI runs without it).
    """
    if not SAMPLES_ROOT.exists():
        pytest.skip("CBIS-DDSM dataset not on disk")
    paths = sorted((SAMPLES_ROOT / "cancer").glob("*.png"))[:3]
    paths += sorted((SAMPLES_ROOT / "not_cancer").glob("*.png"))[:3]
    assert paths, "no test PNGs found"
    for p in paths:
        check = is_plausible_mammogram(p.read_bytes())
        assert check.is_plausible, f"{p.name} rejected: {check.reason}"


def test_bright_photo_is_rejected() -> None:
    img = Image.new("RGB", (640, 480), (180, 130, 100))
    check = is_plausible_mammogram(_png_bytes(img))
    assert not check.is_plausible
    assert "dark" in check.reason.lower() or "background" in check.reason.lower()


def test_dark_theme_screenshot_is_rejected() -> None:
    img = Image.new("RGB", (1024, 768), (30, 30, 35))
    draw = ImageDraw.Draw(img)
    for y in range(50, 700, 60):
        for x in range(50, 950, 200):
            draw.rectangle([x, y, x + 150, y + 25], fill=(200, 200, 210))
    check = is_plausible_mammogram(_png_bytes(img))
    assert not check.is_plausible


def test_dicom_magic_is_admitted_without_inspection() -> None:
    payload = b"\x00" * 128 + b"DICM" + b"\x00" * 100
    check = is_plausible_mammogram(payload)
    assert check.is_plausible
    assert check.reason == "dicom"
