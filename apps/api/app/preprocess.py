"""Image ingestion → normalized tensor.

Accepts DICOM, PNG, and JPEG bytes and produces a `[3, 224, 224]` float32 array
ready for EfficientNet-B0 (ImageNet pretrain stats).
"""

from __future__ import annotations

import io

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixels import apply_voi_lut

# ImageNet normalisation — required for torchvision pretrained models.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

TARGET_SIZE = 224


def is_dicom(data: bytes) -> bool:
    # DICOM Part 10 files have "DICM" at offset 128.
    return len(data) >= 132 and data[128:132] == b"DICM"


def _dicom_to_array(data: bytes) -> np.ndarray:
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = apply_voi_lut(ds.pixel_array, ds) if hasattr(ds, "pixel_array") else ds.pixel_array
    arr = arr.astype(np.float32)
    # MONOCHROME1 means dark = high intensity; invert for consistent display range.
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def _image_to_array(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.asarray(img, dtype=np.float32)


def _normalise(arr: np.ndarray) -> np.ndarray:
    # Percentile clip to reduce sensor outliers, then min-max to [0,1].
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    rng = max(hi - lo, 1e-6)
    return (arr - lo) / rng


def _resize(arr: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def to_tensor(data: bytes) -> np.ndarray:
    """Decode → normalise → resize → tile to 3 channels → ImageNet-normalise.

    Returns a `(3, 224, 224)` float32 array.
    """
    arr = _dicom_to_array(data) if is_dicom(data) else _image_to_array(data)
    arr = _normalise(arr)
    arr = _resize(arr)
    # Tile single grayscale channel to 3 to match EfficientNet-B0 input.
    arr = np.stack([arr, arr, arr], axis=0)
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.astype(np.float32)
