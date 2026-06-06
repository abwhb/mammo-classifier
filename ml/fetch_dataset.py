"""Download the CBIS-DDSM PNG mirror from Hugging Face.

Source: dbaek111/CBIS-DDSM_1024 — 1,024px 8-bit PNGs converted from the canonical
CBIS-DDSM DICOMs, organised as cancer/{train,test} and not_cancer/{train,test}.
~668 MB total. No auth required.

Usage:
    uv run python fetch_dataset.py [--dest data/cbis-ddsm-png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "dbaek111/CBIS-DDSM_1024"
REPO_TYPE = "dataset"
DEFAULT_DEST = Path(__file__).parent / "data" / "cbis-ddsm-png"


def main(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} → {dest}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(dest),
        allow_patterns=["*.zip", "*.md"],
    )
    print(f"Downloaded. Now unpack the .zip files in {dest} to use them.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    sys.exit(main(args.dest))
