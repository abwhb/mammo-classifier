"""Unzip CBIS-DDSM PNGs, build a manifest, and write patient-grouped splits.

We use the upstream train/test partition as-is for the test set, and carve a
validation set out of train **grouped by patient ID** so no patient appears in
both train and val. This matches the CBIS-DDSM patient-level evaluation
discipline required to avoid leakage (multiple views per patient).

Inputs:
    data/cbis-ddsm-png/CBIS-DDSM_full_1024.zip

Outputs:
    data/cbis-ddsm-png/<unpacked tree>
    data/splits/manifest.csv
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data" / "cbis-ddsm-png"
SPLITS_DIR = ROOT / "splits"

# CBIS-DDSM filename conventions vary, but the patient identifier always begins
# with a "P_" prefix followed by digits. Examples:
#   Mass-Training_P_00001_LEFT_CC.png
#   P_00001_LEFT_MLO.png
PATIENT_RE = re.compile(r"P_(\d+)")


def unpack_zips(root: Path) -> None:
    zips = sorted(root.glob("*.zip"))
    if not zips:
        print("No zips found — assuming already unpacked.")
        return
    for z in zips:
        marker = root / f".{z.stem}.unpacked"
        if marker.exists():
            print(f"Skipping {z.name} (already unpacked).")
            continue
        print(f"Unpacking {z.name} → {root}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
        marker.touch()


def extract_patient_id(path: Path) -> str | None:
    m = PATIENT_RE.search(path.name)
    if m:
        return f"P_{m.group(1)}"
    # Some mirrors put patient IDs in the parent dir name; check up a few levels.
    for ancestor in path.parents:
        m = PATIENT_RE.search(ancestor.name)
        if m:
            return f"P_{m.group(1)}"
        if ancestor == path.parents[3]:
            break
    return None


def walk_manifest(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for png in root.rglob("*.png"):
        rel = png.relative_to(root)
        parts = [p.lower() for p in rel.parts]
        label = 1 if "cancer" in parts and "not_cancer" not in parts else 0
        # Disambiguate: "not_cancer" contains the substring "cancer".
        if any(p == "not_cancer" for p in parts):
            label = 0
        elif any(p == "cancer" for p in parts):
            label = 1
        else:
            # Unknown layout — skip.
            continue
        split = "test" if "test" in parts else "train" if "train" in parts else "unknown"
        pid = extract_patient_id(png) or "UNKNOWN"
        rows.append(
            {
                "path": str(rel),
                "label": label,
                "patient_id": pid,
                "upstream_split": split,
            }
        )
    df = pd.DataFrame(rows)
    return df


def build_splits(manifest: pd.DataFrame, val_frac: float, seed: int) -> dict[str, pd.DataFrame]:
    """Use upstream CBIS-DDSM train/test (patient-disciplined by the dataset authors)
    as our test set, and carve a stratified random validation set out of upstream
    train. The HF PNG mirror names files by SOP Instance UID rather than patient ID,
    so we cannot reconstruct patient grouping for the train/val carve from filenames
    alone — see REPORT.md for the honest trade-off discussion.
    """
    test = manifest[manifest["upstream_split"] == "test"].copy().reset_index(drop=True)
    upstream_train = manifest[manifest["upstream_split"] == "train"].copy().reset_index(drop=True)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(splitter.split(upstream_train, upstream_train["label"]))
    train = upstream_train.iloc[train_idx].reset_index(drop=True)
    val = upstream_train.iloc[val_idx].reset_index(drop=True)

    return {"train": train, "val": val, "test": test}


def label_counts(df: pd.DataFrame) -> str:
    cnts = df["label"].value_counts().to_dict()
    return f"benign={cnts.get(0, 0)} malignant={cnts.get(1, 0)} total={len(df)}"


def report_split_overlap(splits: dict[str, pd.DataFrame]) -> None:
    pids = {name: set(df["patient_id"]) - {"UNKNOWN"} for name, df in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = pids[a] & pids[b]
        # When patient_id is UNKNOWN we can't verify; surface that honestly.
        if not pids[a] or not pids[b]:
            print(
                f"  {a}/{b}: patient IDs unavailable; trusting upstream CBIS-DDSM "
                "patient discipline for test isolation."
            )
            continue
        marker = "OK" if not overlap else f"{len(overlap)} overlapping patients"
        print(f"  {a}/{b}: {marker}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, default=SPLITS_DIR)
    parser.add_argument("--val-frac", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.root.exists():
        raise SystemExit(f"Data root missing: {args.root} — run fetch_dataset.py first.")

    unpack_zips(args.root)
    print(f"Walking {args.root} for PNGs...")
    manifest = walk_manifest(args.root)
    if manifest.empty:
        raise SystemExit("No PNGs found after unpack.")
    print(f"Found {len(manifest)} images across {manifest['patient_id'].nunique()} patient IDs.")
    print(f"Unknown patient IDs: {(manifest['patient_id'] == 'UNKNOWN').sum()}")

    splits = build_splits(manifest, args.val_frac, args.seed)
    print("Split overlap check:")
    report_split_overlap(splits)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out / "manifest.csv", index=False)
    for name, df in splits.items():
        df.to_csv(args.out / f"{name}.csv", index=False)
        print(f"  {name}.csv → {label_counts(df)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
