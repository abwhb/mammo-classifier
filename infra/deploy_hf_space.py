"""Stage the API + model into a temp dir and push to a Hugging Face Docker Space.

Usage:
    HF_TOKEN=hf_... uv run python infra/deploy_hf_space.py \
        --repo-id abwhb/mammo-classifier-api

The script is idempotent: re-running re-uploads any changed files. It does NOT
delete files from the Space that have been removed locally — clear the Space
manually if you need a clean slate.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
API_DIR = ROOT / "apps" / "api"
MODEL_PATH = ROOT / "models" / "mammo-v1.pt"

README_TEMPLATE = """\
---
title: Mammogram Classifier API
emoji: 🩺
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Research/demo mammogram classifier (not clinical)
---

# Mammogram Classifier API

FastAPI service that classifies a mammogram (DICOM, PNG, or JPEG) as
**Malignant** or **Benign** with a confidence score. Built for the
FAHM Biotechnology technical assessment.

**Research / demonstration only.** Not a medical device. Not approved or
intended for clinical use.

## Endpoints

- `GET /healthz` — liveness + model version
- `POST /predict` — multipart `file=` form upload, returns JSON

## Privacy

- DICOM PHI is stripped before any other processing
  (PS 3.15 Annex E Basic Profile, private tags removed, UIDs rotated).
- Uploaded images are processed in memory and never written to disk.
- Logs contain only request ID, file size, MIME type, latency, and prediction
  label — no PHI.

Source: <https://github.com/{github_repo}>
"""


def stage(staging: Path, repo_id: str) -> None:
    """Lay out the files exactly as the Space repo expects them."""
    shutil.copytree(API_DIR / "app", staging / "app")
    shutil.copy2(API_DIR / "Dockerfile", staging / "Dockerfile")
    shutil.copy2(API_DIR / ".dockerignore", staging / ".dockerignore")
    shutil.copy2(API_DIR / "pyproject.toml", staging / "pyproject.toml")
    shutil.copy2(API_DIR / "uv.lock", staging / "uv.lock")
    shutil.copy2(MODEL_PATH, staging / "model.pt")
    (staging / "README.md").write_text(
        README_TEMPLATE.format(github_repo="<set after pushing to GitHub>")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. abwhb/mammo-classifier-api")
    parser.add_argument(
        "--allowed-origins",
        default="*",
        help='CORS allowed origins (comma-separated). Default "*" for the demo.',
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN env var with a write-scope HF token", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    print(f"→ Ensuring Space {args.repo_id} exists (Docker SDK)…")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=False,
    )

    # Set CORS via Space secret-style variable (visible to the container as env var).
    print(f"→ Setting ALLOWED_ORIGINS={args.allowed_origins}")
    api.add_space_variable(
        repo_id=args.repo_id,
        key="ALLOWED_ORIGINS",
        value=args.allowed_origins,
    )

    with TemporaryDirectory(prefix="hf-space-") as tmp:
        staging = Path(tmp) / "stage"
        staging.mkdir()
        print(f"→ Staging files in {staging}")
        stage(staging, args.repo_id)
        for p in sorted(staging.rglob("*")):
            if p.is_file():
                print(f"   {p.relative_to(staging)}  ({p.stat().st_size // 1024} KB)")

        print(f"→ Uploading folder to {args.repo_id}…")
        commit_url = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=str(staging),
            commit_message="Deploy mammo-classifier-api",
        )
        print(f"✓ Uploaded. Commit: {commit_url}")

    space_url = f"https://huggingface.co/spaces/{args.repo_id}"
    direct_url = f"https://{args.repo_id.replace('/', '-')}.hf.space"
    print()
    print(f"Space page:    {space_url}")
    print(f"Direct origin: {direct_url}")
    print()
    print("Build takes ~5-10 min. Watch logs on the Space page.")
    print("When build finishes, /healthz should return model_loaded: true.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
