# AGENT.md

Cross-tool project guide for AI coding agents (Claude Code, Cursor, Copilot, Aider, etc.).
For Claude-specific behavior, see `CLAUDE.md`. For architecture and rationale, see `DESIGN.md`.

---

## 1. What this project is

A full-stack AI application that classifies a mammogram image as **Malignant** or **Benign/Normal** with a confidence score. Built for the FAHM Biotechnology technical assessment.

- **Inputs:** DICOM (`.dcm`), PNG, or JPEG
- **Output:** `{ label: "malignant" | "benign", confidence: 0–1, model_version: string }`
- **Users:** clinicians (proxy task; not a real clinical product)

## 2. Repository layout

```
.
├── AGENT.md                # this file
├── CLAUDE.md               # Claude Code-specific instructions
├── DESIGN.md               # architecture, ML, privacy, deployment
├── README.md               # build / run / deploy instructions (for human reviewers)
├── REPORT.md               # source for the technical PDF report
├── apps/
│   ├── api/                # FastAPI backend (Python 3.11)
│   │   ├── app/
│   │   │   ├── main.py             # FastAPI entrypoint
│   │   │   ├── inference.py        # model loading + predict()
│   │   │   ├── preprocess.py       # DICOM/PNG/JPEG → normalized tensor
│   │   │   ├── privacy.py          # DICOM tag stripping, anonymisation
│   │   │   └── schemas.py          # Pydantic request/response models
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   └── web/                # Next.js 16 App Router frontend
│       ├── app/
│       │   ├── page.tsx            # upload + result UI
│       │   └── api/predict/route.ts# proxy to backend (hides API URL)
│       ├── components/
│       └── package.json
├── ml/                     # training code (not deployed)
│   ├── notebooks/
│   ├── train.py
│   ├── evaluate.py         # produces metrics + ROC curve
│   └── data/               # gitignored
├── models/                 # exported model artifact (gitignored, fetched on build)
└── infra/
    └── cloudrun.yaml       # Cloud Run service config
```

## 3. Tech stack (locked)

| Layer | Choice | Why |
|---|---|---|
| ML framework | PyTorch + torchvision | Pretrained model zoo, mature DICOM ecosystem |
| Base model | EfficientNet-B0 (ImageNet pretrained), transfer-learned | Strong accuracy-per-FLOP for small data, fits Cloud Run memory |
| Dataset | CBIS-DDSM (subset) | Largest established, has malignant/benign labels |
| Backend | Python 3.11 + FastAPI + uvicorn | First-class for ML serving; `pydicom` ecosystem |
| Frontend | Next.js 16 (App Router) + Tailwind + shadcn/ui | Fast to ship clean UI; deploys to Vercel in minutes |
| Image handling | `pydicom`, `Pillow`, `numpy` | Standard for medical imaging preprocessing |
| Backend deploy | GCP Cloud Run | Brief prefers GCP; scales to zero; HTTPS by default |
| Frontend deploy | Vercel | Free-tier, custom domain, edge HTTPS |
| Model storage | GCS bucket (private), fetched at container start | Keeps image small; rotatable |

## 4. Key commands

```bash
# install
cd apps/api && uv sync                      # or pip install -e .
cd apps/web && pnpm install

# run locally
cd apps/api && uvicorn app.main:app --reload --port 8080
cd apps/web && pnpm dev                     # http://localhost:3000

# test
cd apps/api && pytest
cd apps/web && pnpm test

# train (one-off, from ml/)
python ml/train.py --data data/cbis-ddsm --epochs 10 --out ../models/mammo-v1.pt
python ml/evaluate.py --model ../models/mammo-v1.pt --report ../REPORT.md

# build + deploy
docker build -t mammo-api apps/api
gcloud run deploy mammo-api --source apps/api --region me-central1
vercel deploy apps/web --prod
```

## 5. Conventions

- **Python:** `ruff` for lint+format, `mypy --strict` on `app/`, `pytest` for tests. Public functions get type hints.
- **TypeScript:** strict mode on, `pnpm` only (no npm/yarn), shadcn/ui for primitives.
- **Commits:** conventional commits (`feat:`, `fix:`, `chore:`). Small, focused.
- **Secrets:** never committed. `.env.local` for dev; Cloud Run env vars / Vercel env vars in deploy.
- **PHI handling:** every code path that touches an uploaded image must route through `apps/api/app/privacy.py` for DICOM tag stripping before any logging, persistence, or response.

## 6. Hard rules

1. **No PHI in logs.** Never log raw DICOM headers, patient IDs, or image bytes. Log only request ID, file size, MIME type, inference latency, prediction label.
2. **No image persistence by default.** Uploaded images live in memory only; deleted after response. Any caching/storage requires explicit opt-in and an encrypted bucket.
3. **TLS everywhere.** Localhost dev is the only exception; staging and prod must be HTTPS-only.
4. **Model is not authoritative.** Every response from the API and every screen on the UI must carry a disclaimer: research/demo only, not for clinical use.
5. **Honest metrics only.** Reported numbers come from `ml/evaluate.py` on a held-out test split; never cherry-picked thresholds without disclosing the operating point.

## 7. Out of scope (for this assessment)

User accounts, auth flows, multi-tenant, billing, audit logs, model retraining UI, multi-class output, segmentation overlays, batch upload, mobile app. Note these in REPORT.md under "What I'd do with more time."

## 8. Pointers

- Brief PDF: `~/Downloads/Full-Stack AI  Feature Engineering Challenge.pdf`
- Design rationale and ML/privacy details: `DESIGN.md`
- Claude Code workflow specifics: `CLAUDE.md`
