# Mammogram Classifier

**FAHM Biotechnology Technical Assessment — Full-Stack AI & Feature Implementation Lead**

End-to-end binary mammogram classifier (Malignant vs Benign/Normal) with a confidence score. Built as the 1-day proxy task described in the brief.

> **Research / demonstration only.** Not a medical device. Not approved or intended for clinical use.

---

## Stack at a glance

| Layer | Tech |
|---|---|
| ML | PyTorch + EfficientNet-B0 (ImageNet transfer-learned on CBIS-DDSM subset) |
| Backend | Python 3.11 + FastAPI + uvicorn + pydicom |
| Frontend | Next.js 16 (App Router) + Tailwind v4 |
| Deploy | Hugging Face Spaces (Docker) for API, Vercel for web |

See [`DESIGN.md`](./DESIGN.md) for architecture and rationale, [`AGENT.md`](./AGENT.md) for repo conventions, and [`REPORT.md`](./REPORT.md) for the technical report.

## Prerequisites

- Python 3.11 (via [`uv`](https://docs.astral.sh/uv/))
- Node.js 22+ and `pnpm`
- (Optional, for training) A GPU — CUDA or Apple Silicon MPS. CPU training works but is slow.

## Live URLs

- **Frontend:** <https://mammo-classifier.vercel.app>
- **Backend health:** <https://abwhb-mammo-classifier-api.hf.space/healthz>

## Quick start (local)

```bash
make install        # sync apps/api (uv) and apps/web (pnpm)
make api            # FastAPI on http://localhost:8080
make web            # Next.js on http://localhost:3000   (separate terminal)
make test           # Run backend tests (7/7 expected)
```

Open <http://localhost:3000> and upload a DICOM, PNG, or JPEG mammogram.

> **No model artifact yet?** The API boots in "degraded" mode with a deterministic stub predictor so you can wire up the frontend before training finishes. `GET /healthz` reports `model_loaded: false` until you train one (Phase 2) and set `MODEL_PATH=/path/to/mammo-v1.pt`.

## Repository layout

```
.
├── AGENT.md / CLAUDE.md / DESIGN.md / REPORT.md    docs
├── apps/
│   ├── api/                 FastAPI backend (Python)
│   └── web/                 Next.js frontend (TypeScript)
├── ml/                      Training + evaluation
├── infra/                   Deploy configs
├── models/                  Model artifacts (gitignored)
├── reports/                 Generated metrics + ROC plot
└── samples/                 Sample inputs for manual testing (gitignored)
```

## Environment variables

### Backend (`apps/api`)

| Var | Default | Notes |
|---|---|---|
| `MODEL_PATH` | `/app/models/mammo-v1.pt` | Path to the trained `.pt`; missing → stub mode |
| `MODEL_VERSION` | `mammo-v0-stub` | Reported in responses for traceability |
| `DECISION_THRESHOLD` | `0.5` | Overridden by the value stored in the model artifact |
| `MAX_UPLOAD_BYTES` | `20971520` (20 MB) | Hard upload cap |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS list |
| `INTERNAL_TOKEN` | _(unset)_ | If set, `/predict` requires header `X-Internal-Token` |

### Frontend (`apps/web`)

| Var | Default | Notes |
|---|---|---|
| `API_URL` | `http://127.0.0.1:8080` | Backend base URL — server-only |
| `INTERNAL_TOKEN` | _(unset)_ | Forwarded as `X-Internal-Token` to backend |

## Training & evaluation

Phase 2 (in progress) will land:

```bash
cd ml
uv sync
python train.py     --data ../ml/data/cbis-ddsm --epochs 10 --out ../models/mammo-v1.pt
python evaluate.py  --model ../models/mammo-v1.pt --report ../REPORT.md
```

Metrics, ROC plot, and confusion matrix land in `reports/`.

## Deploy

```bash
# Backend → Hugging Face Docker Space
HF_TOKEN=hf_xxx uv run --with huggingface_hub --no-project \
    python infra/deploy_hf_space.py --repo-id <user>/<space-name>

# Frontend → Vercel
cd apps/web
vercel link --yes --project mammo-classifier --scope <your-team>
echo "https://<user>-<space-name>.hf.space" | vercel env add API_URL production
vercel deploy --prod --yes
```

The Dockerfile is portable; the same image runs on GCP Cloud Run unchanged
(see `DESIGN.md` §5 for the intended in-Kingdom production target).

## Generating the technical report (PDF)

```bash
uv run --with markdown --with pygments --with playwright --no-project \
    python scripts/render_report.py
# writes reports/REPORT.html and reports/REPORT.pdf
```

The renderer uses Chromium headless via Playwright — no LaTeX toolchain needed.

## Privacy

- DICOM uploads are de-identified server-side **before** any other processing (`apps/api/app/privacy.py`).
- Uploaded images are held in memory only; never written to disk.
- Logs contain only request ID, file size, MIME type, latency, label — no PHI.

Full framework and Saudi PDPL alignment in `DESIGN.md` §4.

## License

Assessment submission. No license granted for redistribution.
