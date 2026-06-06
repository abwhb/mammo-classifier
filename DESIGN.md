# DESIGN.md

System design and decision log for the FAHM mammogram classifier.

> **Status:** proposed defaults. Items marked **[DECIDE]** are open for adjustment in the planning step before code lands.

---

## 1. Goals and non-goals

**Goals**
- End-to-end deployed app: upload a mammogram → get `{malignant|benign, confidence}` in seconds.
- Honest, reproducible ML evaluation (sensitivity, specificity, AUC — not just accuracy).
- Demonstrable privacy posture: DICOM de-identification, TLS in transit, no PHI at rest.
- A reviewer can clone, follow `README.md`, and reproduce both training and inference within an hour.

**Non-goals**
- Clinical-grade accuracy. This is a 1-day proxy task; we will not match state-of-the-art models or radiologist performance.
- Multi-class output (mass vs calcification subtype), segmentation, BI-RADS scoring.
- Auth, multi-tenancy, audit logging, real PHI handling. We strip identifiers, we don't manage them.
- Online retraining or active learning.

## 2. System architecture

```
┌──────────────┐    HTTPS    ┌──────────────────┐    HTTPS    ┌─────────────────────┐
│              │   upload    │                  │   POST      │                     │
│  Clinician   │────────────▶│  Next.js (web)   │────────────▶│  FastAPI (api)      │
│  Browser     │             │  Vercel Edge     │  multipart  │  Cloud Run, me-     │
│              │◀────────────│                  │◀────────────│  central1           │
└──────────────┘   JSON      └──────────────────┘   JSON      └──────────┬──────────┘
                                                                         │ load once
                                                                         ▼
                                                              ┌─────────────────────┐
                                                              │  EfficientNet-B0    │
                                                              │  finetune (.pt)     │
                                                              │  pulled from GCS    │
                                                              │  on container start │
                                                              └─────────────────────┘
```

**Request lifecycle**
1. Browser POSTs multipart file to Next.js route handler `/api/predict`. The route handler streams the file straight to the FastAPI service (it never lands on disk in the Vercel function).
2. FastAPI receives the upload, routes it through `privacy.strip_identifiers()`, then `preprocess.to_tensor()`.
3. `inference.predict()` runs the cached model in a thread pool, returns `(label, prob, model_version, latency_ms)`.
4. JSON response flows back; nothing about the image is persisted; only structured non-PHI metadata is logged.

## 3. Component design

### 3.1 Frontend (`apps/web`)

- **Stack:** Next.js 16 App Router, React 19, TypeScript strict, Tailwind, shadcn/ui.
- **One screen:** dropzone + result card. No router, no auth, no settings page.
- **Components:**
  - `<Dropzone>` — `react-dropzone`, accepts `.dcm`, `.png`, `.jpg`, `.jpeg`, 20 MB cap.
  - `<ResultCard>` — label (color-coded), confidence bar, model version, latency, "research only — not for clinical use" disclaimer.
  - `<PreviewPane>` — for PNG/JPEG previews; for DICOM, just show filename + metadata badge (no client-side render to avoid pulling a DICOM JS lib).
- **State:** plain `useState`. No global store needed.
- **Route handler:** `/api/predict/route.ts` proxies to the Cloud Run URL stored in `API_URL` env var. This hides the backend URL from the client and gives us a single place to add a per-request ID header.

### 3.2 Backend (`apps/api`)

- **Stack:** Python 3.11, FastAPI, uvicorn (1 worker, threadpool for CPU inference).
- **Endpoints:**
  - `GET /healthz` — liveness; returns `{status, model_version}`.
  - `POST /predict` — multipart `file`; returns `{label, confidence, model_version, latency_ms, request_id}`.
- **Hot path:** `main.py` → `privacy.strip_identifiers()` → `preprocess.to_tensor()` → `inference.predict()`.
- **Model loading:** on app startup, pull `gs://$MODEL_BUCKET/$MODEL_VERSION.pt` to `/tmp`, load once into a module-level `torch.nn.Module` in eval mode, `torch.set_grad_enabled(False)`.
- **Concurrency:** Cloud Run gives us 1 vCPU / 2 GB by default. Inference is CPU-bound; we set `--workers 1` and rely on FastAPI's threadpool for I/O overlap. Concurrency cap = 4 to stay within memory.

### 3.3 Preprocessing (`apps/api/app/preprocess.py`)

- **DICOM:** `pydicom.dcmread` → extract `pixel_array` → apply VOI LUT if present (`pydicom.pixel_data_handlers.util.apply_voi_lut`) → window to display range → uint16 → float32.
- **PNG/JPEG:** `PIL.Image.open` → convert to L (grayscale) → numpy float32.
- **Common path:** percentile-clip (1st/99th), min-max normalise to `[0,1]`, resize to 224×224 (bilinear), tile to 3 channels, ImageNet mean/std normalisation. Output: `torch.float32` tensor `[1,3,224,224]`.
- **Rationale for 224×224 + 3ch tiling:** matches EfficientNet-B0 pretrained input; lets us reuse ImageNet features. We accept the information loss vs full-resolution patches given our 1-day budget; flagged in `REPORT.md` under future work.

### 3.4 ML model (`ml/`)

- **Dataset:** **CBIS-DDSM** (Curated Breast Imaging Subset of DDSM). Public, peer-reviewed, has `pathology` labels (MALIGNANT / BENIGN / BENIGN_WITHOUT_CALLBACK) we can binarise. *Hosted on TCIA — large download; we'll work with a stratified subset to fit our time budget.* **[DECIDE]** If TCIA download is slow, fall back to **MIAS** (322 images, ships small, but older labels).
- **Task framing:** binary classification at the image level. Map BENIGN + BENIGN_WITHOUT_CALLBACK → 0, MALIGNANT → 1.
- **Architecture:** EfficientNet-B0 from `torchvision.models`, ImageNet pretrained, final FC replaced with `Linear(in_features, 1)` + sigmoid at inference time. ~5.3M params; fits 2 GB Cloud Run with room.
- **Training recipe:**
  - Split: 70/15/15 train/val/test, stratified by label, **patient-level split** (no patient appears in both train and test — critical for CBIS-DDSM where a patient has multiple views).
  - Augmentation: horizontal flip, small rotations (±10°), random crops at 90% then resize. *No* vertical flips (breast tissue has consistent orientation in standard views).
  - Loss: `BCEWithLogitsLoss(pos_weight=...)` to handle class imbalance.
  - Optimiser: AdamW, lr=1e-4, cosine schedule, 10 epochs, early stop on val AUC.
  - Mixed precision (`torch.cuda.amp`) if we get a GPU; otherwise CPU + smaller subset.
- **Evaluation (`ml/evaluate.py`):**
  - Compute on the held-out test split, report: **Accuracy, Sensitivity (Recall on malignant), Specificity, ROC AUC, PR AUC**, plus the full confusion matrix.
  - Plot and save ROC curve to `reports/roc.png`.
  - Pick operating threshold by **maximising sensitivity at ≥ X% specificity** (X TBD from val) — never the default 0.5 without disclosure. Document the chosen threshold and the trade-off in `REPORT.md`.
- **Honest framing:** the report will say plainly: this is not a clinical model. With 1 day and a subset of CBIS-DDSM, expect AUC in the 0.75–0.85 range, not 0.95+. Class imbalance discussion is mandatory.

### 3.5 Inference path

- Single forward pass per request. No TTA (test-time augmentation) by default — adds latency for marginal gain at this scale.
- Confidence returned as the sigmoid probability of the malignant class.
- Latency target: **< 2 s p95 on Cloud Run** for a single 224×224 inference (B0 on CPU is ~50–150 ms; the rest is upload + preprocess).

## 4. Privacy and security framework

This is the highest-leverage section for evaluation. The brief calls it out explicitly.

### 4.1 In transit

- All public surfaces are HTTPS only.
  - Vercel terminates TLS at the edge for `apps/web` (HSTS on).
  - Cloud Run gives every service a `*.run.app` URL with managed TLS 1.3.
- The Next.js → FastAPI hop is HTTPS server-to-server; the Cloud Run URL is not exposed to the browser.
- We will optionally require an inbound shared-secret header (`X-Internal-Token`) on FastAPI, validated against a Cloud Run secret env var — so the Cloud Run URL is not publicly callable even if leaked.

### 4.2 At rest

- **Uploaded images are never written to disk in production.** FastAPI reads the upload into memory, processes it, discards it. No tempfile.
- **Model weights** sit in a private GCS bucket (uniform bucket-level access, no public read, CMEK-eligible). The bucket is in the same region as Cloud Run.
- **Logs** contain only: request ID (UUID v4 generated server-side), file size in bytes, MIME type, inference latency, predicted label, model version. No filename, no headers, no image bytes.

### 4.3 De-identification protocol (`apps/api/app/privacy.py`)

For DICOM specifically (PNG/JPEG carry no DICOM PHI by definition):
- Parse with `pydicom`.
- Strip all tags in the DICOM Basic Application Confidentiality Profile (PS 3.15 Annex E) Basic Profile set: `PatientName`, `PatientID`, `PatientBirthDate`, `PatientSex`, `AccessionNumber`, `StudyInstanceUID`, `SeriesInstanceUID`, `SOPInstanceUID`, `InstitutionName`, `InstitutionAddress`, `ReferringPhysicianName`, `OperatorsName`, `StudyDate`, `StudyTime`, plus all private tags (`ds.remove_private_tags()`).
- Replace UIDs with newly-generated UIDs (preserves DICOM validity for downstream tooling without retaining identifiers).
- Run *before* anything else touches the file — preprocessing receives an already-anonymised in-memory DICOM.
- Unit-tested with a synthetic DICOM that has all PHI tags populated; test asserts none survive.

### 4.4 Saudi PDPL alignment

The Personal Data Protection Law (Royal Decree M/19, in force 14 Sep 2023) governs personal data processing in Saudi Arabia. Relevant principles and how this design addresses them:

| PDPL principle | How we address it |
|---|---|
| **Lawful basis & purpose limitation** (Arts. 5–6) | The app processes uploads only to produce a single inference response. No secondary use, no analytics on the image. |
| **Data minimisation** (Art. 11) | We extract pixels and discard everything else. No PHI is stored. No user accounts collected. |
| **Storage limitation** (Art. 18) | Images are never persisted. Logs retain only non-identifying metadata. |
| **Security of processing** (Art. 19) | TLS everywhere, private GCS bucket for weights, optional shared-secret on the backend, principle-of-least-privilege service accounts on Cloud Run. |
| **Cross-border transfer** (Art. 29) | Cloud Run region pinned to `me-central1` (Dammam, KSA) to keep processing in-Kingdom. Vercel edge will be configured similarly where possible; if not, we document the gap. |
| **Data subject rights** (Arts. 4, 21–26) | Since we don't store or identify data, access/erasure rights are satisfied by design (there is nothing to access or erase). |
| **Breach notification** (Art. 20) | Documented in `REPORT.md` as a runbook even though we don't process real PHI in this demo. |

This is alignment for a proxy task, not a compliance certification. The report says so plainly.

### 4.5 Disclaimers in-product

- Every UI screen and every `/predict` response includes a "research/demo only — not for clinical use, not a medical device" disclaimer.

## 5. Deployment

### 5.1 Backend → GCP Cloud Run

- Region: `me-central1` (Dammam) for PDPL data-residency posture.
- Service: `mammo-api`, 1 vCPU / 2 GB, min instances 0, max 3, concurrency 4.
- Image: built from `apps/api/Dockerfile`, pushed to Artifact Registry.
- Startup: container fetches `gs://${MODEL_BUCKET}/${MODEL_VERSION}.pt` into `/tmp`, loads, then `uvicorn` binds.
- Env vars: `MODEL_BUCKET`, `MODEL_VERSION`, `INTERNAL_TOKEN`, `ALLOWED_ORIGIN`.
- IAM: service account with `roles/storage.objectViewer` on the model bucket only.

### 5.2 Frontend → Vercel

- Project: `mammo-web`, framework auto-detected (Next.js).
- Env vars: `API_URL` (Cloud Run service URL), `INTERNAL_TOKEN`.
- CORS: Cloud Run sets `Access-Control-Allow-Origin: ${ALLOWED_ORIGIN}` (the Vercel domain).
- The Cloud Run URL is server-only — never exposed to the client bundle. The route handler is the only caller.

### 5.3 CI

- Single GitHub Action: lint + tests on push. No CD initially — deploys are manual via `gcloud run deploy` and `vercel deploy`. CD is a "with more time" item.

## 6. Trade-offs and what we'd do with more time

This list is intentionally explicit so the report can quote it.

| Decision | Trade-off | If we had more time |
|---|---|---|
| Subset of CBIS-DDSM | Higher variance in metrics; less robust | Full dataset, cross-validation, patient-level CV folds |
| EfficientNet-B0 ImageNet pretrain | Domain gap (natural images → mammograms) | Pretrain on RSNA Mammography or similar in-domain corpus |
| 224×224 resize | Loses fine calcification detail | Multi-scale patch ensemble, 512×512+ input |
| Image-level labels only | Ignores view (CC/MLO) and side (L/R) | Per-patient model that fuses 4 views |
| Single-threshold output | Hides the precision/recall trade space | Show ROC interactively in UI, let user pick operating point |
| No authentication | Public endpoint, rate-limit only | OAuth + per-clinician audit log |
| Manual deploys | Drift risk | GitHub Actions → Cloud Run + Vercel on push to `main` |
| No model monitoring | Silent drift | Log prediction distribution, alert on shift |
| In-memory only | No traceability | Encrypted, time-bounded audit storage with patient-consent gating |

## 7. Open questions for the planning step

1. **Dataset access speed** — CBIS-DDSM is hosted on TCIA and can be slow. Confirm we can get a usable subset within the first 2 hours, else pivot to MIAS.
2. **GCP project** — does the user already have a GCP billing account / project, or do we need to create one? Affects deploy time meaningfully.
3. **Model size** — do we have a GPU to train on (Colab free tier? local CUDA?) or are we CPU-only? Determines achievable AUC in the time we have.
4. **Vercel account** — same question; existing account speeds the frontend deploy.
5. **PDF report** — do we generate from `REPORT.md` via Pandoc or do we hand-author in Google Docs/Word? Pandoc is faster and keeps the source in git.

Resolve these in the planning conversation, then we execute.
