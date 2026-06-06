from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, load_settings
from app.inference import ModelBundle
from app.logging_setup import configure_logging, get_logger
from app.preprocess import to_tensor
from app.privacy import strip_identifiers
from app.schemas import HealthResponse, PredictionResponse

configure_logging()
log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    bundle = ModelBundle(
        model_path=settings.model_path,
        model_version=settings.model_version,
        threshold=settings.threshold,
    )
    app.state.settings = settings
    app.state.model = bundle
    log.info(
        "service.startup",
        model_loaded=bundle.loaded,
        model_version=bundle.model_version,
        threshold=bundle.threshold,
    )
    yield
    log.info("service.shutdown")


app = FastAPI(
    title="Mammogram Classifier API",
    version="0.1.0",
    description=(
        "Binary mammogram classifier (research/demo only — not for clinical use). "
        "Built for the FAHM Biotechnology technical assessment."
    ),
    lifespan=lifespan,
)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_model(request: Request) -> ModelBundle:
    return request.app.state.model  # type: ignore[no-any-return]


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(load_settings().allowed_origins),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)


def _check_internal_token(
    settings: Settings,
    x_internal_token: str | None,
) -> None:
    if settings.internal_token is None:
        return
    if x_internal_token != settings.internal_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")


@app.get("/healthz", response_model=HealthResponse)
async def healthz(model: ModelBundle = Depends(get_model)) -> HealthResponse:
    return HealthResponse(
        status="ok" if model.loaded else "degraded",
        model_loaded=model.loaded,
        model_version=model.model_version,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    x_internal_token: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
    model: ModelBundle = Depends(get_model),
) -> PredictionResponse:
    _check_internal_token(settings, x_internal_token)
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        log.warning(
            "predict.too_large",
            request_id=request_id,
            size=len(raw),
            limit=settings.max_upload_bytes,
        )
        raise HTTPException(status_code=413, detail="file too large")

    # Privacy first — strip PHI before anything else touches the bytes.
    cleaned = strip_identifiers(raw)

    try:
        tensor = to_tensor(cleaned)
    except Exception as exc:
        log.warning("predict.preprocess_failed", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=400, detail="unable to decode image") from exc

    prob = model.predict(tensor)
    label = "malignant" if prob >= model.threshold else "benign"
    confidence = prob if label == "malignant" else 1.0 - prob
    latency_ms = (time.perf_counter() - start) * 1000.0

    # Structured log — no PHI, no filename, no headers, no bytes.
    log.info(
        "predict.ok",
        request_id=request_id,
        size=len(raw),
        mime=file.content_type,
        label=label,
        malignant_probability=round(prob, 4),
        threshold=model.threshold,
        latency_ms=round(latency_ms, 2),
        model_version=model.model_version,
    )

    return PredictionResponse(
        label=label,
        confidence=confidence,
        malignant_probability=prob,
        threshold=model.threshold,
        model_version=model.model_version,
        latency_ms=latency_ms,
        request_id=request_id,
    )


@app.exception_handler(HTTPException)
async def _http_exc_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status": exc.status_code},
    )
