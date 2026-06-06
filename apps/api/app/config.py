import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_path: str
    model_version: str
    threshold: float
    max_upload_bytes: int
    allowed_origins: tuple[str, ...]
    internal_token: str | None


def load_settings() -> Settings:
    origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    return Settings(
        model_path=os.getenv("MODEL_PATH", "/app/models/mammo-v1.pt"),
        model_version=os.getenv("MODEL_VERSION", "mammo-v0-stub"),
        threshold=float(os.getenv("DECISION_THRESHOLD", "0.5")),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
        allowed_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        internal_token=os.getenv("INTERNAL_TOKEN") or None,
    )
