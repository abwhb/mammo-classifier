from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


class PredictionResponse(BaseModel):
    label: Literal["malignant", "benign"]
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the predicted label")
    malignant_probability: float = Field(
        ge=0.0, le=1.0, description="Raw sigmoid probability for the malignant class"
    )
    threshold: float = Field(ge=0.0, le=1.0, description="Operating threshold applied")
    model_version: str
    latency_ms: float
    request_id: str
    disclaimer: str = (
        "Research / demonstration only. Not a medical device. "
        "Not for clinical decision-making."
    )
