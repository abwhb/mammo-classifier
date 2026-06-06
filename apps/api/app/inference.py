"""Model loading and inference.

Phase 0 stub: returns a deterministic dummy probability so the API boots and the
frontend can wire up end-to-end before Phase 2 trains the real model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ModelBundle:
    """Holds the loaded model + threshold. Created once at startup."""

    def __init__(self, model_path: str, model_version: str, threshold: float) -> None:
        self.model_version = model_version
        self.threshold = threshold
        self.loaded = False
        self._model: object | None = None
        path = Path(model_path)
        if path.exists():
            self._load_real_model(path)
        # In Phase 0 the model file does not exist yet; we run in stub mode.

    def _load_real_model(self, path: Path) -> None:
        # Wired up in Phase 3 once train.py produces an artifact.
        # Kept thin here to avoid importing torch unless we actually have a model.
        import torch  # noqa: PLC0415

        state = torch.load(path, map_location="cpu", weights_only=True)
        # Expected structure produced by ml/train.py:
        #   {"state_dict": ..., "threshold": float, "model_version": str, "arch": "efficientnet_b0"}
        from torchvision.models import efficientnet_b0  # noqa: PLC0415

        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 1)
        model.load_state_dict(state["state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        self.threshold = float(state.get("threshold", self.threshold))
        self.model_version = str(state.get("model_version", self.model_version))
        self.loaded = True

    def predict(self, tensor: np.ndarray) -> float:
        """Return P(malignant) in [0, 1]."""
        if self._model is None:
            # Deterministic stub: hash the tensor bytes to a stable probability so
            # the same upload always returns the same response in dev.
            digest = int(np.abs(tensor).sum() * 1e6) % 1000
            return 0.1 + 0.8 * (digest / 1000.0)
        import torch  # noqa: PLC0415

        with torch.no_grad():
            x = torch.from_numpy(tensor).float()
            if x.ndim == 3:
                x = x.unsqueeze(0)
            logit = self._model(x)  # type: ignore[operator]
            prob = torch.sigmoid(logit).squeeze().item()
        return float(prob)
