# syntax=docker/dockerfile:1.7

# ---------- build deps ----------
# Resolve the venv at the SAME absolute path the runtime image will use
# (/app/.venv) so script shebangs stay valid across stages.
FROM python:3.11-slim AS deps

COPY --from=ghcr.io/astral-sh/uv:0.5.6 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

# Tini reaps zombies and forwards signals — Cloud Run / HF Spaces both expect
# graceful shutdown on SIGTERM.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 1000 app \
 && useradd --uid 1000 --gid 1000 --home /home/app --create-home app

WORKDIR /app

# Copy resolved venv (already at /app/.venv).
COPY --from=deps /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Application code + model weights baked into the image — single artifact,
# no startup network dependency.
COPY app /app/app
COPY model.pt /app/model.pt

ENV MODEL_PATH=/app/model.pt \
    MODEL_VERSION=mammo-v1 \
    PORT=7860

# Make /app and the venv world-readable so the non-root user can use them.
RUN chown -R app:app /app

USER app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/healthz', timeout=4); sys.exit(0)" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
# Use `python -m uvicorn` rather than the `uvicorn` script — works even if
# the venv is relocated.
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
