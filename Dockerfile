# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# Multi-stage build. The builder compiles wheels (LightGBM needs libgomp and a
# compiler); the runtime carries only the virtualenv, the package and the model
# bundle. Result: no compilers, no pip cache, no source tree in the shipped image.
# -----------------------------------------------------------------------------

# --------------------------------- builder -----------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency layer first: it only invalidates when pyproject changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# --------------------------------- runtime -----------------------------------
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="mlops-churn-service" \
      org.opencontainers.image.description="Calibrated churn scoring API with cost-based thresholding" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/champion.joblib \
    LOG_LEVEL=INFO

# libgomp is LightGBM's OpenMP runtime; it is needed at predict time too.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 10001 appuser \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser configs ./configs
COPY --chown=appuser:appuser models ./models

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "churn.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
