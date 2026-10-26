"""FastAPI service.

Endpoints
---------
``GET  /health``        liveness + which model version is loaded
``POST /predict``       one customer
``POST /predict_batch`` up to 1000 customers in one call
``GET  /metrics``       Prometheus exposition
``GET  /``              service card

Cross-cutting behaviour lives in one middleware: a request id (accepted from
``X-Request-ID`` or minted), the id bound to a ContextVar so every log line emitted
while handling the request carries it, a latency measurement, Prometheus counters,
and the ``X-Model-Version`` header on every response so a prediction can always be
traced back to the artefact that produced it.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from churn import __version__
from churn.api.metrics import (
    LATENCY,
    MODEL_INFO,
    PREDICTED_PROBABILITY,
    PREDICTIONS,
    REGISTRY,
    REQUESTS,
)
from churn.api.model_registry import ModelNotLoadedError, holder
from churn.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
)
from churn.logging_utils import configure_logging, get_logger, request_id_var

log = get_logger(__name__)
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup, not per request.

    A model already injected into the holder (tests, or a warm reload) wins: the
    startup hook must never overwrite a live model with a failed disk read.
    """
    configure_logging()
    if not holder.is_loaded:
        holder.load()
    if holder.is_loaded:
        MODEL_INFO.labels(model_version=holder.model.version).set(1)
    else:
        log.warning("starting without a model; /health will report degraded")
    yield
    log.info("shutting down")


app = FastAPI(
    title="Churn prediction service",
    description=(
        "Calibrated churn probabilities with a cost-optimal decision threshold. "
        "Every response carries the model version that produced it."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def observability_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    endpoint = request.url.path
    try:
        response = await call_next(request)
    except Exception:
        REQUESTS.labels(endpoint=endpoint, method=request.method, status="500").inc()
        log.exception("unhandled error", extra={"endpoint": endpoint})
        raise
    finally:
        elapsed = time.perf_counter() - started
        LATENCY.labels(endpoint=endpoint).observe(elapsed)
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.2f}"
    if holder.is_loaded:
        response.headers["X-Model-Version"] = holder.model.version
    REQUESTS.labels(
        endpoint=endpoint, method=request.method, status=str(response.status_code)
    ).inc()
    if endpoint != "/metrics":
        log.info(
            "request",
            extra={
                "endpoint": endpoint,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": round(elapsed * 1000, 2),
            },
        )
    return response


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"model unavailable: {exc}", "request_id": request_id_var.get()},
    )


def _to_prediction(prob: float, threshold: float, customer_id: str | None) -> Prediction:
    label = prob >= threshold
    PREDICTED_PROBABILITY.observe(prob)
    PREDICTIONS.labels(decision="retention_offer" if label else "no_action").inc()
    return Prediction(
        customer_id=customer_id,
        churn_probability=round(prob, 6),
        churn_label=bool(label),
        threshold=threshold,
        recommended_action="retention_offer" if label else "no_action",
    )


@app.get("/", response_model=dict)
async def root() -> dict[str, Any]:
    return {
        "service": "churn-prediction",
        "version": __version__,
        "model_loaded": holder.is_loaded,
        "endpoints": ["/health", "/predict", "/predict_batch", "/metrics", "/docs"],
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Never fails; reports degraded when no model is loaded."""
    if not holder.is_loaded:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            api_version=__version__,
            uptime_seconds=round(time.time() - START_TIME, 2),
        )
    meta = holder.model.metadata
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=holder.model.version,
        model_name=str(meta.get("model_name", "unknown")),
        threshold=holder.model.threshold,
        trained_at=str(meta.get("trained_at")) if meta.get("trained_at") else None,
        api_version=__version__,
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    started = time.perf_counter()
    model = holder.model
    try:
        prob = model.predict_proba([payload.customer.model_dump()])[0]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"could not score the payload: {exc}") from exc
    return PredictResponse(
        prediction=_to_prediction(prob, model.threshold, payload.customer_id),
        model_version=model.version,
        request_id=request_id_var.get() or "",
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    started = time.perf_counter()
    model = holder.model
    ids = payload.customer_ids
    if ids is not None and len(ids) != len(payload.customers):
        n_ids, n_customers = len(ids), len(payload.customers)
        raise HTTPException(
            status_code=422,
            detail=f"customer_ids has {n_ids} entries but customers has {n_customers}",
        )
    try:
        probs = model.predict_proba([c.model_dump() for c in payload.customers])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"could not score the batch: {exc}") from exc
    predictions = [
        _to_prediction(p, model.threshold, ids[i] if ids else None) for i, p in enumerate(probs)
    ]
    return BatchPredictResponse(
        predictions=predictions,
        model_version=model.version,
        request_id=request_id_var.get() or "",
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        n=len(predictions),
    )


@app.get("/metrics")
async def metrics() -> Response:
    return PlainTextResponse(
        generate_latest(REGISTRY).decode("utf-8"), media_type=CONTENT_TYPE_LATEST
    )
