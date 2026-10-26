"""Prometheus instruments.

Four signals, chosen to cover the RED method (Rate, Errors, Duration) plus the
one thing generic HTTP metrics cannot tell you: the *distribution of predictions*.
A silent model failure usually shows up there first -- the churn rate the model
reports drifts away from the training base rate long before accuracy can be
measured, because labels arrive weeks later.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS = Counter(
    "churn_requests_total",
    "HTTP requests handled by the churn service",
    labelnames=("endpoint", "method", "status"),
    registry=REGISTRY,
)

LATENCY = Histogram(
    "churn_request_latency_seconds",
    "End-to-end request latency",
    labelnames=("endpoint",),
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)

PREDICTIONS = Counter(
    "churn_predictions_total",
    "Predictions emitted, split by the decision taken",
    labelnames=("decision",),
    registry=REGISTRY,
)

PREDICTED_PROBABILITY = Histogram(
    "churn_predicted_probability",
    "Distribution of predicted churn probabilities",
    buckets=(0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0),
    registry=REGISTRY,
)

MODEL_INFO = Gauge(
    "churn_model_loaded",
    "1 when a model bundle is loaded and serving",
    labelnames=("model_version",),
    registry=REGISTRY,
)
