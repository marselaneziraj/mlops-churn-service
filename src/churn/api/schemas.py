"""Pydantic v2 request/response schemas.

The schema is the service's contract. ``extra="forbid"`` means a typo in a field
name is a 422 at the edge rather than a silently-imputed median deep inside the
pipeline, and the ``Literal`` enums reject category levels the encoder has never
seen. Field constraints mirror the observed ranges in the training data; the
monitoring job is what tells you when reality has moved outside them.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
YesNoPhone = Literal["Yes", "No", "No phone service"]
YesNoNet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer's features, named exactly as the training columns."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)

    gender: Literal["Female", "Male"]
    SeniorCitizen: Annotated[int, Field(ge=0, le=1)]
    Partner: YesNo
    Dependents: YesNo
    tenure: Annotated[int, Field(ge=0, le=120, description="months on the books")]
    PhoneService: YesNo
    MultipleLines: YesNoPhone
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNet
    OnlineBackup: YesNoNet
    DeviceProtection: YesNoNet
    TechSupport: YesNoNet
    StreamingTV: YesNoNet
    StreamingMovies: YesNoNet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: Annotated[float, Field(ge=0, le=1000)]
    TotalCharges: Annotated[float | None, Field(ge=0, le=100000)] = None


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer: Customer
    customer_id: str | None = Field(default=None, description="echoed back for joining")


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: Annotated[list[Customer], Field(min_length=1, max_length=1000)]
    customer_ids: list[str] | None = None


class Prediction(BaseModel):
    customer_id: str | None = None
    churn_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    churn_label: bool = Field(description="probability >= the cost-optimal threshold")
    threshold: float
    recommended_action: Literal["retention_offer", "no_action"]


class PredictResponse(BaseModel):
    prediction: Prediction
    model_version: str
    request_id: str
    latency_ms: float


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    request_id: str
    latency_ms: float
    n: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    model_name: str | None = None
    threshold: float | None = None
    trained_at: str | None = None
    api_version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
