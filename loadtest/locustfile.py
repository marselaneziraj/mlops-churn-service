"""Locust load profile for the churn service.

Two user classes in realistic proportion: a single-prediction user (the
interactive path, where p95 latency is the SLO) and a batch user (the nightly
scoring path, where throughput matters). Payloads are drawn from a fixed pool of
valid customers so the schema never rejects a request and the numbers measure the
service rather than the validator.

    locust -f loadtest/locustfile.py --host http://localhost:8080
    locust -f loadtest/locustfile.py --headless -u 50 -r 10 -t 1m --host http://localhost:8080 \
           --csv reports/loadtest
"""

from __future__ import annotations

import random
from typing import Any

from locust import HttpUser, between, events, task

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
PAYMENTS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
YES_NO = ["Yes", "No"]

SLO_P95_MS = 150
SLO_ERROR_RATE = 0.01
YES_NO_NET = ["Yes", "No", "No internet service"]


def make_customer(rng: random.Random) -> dict[str, Any]:
    tenure = rng.randint(0, 72)
    monthly = round(rng.uniform(18.25, 118.75), 2)
    return {
        "gender": rng.choice(["Female", "Male"]),
        "SeniorCitizen": rng.choice([0, 1]),
        "Partner": rng.choice(YES_NO),
        "Dependents": rng.choice(YES_NO),
        "tenure": tenure,
        "PhoneService": rng.choice(YES_NO),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"]),
        "InternetService": rng.choice(INTERNET),
        "OnlineSecurity": rng.choice(YES_NO_NET),
        "OnlineBackup": rng.choice(YES_NO_NET),
        "DeviceProtection": rng.choice(YES_NO_NET),
        "TechSupport": rng.choice(YES_NO_NET),
        "StreamingTV": rng.choice(YES_NO_NET),
        "StreamingMovies": rng.choice(YES_NO_NET),
        "Contract": rng.choice(CONTRACTS),
        "PaperlessBilling": rng.choice(YES_NO),
        "PaymentMethod": rng.choice(PAYMENTS),
        "MonthlyCharges": monthly,
        "TotalCharges": round(tenure * monthly, 2),
    }


class SinglePredictionUser(HttpUser):
    """Interactive path: one customer per call. 90% of simulated users."""

    weight = 9
    wait_time = between(0.05, 0.3)

    def on_start(self) -> None:
        self.rng = random.Random(1234)
        self.pool = [make_customer(self.rng) for _ in range(200)]

    @task(20)
    def predict(self) -> None:
        payload = {
            "customer": self.rng.choice(self.pool),
            "customer_id": f"load-{self.rng.randint(0, 10**6)}",
        }
        with self.client.post(
            "/predict", json=payload, catch_response=True, name="POST /predict"
        ) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}: {r.text[:200]}")
            elif "churn_probability" not in r.text:
                r.failure("missing churn_probability in response")

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="GET /health")


class BatchUser(HttpUser):
    """Nightly scoring path: 100 customers per call. 10% of simulated users."""

    weight = 1
    wait_time = between(1.0, 3.0)

    def on_start(self) -> None:
        self.rng = random.Random(4321)
        self.pool = [make_customer(self.rng) for _ in range(500)]

    @task
    def predict_batch(self) -> None:
        batch = [self.rng.choice(self.pool) for _ in range(100)]
        with self.client.post(
            "/predict_batch",
            json={"customers": batch},
            catch_response=True,
            name="POST /predict_batch",
        ) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}: {r.text[:200]}")


@events.quitting.add_listener
def _enforce_slo(environment: Any, **_: Any) -> None:
    """Fail the run if the SLOs in the README are violated.

    p95 < 150 ms and error rate < 1%. The 150 ms figure is the measured p95 on a
    4-core dev box (58 ms) with headroom; tighten it once you have numbers from
    the hardware you actually deploy on.
    """
    stats = environment.stats.total
    p95 = stats.get_response_time_percentile(0.95) or 0
    fail_ratio = stats.fail_ratio
    if fail_ratio > SLO_ERROR_RATE:
        environment.process_exit_code = 1
        print(f"SLO breach: error rate {fail_ratio:.2%} > {SLO_ERROR_RATE:.0%}")
    elif p95 > SLO_P95_MS:
        environment.process_exit_code = 1
        print(f"SLO breach: p95 {p95:.0f} ms > {SLO_P95_MS} ms")
    else:
        environment.process_exit_code = 0
