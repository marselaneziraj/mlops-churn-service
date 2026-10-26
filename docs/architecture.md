# Architecture

## Request path

```
client
  │  POST /predict  (X-Request-ID optional)
  ▼
observability middleware ─ mint/accept request id → bind to ContextVar
  │                        start latency timer
  ▼
pydantic v2 validation (extra="forbid", Literal enums, range bounds)
  │   invalid → 422 before any model work
  ▼
ModelHolder → ChurnModel  (loaded once at startup, not per request)
  │   records → DataFrame in training column order & dtypes
  ▼
Pipeline: ColumnTransformer(impute+scale | impute+one-hot) → LightGBM → isotonic calibrator
  │
  ▼
threshold comparison (0.074, carried in the same bundle as the model)
  │
  ▼
Prometheus: churn_predictions_total{decision}, churn_predicted_probability
  │
  ▼
response + X-Model-Version, X-Request-ID, X-Response-Time-Ms
```

## Design decisions and the reasons for them

**The threshold ships inside the model bundle.** A threshold is only valid for the
probability distribution of the model it was tuned on. Storing them separately is how a
service quietly starts applying last month's cut-off after a retrain.

**One model load per process, at startup.** `lifespan` loads the bundle; a model already
injected into the holder wins, so tests and warm reloads cannot be clobbered by a failed
disk read.

**Degraded, not dead.** A missing bundle makes `/health` report `degraded` with HTTP 200
and `/predict` return 503. An orchestrator can then distinguish "container up, model
missing" from "container dead" — different alerts, different runbooks.

**`extra="forbid"` on every schema.** A typo in a field name becomes a 422 at the edge
instead of a silently-imputed median deep inside the pipeline.

**`handle_unknown="ignore"` on the encoder.** A category level never seen in training
becomes an all-zero block rather than a 500. The data-contract test asserts the known
vocabulary separately, so the two mechanisms cover different failure modes.

**Request id in a `ContextVar`, not a parameter.** Every log line emitted while handling a
request carries the id, including from code that knows nothing about HTTP.

**Prediction distribution as a first-class metric.** Generic HTTP metrics cover rate,
errors and duration. They cannot tell you the model has started reporting a 44% churn rate
against a 26% base — and labels to prove it are weeks away.

## Container

Multi-stage. The builder installs into `/opt/venv` with `build-essential` present;
the runtime copies only the virtualenv and carries `libgomp1` (LightGBM's OpenMP runtime,
needed at predict time too) plus `curl` for the healthcheck. Runs as uid 10001; CI asserts
the container's effective uid is not 0.

`models/` is mounted read-only in compose so a retrain does not require an image rebuild.

## Compose topology

```
postgres  ──backend store──▶  mlflow  ──serve-artifacts──▶  /mlartifacts volume
                                 ▲
                                 │ (tracking only; not on the request path)
                              api :8080  ──/metrics──▶  prometheus :9090  [profile: observability]
```

MLflow needs Postgres rather than the file store because **the model registry is not
supported on a file backend**. The API depends on MLflow only for tracking, never for
serving, so the service starts and scores with MLflow down.

## CI pipeline

| Job | Gate |
|---|---|
| `lint` | ruff check, ruff format --check, mypy |
| `test` | pytest on 3.11 and 3.12, coverage uploaded |
| `pipeline` | ingest → leakage audit → EDA → train → drift, on synthetic data; asserts a no-drift window does **not** alert and a drifted window **does** |
| `container` | build, run, probe `/health`, `/predict`, `/metrics`, assert non-root, then push to GHCR on `main` |

The drift job asserting both directions is the part most CI pipelines omit: a detector
that always fires is as useless as one that never does.
