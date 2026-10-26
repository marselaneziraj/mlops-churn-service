# Windows note: these targets assume a POSIX-ish shell. On Windows use Git Bash,
# WSL, or run the underlying commands directly (they are all plain `python -m`).
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help setup install hooks lint format type test test-int cov data audit eda train \
        promote api docker-build docker-up docker-down drift load clean all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv and install the project with dev extras
	$(PY) -m venv .venv
	.venv/bin/python -m pip install -U pip
	.venv/bin/python -m pip install -e ".[dev,tracking,monitoring,loadtest]"

install: ## Install the project (editable) into the active interpreter
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev,tracking,monitoring,loadtest]"

hooks: ## Install pre-commit hooks
	pre-commit install

lint: ## ruff check
	ruff check src tests
	ruff format --check src tests

format: ## ruff autofix + format
	ruff check --fix src tests
	ruff format src tests

type: ## mypy
	mypy

test: ## Unit tests
	pytest -m "not integration and not slow"

test-int: ## Integration tests (API)
	pytest -m integration

cov: ## Full suite with coverage
	pytest --cov --cov-report=term-missing --cov-report=xml

data: ## Download/generate the raw dataset
	$(PY) -m churn.data.make_dataset --config configs/config.yaml

audit: ## Leakage audit
	$(PY) -m churn.data.leakage_audit --config configs/config.yaml

eda: ## EDA figures into docs/figures
	$(PY) -m churn.data.eda --config configs/config.yaml

train: ## Train baselines + LightGBM, calibrate, pick threshold, log to MLflow
	$(PY) -m churn.models.train --config configs/config.yaml

promote: ## Promote the best run to the champion alias in the registry
	$(PY) -m churn.models.promote_champion --config configs/config.yaml

api: ## Run the API locally with reload
	uvicorn churn.api.main:app --reload --port 8080

docker-build: ## Build the service image
	docker build -t mlops-churn-service:local .

docker-up: ## api + mlflow + postgres
	docker compose up -d --build

docker-down: ## Stop the stack
	docker compose down -v

drift: ## Simulate drifting traffic and write the Evidently report
	$(PY) -m churn.monitoring.simulate_traffic --config configs/config.yaml
	$(PY) -m churn.monitoring.drift_report --config configs/config.yaml

load: ## Locust headless load test against a running API
	locust -f loadtest/locustfile.py --headless -u 50 -r 10 -t 1m --host http://localhost:8080

all: data audit eda train ## Full offline pipeline

clean: ## Remove caches and generated reports
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml reports/*.json
