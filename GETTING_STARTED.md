# Getting started — VS Code and GitHub

Two parts: run it locally in VS Code, then publish it to GitHub.
Commands are written for **Windows PowerShell** (the VS Code default terminal on Windows);
the macOS/Linux equivalent is noted where it differs.

---

## Part 1 — Run it in VS Code

### 1. Open the folder

Unzip `mlops-churn-service.zip`, then in VS Code: **File → Open Folder…** and pick the
`mlops-churn-service` folder (the one containing `pyproject.toml`).

VS Code will offer to install the recommended extensions from `.vscode/extensions.json`.
Accept — you want Python, Pylance, Ruff, and Docker.

### 2. Create the virtual environment

Open a terminal in VS Code with **Ctrl+`** , then:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,tracking,monitoring,loadtest]"
```

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate` then the same two pip lines.

If PowerShell blocks the activate script with an execution-policy error, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Expect the install to take 2–5 minutes; it pulls LightGBM, MLflow, Evidently and SHAP.

### 3. Point VS Code at the interpreter

**Ctrl+Shift+P → "Python: Select Interpreter"** → choose the one that says
`.venv` (`./.venv/Scripts/python.exe`). `.vscode/settings.json` already expects this path,
so the editor, the test runner and the debugger all line up.

Reload the window afterwards (**Ctrl+Shift+P → "Developer: Reload Window"**) so Pylance
picks up the `src/` layout.

### 4. Run the pipeline

Press **Ctrl+Shift+B** — the default build task runs the whole offline pipeline:
ingest → leakage audit → EDA → train.

Or by hand:

```powershell
python -m churn.data.make_dataset  --config configs/config.yaml
python -m churn.data.leakage_audit --config configs/config.yaml
python -m churn.data.eda           --config configs/config.yaml
python -m churn.models.train       --config configs/config.yaml
```

Training takes roughly 1–2 minutes on a laptop. When it finishes you will have
`models/champion.joblib`, `reports/metrics.json`, and fresh figures in `docs/figures/`.

No internet? Set `data.source: synthetic` in `configs/config.yaml` — the pipeline then
generates a seeded dataset with the identical schema and never touches the network.

### 5. Run the tests

Open the **Testing** panel (the beaker icon in the sidebar) — pytest discovery is already
configured, so all 80 tests appear in a tree and can be run or debugged individually.
From the terminal: `pytest`.

### 6. Run the API

Press **F5** and choose **"FastAPI (uvicorn)"** from the debug dropdown, or:

```powershell
uvicorn churn.api.main:app --reload --port 8080
```

Then open <http://localhost:8080/docs> for the interactive Swagger UI, and try a request:

```powershell
curl.exe -s -X POST http://localhost:8080/predict `
  -H "Content-Type: application/json" `
  -d "@tests/fixtures/sample_request.json"
```

Breakpoints work in any debug configuration — including inside `predict()` while a request
is in flight.

### 7. Optional — the full stack

With Docker Desktop running:

```powershell
docker compose up -d --build
```

API on <http://localhost:8080>, MLflow UI on <http://localhost:5000>.
Stop with `docker compose down -v`.

---

## Part 2 — Push it to GitHub

Three stages, and **the GitHub repository has to be created in the middle**, so read each
block's output before moving to the next.

### Stage A — commit locally

Paste this whole block into the VS Code terminal, in the project folder:

```powershell
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"

git init
git branch -M main
git add .
git commit -m "MLOps churn service: pipeline, calibrated model, API, CI, drift monitoring"
git log --oneline
git status --short
```

Two things to get right here:

* **Use the email address attached to your GitHub account.** If it does not match, the
  commits land in the repo but are not linked to your profile and show no contribution.
  (If you use GitHub's private-email setting, use the `…@users.noreply.github.com` address
  from **GitHub → Settings → Emails**.)
* If `git commit` fails with *"Please tell me who you are"*, the two `git config` lines did
  not run — run them and commit again.

`git status --short` should print nothing. `git log --oneline` should show one commit.
Data, models and reports are deliberately excluded by `.gitignore`; `docs/figures/` is
**not**, so the README images will render on GitHub.

### Stage B — create the empty repository on GitHub

Go to <https://github.com/new> and:

* **Repository name:** `mlops-churn-service`
* **Description:**
  `LightGBM churn model with leakage audit, calibration, cost-based thresholding, FastAPI serving, MLflow registry and PSI drift monitoring.`
* **Public**
* **Do NOT** tick "Add a README", "Add .gitignore" or "Choose a licence" — the repo already
  has all three, and initialising them on GitHub creates a conflicting commit you would
  then have to merge.

Click **Create repository**. This step is the one people skip; without it `git push` has
nowhere to go and fails with *"repository not found"*.

### Stage C — push

Replace `YOUR-USERNAME`, then paste:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/mlops-churn-service.git
git push -u origin main
```

The first push opens a **browser window to sign in to GitHub** (handled by Git Credential
Manager). Complete it there and the push continues on its own.

If a terminal *password* prompt appears instead, close it — GitHub no longer accepts
account passwords on the command line. Use the browser sign-in, or create a personal access
token at **Settings → Developer settings → Personal access tokens** and paste that as the
password.

### Afterwards

* **Actions** tab: the CI workflow runs automatically on the first push — lint, mypy,
  tests on two Python versions, the offline pipeline, and a container build with a smoke
  test. Allow 5–10 minutes.
* The container push to GHCR only runs on `main` and needs no extra secret —
  `GITHUB_TOKEN` is provided automatically.
* Add the topics `mlops`, `machine-learning`, `fastapi`, `mlflow`, `lightgbm`, `docker`
  under the repo's **About** section so it is findable.

### Later changes

```powershell
git add .
git commit -m "Describe what changed"
git push
```

Or use VS Code's **Source Control** panel (Ctrl+Shift+G): stage with `+`, type a message,
click **Commit**, then **Sync Changes**.
