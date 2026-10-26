# Model card — churn-classifier

Generated for the artefact `models/champion.joblib`, model version
`lightgbm-20261026194237`, trained 2026-10-26 from `configs/config.yaml` with seed 42.
Regenerate every number here with `make all`.

---

## 1. Model details

| Field | Value |
|---|---|
| Name | `churn-classifier` (MLflow registered model) |
| Version | `lightgbm-20261026194237` |
| Type | LightGBM binary classifier inside a scikit-learn `Pipeline`, wrapped in `CalibratedClassifierCV` (isotonic, `cv=5`, `ensemble=False`) |
| Features | 3 numeric + 17 categorical → 45 columns after one-hot encoding |
| Output | Calibrated P(churn) ∈ [0, 1], plus a binary decision at threshold **0.074** |
| Framework | lightgbm ≥ 4.3, scikit-learn ≥ 1.5, Python 3.11 |
| Seed | 42 (`split`, `CalibratedClassifierCV`, LightGBM `random_state`) |
| Artefact | `models/champion.joblib` — pipeline + threshold + metadata in one bundle |
| Licence | MIT |

### Hyperparameters

Deliberately conservative for a 5,634-row training set with ~45 one-hot columns: the
library defaults (`num_leaves: 31`, 400 estimators at lr 0.05) overfit and scored **0.829**
CV ROC-AUC — two points *below* the logistic baseline.

```yaml
objective: binary        n_estimators: 300      learning_rate: 0.03
num_leaves: 8            max_depth: 4           min_child_samples: 60
subsample: 0.8           colsample_bytree: 0.7  reg_lambda: 5.0
```

---

## 2. Intended use

**Intended.** Scoring an existing telecom subscriber base to prioritise retention outreach.
Both a batch nightly scoring job and a low-volume interactive lookup are supported.

**Users.** A retention/marketing team consuming ranked risk, via an internal service. Not
customer-facing.

**Out of scope.**

* Any pricing, credit, eligibility or collections decision.
* Any adverse action against a customer. The model predicts *departure*, and nothing in
  its training supports a punitive use.
* Populations other than the one it was trained on — a different market, product mix or
  billing system needs retraining, not transfer.
* Causal questions. "Will this offer retain this customer?" is an **uplift** question;
  this model answers "is this customer likely to leave?". See limitation 6.

---

## 3. Training data

| Property | Value |
|---|---|
| Source | IBM Telco Customer Churn (public reference dataset) |
| Rows | 7,043 (5,634 train / 1,409 test) |
| Positive rate | 26.54% overall; 26.54% train, 26.54% test |
| Split | Stratified random, 80/20, seed 42 |
| Missingness | `TotalCharges` blank for 11 rows, all with `tenure == 0` |
| Provenance | Recorded per run in `reports/dataset_source.json` |

The dataset has **no event timestamp**, so temporal validation is impossible and the
stratified split assumes a stationary churn process. A `time` split strategy is
implemented in `make_split` and is the correct choice whenever a timestamp exists.

A seeded synthetic generator with the identical schema ships as `data.source: synthetic`
for offline/CI runs. Results in this card come from the real CSV.

### Excluded by construction

`customerID` (identifier) and `Churn` (target) are on a `features.forbidden` list and the
feature set is an explicit allow-list, so a column added upstream cannot silently become a
feature. The leakage audit asserts both.

---

## 4. Evaluation

### Cross-validation (5-fold stratified, training set only)

| Model | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---|---|---|---|
| Majority class | 0.500 ± 0.000 | 0.265 | 0.1949 | 0.5786 |
| Logistic regression | 0.846 ± 0.013 | 0.661 | 0.1351 | 0.4167 |
| LightGBM (selected) | 0.849 ± 0.011 | 0.671 | 0.1336 | 0.4119 |

### Held-out test set (n = 1,409)

| Variant | ROC-AUC | PR-AUC | Brier | Log loss | ECE (10 bins) |
|---|---|---|---|---|---|
| Raw | 0.8484 | 0.6632 | 0.1346 | 0.4131 | 0.0197 |
| Isotonic-calibrated (shipped) | 0.8477 | 0.6416 | 0.1353 | 0.4151 | 0.0236 |

### Decision quality at threshold 0.074

| Policy | Precision | Recall | Alert rate | Cost / customer |
|---|---|---|---|---|
| Treat nobody | — | 0.000 | 0% | 132.72 |
| Naive 0.5 | 0.640 | 0.594 | 24.6% | 58.37 |
| Treat everybody | 0.265 | 1.000 | 100% | 36.73 |
| **Shipped (0.074)** | 0.355 | 0.981 | 73.4% | **26.15** |

Threshold selected on out-of-fold *training* predictions, never on the test set.

---

## 5. Calibration

Isotonic regression, `cv=5`, `ensemble=False`.

**It did not improve this model.** Brier 0.1346 → 0.1353, ECE 0.0197 → 0.0236. The
conservative hyperparameters leave LightGBM already near-calibrated, and isotonic
regression spends degrees of freedom for nothing. The step is retained because a
cost-derived threshold is a claim about probabilities and the next retrain may not be so
well-behaved — but the measured effect here is a marginal regression, and this card does
not claim otherwise.

`ensemble=False` is a serving decision as much as a statistical one: the default
(`True`) keeps `cv` calibrated copies and runs all of them per request, measured at
**162 ms** vs **26 ms** single-row.

---

## 6. Decision threshold and cost matrix

```
C_FN = 500   missed churner (lost remaining margin)
C_FP =  50   wasted retention offer
C_TP = C_TN = 0
p*   = C_FP / (C_FP + C_FN) = 0.0909   (analytic break-even)
t*   = 0.074                           (empirical, out-of-fold cost sweep)
```

The two agree to within 0.017, which is the practical calibration check.

**The cost matrix is an assumption, not a measurement.** 500 and 50 are plausible
placeholders. Every figure downstream — threshold, savings, cost SLO — inherits them.
Replace them with finance's numbers before quoting any of this.

---

## 7. Explainability

SHAP (TreeExplainer) over a 2,000-row training sample; see `docs/figures/shap_summary.png`.
Top contributors to the log-odds of churn:

1. `Contract = Month-to-month` (largest single effect)
2. `tenure` (short tenure → higher risk)
3. `OnlineSecurity = No`
4. `Contract = Two year` (strongly protective)
5. `MonthlyCharges` (higher → higher risk)
6. `InternetService = Fiber optic`
7. `TechSupport = No`
8. `PaymentMethod = Electronic check`

Consistent with the EDA and with published analyses of this dataset. This is the weakest
but most necessary validation: the model has not learned something absurd.

---

## 8. Fairness considerations

`gender` and `SeniorCitizen` are in the feature set, inherited from the source dataset.

* `gender` ranks last among the 20 displayed SHAP features — near-zero contribution — and
  is a candidate for removal on principle rather than performance.
* `SeniorCitizen` carries real signal, which makes it the harder case: age is a protected
  characteristic in many jurisdictions and the model uses it to allocate a *benefit*
  (a discount), which is a materially different posture from denying one.

**No subgroup performance analysis has been run.** Before any production use, evaluate
ROC-AUC, calibration and alert rate per subgroup for `gender` and `SeniorCitizen`, and
decide explicitly whether either belongs in the feature set. The pipeline makes this a
one-line config change. This is a known gap, not a clean bill of health.

---

## 9. Monitoring plan

| Signal | Source | Warn | Alert | Cadence |
|---|---|---|---|---|
| Feature PSI | `drift_report.py` | 0.10 | 0.25 | daily |
| Prediction PSI | `drift_report.py` | 0.10 | 0.25 | daily |
| Mean P(churn) vs base rate | `/metrics` histogram | ±20% | ±40% | continuous |
| p95 latency | Prometheus | 100 ms | 150 ms | continuous |
| Error rate | Prometheus | 0.5% | 1% | continuous |
| Model age | `/health` `trained_at` | 30 d | 60 d | daily |
| Realised precision/recall | labels, on delay | — | — | monthly |

Validated against injected drift: no-drift windows give max PSI 0.003 (no alert);
a covariate shift gives 0.889 with five signals breaching (alert, exit 1).

---

## 10. Caveats and recommendations

1. Retrain at least every 30 days, or when feature PSI breaches 0.25.
2. Re-derive the threshold on every retrain — it is only valid for its own model.
3. Replace the invented cost matrix before acting on any cost figure.
4. Run subgroup fairness analysis (section 8) before production.
5. Prefer the logistic baseline if operational simplicity matters: 0.003 AUC is not worth
   a gradient-boosting dependency in isolation.
6. Treat the 73.4% alert rate as a signal that the real problem is top-K under a capacity
   constraint, not threshold selection.
7. Batch requests wherever possible: 0.43 ms/row batched vs ~41 ms single.
