# Home Energy Copilot

**CPSC 393 Final Project — Spring 2026**

A two-layer machine-learning pipeline for residential energy efficiency, built on the U.S. Energy Information Administration's RECS 2020 microdata. The system classifies households as efficient / average / inefficient within their climate peer group and predicts annual energy expenditure.

---

## Motivation

Residential energy use accounts for roughly 20% of U.S. greenhouse-gas emissions. Existing energy-audit tools either require an in-person visit (expensive, slow) or rely on naive whole-home benchmarks that ignore climate, occupancy, and equipment context. *Home Energy Copilot* is a teaching example of how to combine classification and regression into a coherent pipeline on a real, weighted survey dataset where target engineering and metric weighting both matter for the conclusions.

The pipeline answers two questions:

1. **Where does this household sit relative to its climate peers?** (classification)
2. **What annual energy bill should we expect?** (regression)

---

## Dataset

**Residential Energy Consumption Survey (RECS) 2020 — Public Use Microdata**, U.S. Energy Information Administration. 18,496 occupied primary residences × 799 columns. Each row carries `NWEIGHT`, a survey weight that scales the sample to roughly 123.5 million U.S. households.

> EIA. *Residential Energy Consumption Survey — 2020 Microdata.* https://www.eia.gov/consumption/residential/data/2020/index.php?view=microdata

The raw CSV (`recs2020_public_v7.csv`) lives in `data/`. After cleaning, the working dataset is 18,495 × 69 (one row dropped for negative `TOTALDOL`; 65 features kept plus targets, weights, and IDs).

### Targets

- **Regression:** `TOTALDOL` — total annual energy expenditure in U.S. dollars.
- **Classification:** `efficiency_class` ∈ {efficient, average, inefficient}, defined as the within-climate-group **tertile** of `TOTALDOL / TOTSQFT_EN` (cost per square foot). Tertiles are computed inside each `BA_climate_grp` so that "efficient" means "low $/sqft *for your climate*", not "low $/sqft globally".

### Climate grouping

`BA_climate_grp` is the Building America climate zone with one merge: **Subarctic** (only 54 households) is folded into **Very-Cold** to keep tertile cuts statistically stable. The seven remaining peer groups are: Cold, Hot-Dry, Hot-Humid, Marine, Mixed-Dry, Mixed-Humid, Very-Cold.

### Sample weights

Every fit and every metric in this project uses `sample_weight = NWEIGHT` so results report at the U.S. residential population scale, not the sample scale.


## Repository layout

```
CPSC393Final/
├── README.md                       # this file
├── data/
│   ├── recs2020_public_v7.csv      # raw RECS 2020 microdata (input)
│   ├── recs2020_clean.pkl          # cleaned 18,495 × 69 DataFrame (notebook 01 output)
│   ├── recs2020_clean.parquet      # parquet copy (when pyarrow is installed)
│   ├── recs2020_clean_meta.json    # column inventory + sentinel scan
│   └── train_test_split.json       # frozen 80/20 stratified split (DOEID lists)
├── notebooks/
│   ├── 01_data_loading.ipynb       # load, clean, derive end-use $, freeze targets
│   ├── 02_eda.ipynb                # weighted EDA, distributions, correlations
│   ├── 03_classification.ipynb     # multinomial logistic regression
│   ├── 04_regression.ipynb         # Elastic Net + Gradient Boosting on TOTALDOL
│   └── 05_results_and_discussion.ipynb   # synthesis, analysis
└── models/
    ├── logreg_classification.joblib
    ├── logreg_test_predictions.csv
    ├── classifier_benchmark.csv
    ├── elasticnet_regression.joblib
    ├── gbr_regression.joblib
    └── regression_test_predictions.csv
```

## How to run

### Environment

Python 3.11+ with:

```bash
pip install -r requirements.txt
```

The versions are pinned because the saved `.joblib` model artifacts are sensitive to `numpy` / `scikit-learn` compatibility.

### Execution order

The notebooks are designed to run **in order**, sharing state through saved CSVs and joblib files. Re-running an upstream notebook invalidates downstream artifacts.

1. `01_data_loading.ipynb` — produces `data/recs2020_clean.pkl`, `data/recs2020_clean.parquet`, and `data/recs2020_clean_meta.json`.
2. `02_eda.ipynb` — read-only inspection of the cleaned data (no artifacts saved beyond plots).
3. `03_classification.ipynb` — produces `data/train_test_split.json`, `models/logreg_classification.joblib`, `models/logreg_test_predictions.csv`, and `models/classifier_benchmark.csv`.
4. `04_regression.ipynb` — produces `models/elasticnet_regression.joblib`, `models/gbr_regression.joblib`, `models/regression_test_predictions.csv`.
5. `05_results_and_discussion.ipynb` — read-only synthesis of everything above.

Notebook 04 grid-searches a Gradient Boosting model and is the slowest step (a few minutes on a laptop). Everything else is sub-minute.

## Pipeline architecture

### Layer 1 — Classification

**Model:** `LogisticRegression(solver='lbfgs', class_weight=None)` inside an `sklearn.Pipeline` with a `ColumnTransformer` (median-imputed + standard-scaled numeric features, one-hot-encoded categoricals). `GridSearchCV` over `C ∈ {0.01, 0.1, 1, 10}` with `scoring='f1_weighted'`.

**Features:** 55 columns (29 numeric, 26 categorical). After OHE: ~216 features.

**Metrics:** weighted overall accuracy ≈ 60% (33% baseline), per-climate accuracy reported separately.

**Why this model:** Multinomial Logistic Regression is the main report classifier because it is fast, stable on one-hot encoded tabular features, and interpretable: the coefficient table can explain which household features push a prediction toward "inefficient." It is not the highest-complexity model we tried; it is the model that best matches the project goal of a transparent first-pass energy diagnosis.

**Classifier benchmark:** we also ran quick nonlinear classifier checks on the same split and features. A Gradient Boosting classifier can improve held-out weighted F1 modestly, while Random Forest did not beat Logistic Regression. We kept Logistic Regression as the report model for interpretability and presentation consistency, but an accuracy-first revision should promote and tune the Gradient Boosting classifier.

| Classifier | Weighted accuracy | Weighted F1 | Role |
|---|---:|---:|---|
| Dummy majority | 0.352 | 0.183 | naive baseline |
| Logistic Regression | 0.604 | 0.600 | final interpretable report model |
| Random Forest quick check | 0.582 | 0.575 | nonlinear comparison, worse |
| Gradient Boosting Classifier quick check | 0.621 | 0.620 | best quick accuracy check; future promoted model |

### Layer 2 — Regression

Two models compared head-to-head on the same train/test split:

- **`ElasticNet`** — combined L1/L2 regularization. `GridSearchCV` over `alpha ∈ {0.1, 1.0, 10, 100}` × `l1_ratio ∈ {0.1, 0.5, 0.9}`, `max_iter=20000`, scored by negative RMSE.
- **`GradientBoostingRegressor(max_depth=4)`** — captures nonlinear interactions. `GridSearchCV` over `n_estimators ∈ {200, 400}` × `learning_rate ∈ {0.05, 0.1}`.

**Metrics (weighted):** GBR achieves RMSE ≈ $745, R² ≈ 0.51, marginally better than Elastic Net on most climate groups.

The original proposal called for SVR; we replaced it with Gradient Boosting because SVR scales poorly past a few thousand rows and ~200 features.

### What we did not do, and why

- **We did not keep SVR.** Kernel SVR training scales poorly at this sample size after one-hot encoding, and the RBF kernel is a poor practical fit for a 200+ dimensional mixed survey feature matrix.
- **We did not use a neural network.** RECS is tabular, modest-sized, and heavily categorical; interpretability and reproducibility are more valuable here than adding a black-box model.
- **We did not build a production recommender or ROI calculator.** The validated project scope is modeling, weighted evaluation, and interpretation. Recommendation claims would require intervention costs and causal assumptions that are not in RECS.
- **We did not optimize only for one metric.** The final story balances weighted performance, runtime, interpretability, and whether the model can be defended in a report/presentation.

---

## Key design choices

| Decision | Choice | Why |
|---|---|---|
| Classification target definition | tertile of `cost_per_sqft` within `BA_climate_grp` | Climate normalizes for unavoidable load; per-sqft normalizes for size. |
| Sample weighting | `NWEIGHT` for final fits and held-out metrics | Reports population-scale results, not sample-scale. CV passes weights into fitting; the held-out weighted test set is the final comparison. |
| Sentinel handling (`-2`, `-9`) | Kept as own category for OHE features; median-imputed for numeric features in the pipeline | Preserves "not applicable" signal for categoricals; keeps numeric scales sane. |
| Train/test split | 80/20 stratified on `efficiency_class × BA_climate_grp`, frozen as DOEID list | Same split shared across notebooks 03–05. |
| Regression model swap | `GradientBoostingRegressor` instead of SVR | SVR doesn't scale; GBR captures nonlinearities. |
| Classification model choice | Logistic Regression as report model; Gradient Boosting classifier as accuracy-first upgrade | Logistic is more interpretable and already strong; Gradient Boosting improved quick held-out F1 but needs report/slide promotion before becoming the headline model. |
| End-use $ derivation | Derive `heat_cool_dol`, `water_heat_dol`, `lighting_dol`, `fridge_dol` from per-fuel BTU columns × per-fuel $/BTU rates; `other_dol` as the residual | Lets EDA show *where* the dollars are concentrated within `TOTALDOL`. Reconciles to total within $0.04 mean error. |

## Headline results

| Layer | Metric | Result | Baseline |
|---|---|---|---|
| Classification | Weighted accuracy | **60.4%** | 33.3% (random) |
| Regression (GBR) | Weighted RMSE | **$745** | std(TOTALDOL) ≈ $1,063 |
| Regression (GBR) | Weighted R² | **0.51** | — |

The two layers are **complementary, not redundant**: classification ranks households relative to their climate peers on cost per square foot, while regression predicts absolute dollar bills. A small drafty home can be classified inefficient while still having a modest total bill, and a large well-insulated home can have a high total bill while remaining efficient for its size and climate. Notebook 05 verifies this empirically.

## Limitations

- **RECS scope.** Cross-sectional, occupied primary U.S. residences only. Vacant homes, second homes, and territories are excluded. Survey was fielded during the 2020 pandemic year.
- **Self-report.** Most categorical features are respondent-reported, not measured. Equipment age in particular is bucketed into 6 bins.
- **Accuracy vs. interpretability in classification.** Multinomial logistic regression assumes log-odds are linear in the OHE features. The quick Gradient Boosting classifier benchmark improves weighted F1, but gives up the clean coefficient story.
- **High-bill regression outliers.** RMSE is sensitive to extreme annual bills. Gradient Boosting improves average performance but still underpredicts some high-`TOTALDOL` households.
- **Frozen-tertile issue.** Tertile cuts are sample-defined. A production deployment needs to freeze them for new-household scoring.

See notebook 05 for the full discussion.


## Future work

1. **Promote and tune the tree-based classifier.** A quick Gradient Boosting classifier improves held-out weighted F1 over Logistic Regression; a next iteration should tune it formally and update the report/slides if accuracy becomes the main objective.
2. **Calibration.** Plot reliability diagrams for the classifier; predicted probabilities `p_efficient`, `p_average`, `p_inefficient` are already saved in `models/logreg_test_predictions.csv`.
3. **Quantile regression.** RMSE is dominated by a few high-bill outliers. A quantile-loss model (or a Huber loss) would tell a different story about typical-case accuracy.
4. **Time-of-use & locational pricing.** RECS dollar amounts are annual averages. Tying to EIA Form 861 rates would expose state-level variability.
5. **Frozen-threshold productionization.** Persist the cost-per-sqft tertile cuts per climate so a new household can be scored without re-fitting the boundaries.


## References

- U.S. Energy Information Administration. *2020 Residential Energy Consumption Survey (RECS) Public Use Microdata.* Released 2023. https://www.eia.gov/consumption/residential/data/2020/


## Contributors

- Arya Kumar
- Krish Garg
