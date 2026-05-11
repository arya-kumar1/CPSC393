# Home Energy Copilot

**CPSC 393 Final Project — Spring 2026**

A three-layer machine-learning pipeline for residential energy efficiency, built on the U.S. Energy Information Administration's RECS 2020 microdata. The system classifies households as efficient / average / inefficient within their climate peer group, predicts annual energy expenditure, and produces ROI-aware retrofit recommendations.

---

## Motivation

Residential energy use accounts for roughly 20% of U.S. greenhouse-gas emissions. Existing energy-audit tools either require an in-person visit (expensive, slow) or rely on naive whole-home benchmarks that ignore climate, occupancy, and equipment context. *Home Energy Copilot* is a teaching example of how to stitch classification, regression, and rule-based recommendation into a single coherent product on a real, weighted survey dataset.

The pipeline answers three questions:

1. **Where does this household sit relative to its climate peers?** (classification)
2. **What annual energy bill should we expect?** (regression)
3. **Which retrofits offer the best simple-payback for this specific household?** (recommendation)

---

## Dataset

**Residential Energy Consumption Survey (RECS) 2020 — Public Use Microdata**, U.S. Energy Information Administration. 18,496 occupied primary residences × 799 columns. Each row carries `NWEIGHT`, a survey weight that scales the sample to roughly 123.5 million U.S. households.

> EIA. *Residential Energy Consumption Survey — 2020 Microdata.* https://www.eia.gov/consumption/residential/data/2020/index.php?view=microdata

The raw CSV (`recs2020_public_v7.csv`, ~56 MB) lives in `data/`. After cleaning, the working dataset is 18,495 × 69 (one row dropped for negative `TOTALDOL`; 65 features kept plus targets, weights, and IDs).

### Targets

- **Regression:** `TOTALDOL` — total annual energy expenditure in U.S. dollars.
- **Classification:** `efficiency_class` ∈ {efficient, average, inefficient}, defined as the within-climate-group **tertile** of `TOTALDOL / TOTSQFT_EN` (cost per square foot). Tertiles are computed inside each `BA_climate_grp` so that "efficient" means "low $/sqft *for your climate*", not "low $/sqft globally".

### Climate grouping

`BA_climate_grp` is the Building America climate zone with one merge: **Subarctic** (only 54 households) is folded into **Very-Cold** to keep tertile cuts statistically stable. The seven remaining peer groups are: Cold, Hot-Dry, Hot-Humid, Marine, Mixed-Dry, Mixed-Humid, Very-Cold.

### Sample weights

Every fit and every metric in this project uses `sample_weight = NWEIGHT` so results report at the U.S. residential population scale, not the sample scale.

---

## Repository layout

```
CPSC393Final/
├── README.md                       # this file
├── data/
│   ├── recs2020_public_v7.csv      # raw RECS 2020 microdata (input)
│   ├── recs2020_clean.pkl          # cleaned 18,495 × 69 DataFrame (notebook 01 output)
│   ├── recs2020_clean.parquet      # parquet copy (when pyarrow is installed)
│   ├── recs2020_clean_meta.json    # column inventory + sentinel scan
│   ├── train_test_split.json       # frozen 80/20 stratified split (DOEID lists)
│   └── upgrade_costs.csv           # 15-row retrofit catalog (DOE / ENERGY STAR)
├── notebooks/
│   ├── 01_data_loading.ipynb       # load, clean, derive end-use $, freeze targets
│   ├── 02_eda.ipynb                # weighted EDA, distributions, correlations
│   ├── 03_classification.ipynb     # multinomial logistic regression
│   ├── 04_regression.ipynb         # Elastic Net + Gradient Boosting on TOTALDOL
│   ├── 05_recommendations.ipynb    # rule-based retrofit recommender
│   └── 06_results_and_discussion.ipynb   # synthesis, cross-cutting analysis
└── models/
    ├── logreg_classification.joblib
    ├── logreg_test_predictions.csv
    ├── elasticnet_regression.joblib
    ├── gbr_regression.joblib
    ├── regression_test_predictions.csv
    ├── recommendations_top5.csv
    ├── recommendations_household_summary.csv
    └── recommendations_upgrade_frequency.csv
```

---

## How to run

### Environment

Python 3.10+ with:

```
pandas
numpy
scikit-learn>=1.5
matplotlib
seaborn
joblib
pyarrow            # optional, for parquet output
jupyter
```

A single `pip install pandas numpy scikit-learn matplotlib seaborn joblib pyarrow jupyter` covers it.

### Execution order

The notebooks are designed to run **in order**, sharing state through saved CSVs and joblib files. Re-running an upstream notebook invalidates downstream artifacts.

1. `01_data_loading.ipynb` — produces `data/recs2020_clean.pkl` and `data/recs2020_clean_meta.json`.
2. `02_eda.ipynb` — read-only inspection of the cleaned data (no artifacts saved beyond plots).
3. `03_classification.ipynb` — produces `data/train_test_split.json`, `models/logreg_classification.joblib`, `models/logreg_test_predictions.csv`.
4. `04_regression.ipynb` — produces `models/elasticnet_regression.joblib`, `models/gbr_regression.joblib`, `models/regression_test_predictions.csv`.
5. `05_recommendations.ipynb` — produces `models/recommendations_top5.csv`, `models/recommendations_household_summary.csv`, `models/recommendations_upgrade_frequency.csv`.
6. `06_results_and_discussion.ipynb` — read-only synthesis of everything above.

Notebook 04 grid-searches a Gradient Boosting model and is the slowest step (a few minutes on a laptop). Everything else is sub-minute.

---

## Pipeline architecture

### Layer 1 — Classification

**Model:** `LogisticRegression(solver='lbfgs', class_weight=None)` inside an `sklearn.Pipeline` with a `ColumnTransformer` (median-imputed + standard-scaled numeric features, one-hot-encoded categoricals). `GridSearchCV` over `C ∈ {0.01, 0.1, 1, 10}` with `scoring='f1_weighted'`.

**Features:** 55 hand-curated columns (29 numeric, 26 categorical). After OHE: ~216 features.

**Metrics:** weighted overall accuracy ≈ 60% (33% baseline), per-climate accuracy reported separately.

### Layer 2 — Regression

Two models compared head-to-head on the same train/test split:

- **`ElasticNet`** — combined L1/L2 regularization. `GridSearchCV` over `alpha ∈ {0.1, 1.0, 10, 100}` × `l1_ratio ∈ {0.1, 0.5, 0.9}`, `max_iter=20000`, scored by negative RMSE.
- **`GradientBoostingRegressor(max_depth=4)`** — captures nonlinear interactions. `GridSearchCV` over `n_estimators ∈ {200, 400}` × `learning_rate ∈ {0.05, 0.1}`.

**Metrics (weighted):** GBR achieves RMSE ≈ $745, R² ≈ 0.51, marginally better than Elastic Net on most climate groups.

The original proposal called for SVR; we replaced it with Gradient Boosting because SVR scales poorly past a few thousand rows and ~200 features.

### Layer 3 — Recommendations

A 15-row retrofit catalog (`data/upgrade_costs.csv`, sourced from DOE and ENERGY STAR) drives a rule-based recommender:

- **Eligibility:** `target_homes` rules (e.g. `EQUIPM IN (2;3;4) OR EQUIPAGE>=5`) are compiled into pandas masks via a tiny custom DSL — `IN`, `AND`, `OR`, scalar comparisons. Compilation uses regex transforms + sandboxed `eval` (no `__builtins__`).
- **Savings:** `annual_savings = savings_pct_mid × matching_end_use_dol`. Each retrofit references one of `heat_cool_dol`, `water_heat_dol`, `lighting_dol`, or `DOLLAREL` — chosen so insulation only saves heating dollars, LED only saves lighting dollars, etc.
- **Ranking:** simple payback in years, ties broken by larger absolute savings.
- **Output:** top-5 retrofits per household.

**Population result:** if every test household installed all five of its top-ranked retrofits, total annual savings would be ~$10.5 B — roughly 22% of the $47 B annual bill across the population represented by the test set.

---

## Key design choices

| Decision | Choice | Why |
|---|---|---|
| Classification target definition | tertile of `cost_per_sqft` within `BA_climate_grp` | Climate normalizes for unavoidable load; per-sqft normalizes for size. |
| Sample weighting | `NWEIGHT` for both fit and metrics | Reports population-scale results, not sample-scale. |
| Sentinel handling (`-2`, `-9`) | Kept as own category for OHE features; median-imputed for numeric features in the pipeline | Preserves "not applicable" signal for categoricals; keeps numeric scales sane. |
| Train/test split | 80/20 stratified on `efficiency_class × BA_climate_grp`, frozen as DOEID list | Same split shared across notebooks 03–05. |
| Regression model swap | `GradientBoostingRegressor` instead of SVR | SVR doesn't scale; GBR captures nonlinearities. |
| Recommendation savings basis | End-use specific (insulation hits `heat_cool_dol`, etc.) | More physically defensible than scaling `TOTALDOL`. |
| ROI metric | Simple payback years | Standard in DOE/ENERGY STAR reporting; intuitive; no discount-rate assumption. |
| Top-N | 5 per household | Balances signal density and UX. |

---

## Headline results

| Layer | Metric | Result | Baseline |
|---|---|---|---|
| Classification | Weighted accuracy | **60.4%** | 33.3% (random) |
| Regression (GBR) | Weighted RMSE | **$745** | std(TOTALDOL) ≈ $1,063 |
| Regression (GBR) | Weighted R² | **0.51** | — |
| Recommendation | Population potential savings | **$10.5 B / yr** | — |
| Recommendation | Share of total bill | **22.3%** | — |

A noteworthy cross-cutting finding: classifier-flagged *inefficient* households receive **smaller** absolute dollar savings ($407/yr) than classifier-flagged *efficient* households ($469/yr). This is a consequence of the cost-per-sqft target definition: small drafty homes flag inefficient even though their absolute end-use bills are modest. Notebook 06 discusses this honestly rather than papering over it.

---

## Limitations

- **RECS scope.** Cross-sectional, occupied primary U.S. residences only. Vacant homes, second homes, and territories are excluded. Survey was fielded during the 2020 pandemic year.
- **Self-report.** Most categorical features are respondent-reported, not measured. Equipment age in particular is bucketed into 6 bins.
- **Savings percentages from marketing sources.** DOE / ENERGY STAR figures may overstate field savings (the prebound/rebound effect).
- **No retrofit interactions.** The current ranker treats retrofits independently. Top-5 sums can over-count savings (insulation reduces heat-pump savings, etc.).
- **Frozen-tertile issue.** Tertile cuts are sample-defined. A production deployment needs to freeze them for new-household scoring.

See notebook 06 for the full discussion.

---

## Future work

1. Replace catalog savings with conditional-average-treatment-effect estimates from retrofit-program EM&V studies.
2. Joint per-household knapsack optimization that respects retrofit interactions.
3. Probability-weighted recommendations using classifier soft outputs.
4. Time-of-use and locational pricing via EIA Form 861 utility data.
5. Calibration diagnostics (reliability diagrams, residual heteroscedasticity).

---

## References

- U.S. Energy Information Administration. *2020 Residential Energy Consumption Survey (RECS) Public Use Microdata.* Released 2023. https://www.eia.gov/consumption/residential/data/2020/
- U.S. Department of Energy. *Energy Saver — Home Energy Audits and Retrofit Cost Reference.* https://www.energy.gov/energysaver
- ENERGY STAR. *Product Specifications and Estimated Annual Savings.* U.S. EPA. https://www.energystar.gov

---

*Built for CPSC 393 — Spring 2026.*
