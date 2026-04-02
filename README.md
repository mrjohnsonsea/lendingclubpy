# LendingClub Interest Rate Prediction

Predicting loan interest rates from borrower and loan origination characteristics using Random Forest and Neural Network regression models on 100,000 LendingClub loans (2007–2020).

**Course**: Georgetown University MSBA — OPAN 6604: Predictive Analytics

---

## Business Problem

LendingClub's peer-to-peer lending platform matches borrowers with investors by assigning interest rates that reflect credit risk. Accurately predicting those rates from origination-time data supports better credit risk modeling, loan pricing, and investor decision-making — and lets us understand *which borrower characteristics drive the cost of credit*.

---

## Dataset

| Property | Value |
|---|---|
| Source | LendingClub public loan data |
| Period | 2007–2020 |
| Records | 100,000 loans |
| Features | 23 predictors + 1 target |
| Target | `int_rate` — annualized interest rate (%) |

All features are **loan origination variables only** (no post-origination performance data), making this a realistic forward-looking prediction task. Feature definitions are in [`docs/LCDataDictionary.xlsx`](docs/LCDataDictionary.xlsx).

Key predictors include FICO credit score range, debt-to-income ratio, loan amount, loan term, employment length, home ownership, income verification status, and credit history variables.

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution analysis with histogram plots for all numeric features
- Pairplot and correlation matrix to identify feature relationships
- Box plots of categorical features vs. interest rate
- FICO score non-linearity analysis (rates flatten above 825)

### 2. Preprocessing
- **Missing values**: Median/zero imputation with missing indicator variables for features with informative missingness; dropped `mths_since_last_record` (>10% missing)
- **Outliers**: 99th-percentile thresholds applied to heavily right-skewed features
- **Transformations**: Square root applied to `annual_inc` and `total_bal_ex_mort` to reduce skew
- **Encoding**: One-hot encoding for nominal categoricals; ordinal encoding for `emp_length`
- **Train/Dev/Test split**: 70/15/15 with stratification on binned `int_rate`
- **Scaling**: Min-Max normalization (fitted on training set only) for neural network input

### 3. Modeling

| Model | Test RMSE |
|---|---|
| Random Forest (tuned) | **4.29** |
| Neural Network (MLP) | 4.50 |

- **Random Forest**: Hyperparameters tuned via random search over `n_estimators` (50–1000) and `max_features` (1–8)
- **Neural Network**: Grid search over hidden layer architectures and activation functions (`tanh`, `relu`); up to 1500 training iterations

### 4. Interpretability
- **Permutation Feature Importance** — global view of which features drive predictions across the full test set
- **LIME** — local explanations for individual loan predictions, showing how each feature pushes the rate up or down for a specific applicant

---

## Key Findings

1. **FICO score is the dominant predictor** (~47% of permutation importance). Applicants with higher credit scores receive meaningfully lower interest rates.
2. **DTI and loan term are the next most important** (~16-17% each). Higher debt-to-income ratios and 60-month terms both push rates up.
3. **FICO score has a non-linear ceiling effect**: Rates stop declining above a FICO score of approximately 825.
4. **Random Forest substantially outperforms MLP** (RMSE 4.29 vs. 4.5), suggesting tree-based methods better capture the interaction effects in this tabular credit dataset.
5. **Feature engineering matters**: Missing indicator variables and transformations for skewed features meaningfully improved model performance.

---

## Project Structure

```
lendingclubpy/
├── data/
│   ├── raw/                    # LC_HW2.csv — 100K loan records (source data)
│   ├── interim/                # Intermediate preprocessing outputs
│   └── processed/              # Final cleaned and encoded dataset
├── docs/
│   └── LCDataDictionary.xlsx   # Definitions for all 24 features
├── models/                     # Saved model artifacts
├── notebooks/
│   └── lendingclub_interest_rate_prediction.ipynb  # Full analysis pipeline
├── references/                 # Supporting reference materials
├── reports/
│   ├── figures/                # 33 visualization outputs (PNG)
│   ├── project_2_final.html    # Rendered HTML report
│   └── project_2_final.pdf     # PDF report
└── README.md
```

---

## Technologies

| Category | Libraries / Tools |
|---|---|
| Data manipulation | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Modeling | scikit-learn (`RandomForestRegressor`, `MLPRegressor`) |
| Interpretability | LIME (`lime`) |
| Environment | Python 3, Jupyter Notebook |

---

## Running the Analysis

1. Clone the repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn lime openpyxl`
3. Place the raw data file at `data/raw/LC_HW2.csv`
4. Open and run [`notebooks/lendingclub_interest_rate_prediction.ipynb`](notebooks/lendingclub_interest_rate_prediction.ipynb) top to bottom
