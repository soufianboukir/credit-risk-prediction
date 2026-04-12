# Credit Risk Prediction System — End-to-End Guide

> **Goal:** Build a system that predicts whether a customer will default on a loan, outputting a probability of default and a risk class (Low / Medium / High).

---

## Table of Contents

1. [Problem Understanding](#1-problem-understanding)
2. [Data Collection](#2-data-collection)
3. [Project Structure](#3-project-structure)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis-eda)
5. [Data Cleaning](#5-data-cleaning)
6. [Feature Engineering](#6-feature-engineering)
7. [Train / Test Split](#7-train--test-split)
8. [Model Training](#8-model-training)
9. [Model Evaluation](#9-model-evaluation)
10. [Model Comparison](#10-model-comparison)
11. [Error Analysis](#11-error-analysis)
12. [Feature Importance](#12-feature-importance)
13. [Final Model Pipeline](#13-final-model-pipeline)
14. [Save Model](#14-save-model)
15. [Build API](#15-build-api)
16. [Frontend](#16-simple-frontend)
17. [Deployment](#17-deployment-options)
18. [Documentation](#18-final-documentation)
19. [Bonus](#19-bonus--stand-out-additions)

---

## 1. Problem Understanding

| Element | Details |
|---|---|
| **Input** | Customer financial & demographic data |
| **Output** | Default risk — `0/1` or probability |
| **False Negative** |  Dangerous — approves a risky customer |
| **False Positive** |  Lost opportunity — rejects a safe customer |

> This cost asymmetry will directly guide your choice of evaluation metric.

---

## 2. Data Collection

**Recommended Kaggle datasets:**

- [`Home Credit Default Risk`](https://www.kaggle.com/c/home-credit-default-risk)
- [`Give Me Some Credit`](https://www.kaggle.com/c/GiveMeSomeCredit)

**What you'll find:**

- Customer info (age, employment, income)
- Loan history & debt
- Target column: `default` (0 or 1)

---

## 3. Project Structure

```
credit-risk-project/
│
├── data/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
├── api/
├── reports/
└── README.md
```

---

## 4. Exploratory Data Analysis (EDA)

**Tasks:**

- Check missing values
- Understand feature distributions
- Detect outliers
- Check class imbalance *(critical for credit risk)*

**Key checks:**

- % of defaults in the dataset
- Correlation of features with the target
- Income distribution
- Debt ratio patterns

**Tools:** `pandas`, `seaborn`, `matplotlib`

---

## 5. Data Cleaning

| Issue | Treatment |
|---|---|
| Missing numerical values | Replace with **median** |
| Missing categorical values | Replace with **mode** or `"Unknown"` |
| Duplicates | Remove |
| Impossible values (`age < 0`, `income = 0`) | Remove or correct |
| Outliers | Cap with **winsorization** |

---

## 6. Feature Engineering

> This is where your project becomes industry-level.

### Financial Ratios
```python
debt_to_income  = debt / income
loan_to_income  = loan_amount / income
```

### Behavioral Features
- Number of past defaults
- Number of open accounts

### Time-Based Features
- Employment duration
- Credit history length

### Encoding & Scaling

| Type | Method |
|---|---|
| Categorical (multi-class) | One-hot encoding |
| Categorical (binary) | Label encoding |
| Numerical | `StandardScaler` |

---

## 7. Train / Test Split

| Set | Size |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

> Always use **stratified split** to preserve class balance across sets.

---

## 8. Model Training

Train and compare the following models:

| Step | Model |
|---|---|
| Baseline | Logistic Regression |
| Tree-based | Random Forest |
| Boosting *(best performance)* | XGBoost, LightGBM |

---

## 9. Model Evaluation

> Do **not** use accuracy — data is imbalanced.

**Core metrics:**

| Metric | Why It Matters |
|---|---|
| **ROC-AUC** *(most important)* | Measures ranking ability across all thresholds |
| **Precision** | Of predicted defaults, how many were real? |
| **Recall** | Of real defaults, how many did we catch? |
| **F1-Score** | Balance between precision and recall |

---

## 10. Model Comparison

| Model | ROC-AUC | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Random Forest | | | | |
| XGBoost | | | | |
| LightGBM | | | | |

> Select the best model based on **business cost**, not just the highest metric.

---

## 11. Error Analysis

| Error Type | Meaning | Impact |
|---|---|---|
| **False Negative** | Predicted safe, actually defaults | Dangerous — financial loss |
| **False Positive** | Predicted risky, actually safe | Lost revenue opportunity |

**What to investigate:**

- Which features are most common in misclassified cases?
- Are certain customer segments systematically misclassified?

---

## 12. Feature Importance

For tree-based models:

- **Importance scores** — built-in from `RandomForest`, `XGBoost`, `LightGBM`
- **SHAP values** *(advanced)* — model-agnostic explanations

> Answer: *"What actually drives loan default?"*

---

## 13. Final Model Pipeline

```python
def predict(customer_data):
    # 1. Clean data
    # 2. Apply feature transformations
    # 3. Load trained model
    # 4. Return probability + risk level
    return {
        "risk_probability": ...,
        "risk_level": "Low" | "Medium" | "High"
    }
```

---

## 14. Save Model

Save both the model and the preprocessing pipeline:

```python
import joblib

joblib.dump(model,    "models/model.pkl")
joblib.dump(pipeline, "models/pipeline.pkl")
```

**Tools:** `joblib` or `pickle`

---

## 15. Build API

**Framework:** FastAPI

### Endpoint: `POST /predict`

**Request:**
```json
{
  "income": 5000,
  "loan": 20000,
  "age": 35
}
```

**Response:**
```json
{
  "risk_probability": 0.78,
  "risk_level": "High"
}
```

---

## 16. Simple Frontend

**Options:**

- **Streamlit** — fast, Python-native dashboard
- **HTML/CSS** — lightweight custom form

**Features to include:**

- Customer input form
- Real-time prediction display
- Risk level badge (Low / Medium / High)

---

## 17. Deployment Options

| Option | Tool | Level |
|---|---|---|
| **A — Easy** | Streamlit Cloud | Beginner |
| **B — Professional** | Render / Railway / AWS | Intermediate |
| **C — Advanced** | Docker container | Advanced |

---

## 18. Final Documentation (README)

Your README must include:

- [ ] Problem description
- [ ] Dataset source & description
- [ ] Pipeline steps (with diagram if possible)
- [ ] Models used & results
- [ ] Model comparison table
- [ ] How to run the project locally
- [ ] API usage examples

---

## 19. Bonus — Stand-Out Additions

| Addition | Tool / Technique |
|---|---|
| Handle class imbalance | SMOTE (`imbalanced-learn`) |
| Hyperparameter tuning | `GridSearchCV` / `Optuna` |
| Explainability dashboard | SHAP + Streamlit |
| Logging system | Python `logging` / MLflow |

---

## Final Mental Model

> You are not building a notebook.
> You are building a **real banking risk system** used to decide who gets loans — a production ML system.

---

*Next steps available: dataset + feature list · step-by-step coding plan · GitHub repo template · FastAPI + Streamlit deployment guide*