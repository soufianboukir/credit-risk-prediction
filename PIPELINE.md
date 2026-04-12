# ML Project Pipeline — Data Cleaning to Deployment

> A structured, step-by-step reference for building production-grade machine learning systems.

---

## Table of Contents

1. [Data Cleaning](#1-data-cleaning)
2. [Exploratory Data Analysis](#2-exploratory-data-analysis)
3. [Feature Engineering](#3-feature-engineering)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Train / Validation / Test Split](#5-train--validation--test-split)
6. [Model Selection](#6-model-selection)
7. [Model Training](#7-model-training)
8. [Model Evaluation](#8-model-evaluation)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [Error Analysis](#10-error-analysis)
11. [Model Explainability](#11-model-explainability)
12. [Pipeline Serialization](#12-pipeline-serialization)
13. [API Development](#13-api-development)
14. [Testing](#14-testing)
15. [Deployment](#15-deployment)
16. [Monitoring](#16-monitoring)

---

## 1. Data Cleaning

The goal is to produce a dataset that is consistent, complete, and free of noise before any modeling begins.

### 1.1 Understand the Raw Data

```python
df.shape           # rows and columns
df.dtypes          # data types per column
df.head(10)        # first rows
df.describe()      # statistical summary
```

### 1.2 Handle Missing Values

Identify missing values first, then decide on a strategy per column.

```python
df.isnull().sum()
df.isnull().mean() * 100  # percentage missing per column
```

| Missing Rate | Strategy |
|---|---|
| < 5% | Impute (median / mode) |
| 5-30% | Impute with indicator flag |
| > 30% | Consider dropping the column |
| Structural (not random) | Investigate cause before imputing |

```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
```

### 1.3 Remove Duplicates

```python
df.duplicated().sum()
df.drop_duplicates(inplace=True)
```

### 1.4 Fix Data Types

```python
df["date_col"]     = pd.to_datetime(df["date_col"])
df["category_col"] = df["category_col"].astype("category")
df["income"]       = pd.to_numeric(df["income"], errors="coerce")
```

### 1.5 Handle Impossible Values

Domain-specific rules should be applied explicitly:

```python
df = df[df["age"] > 0]
df = df[df["income"] >= 0]
df = df[df["loan_amount"] > 0]
```

### 1.6 Handle Outliers

```python
Q1  = df["income"].quantile(0.25)
Q3  = df["income"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df["income"] = df["income"].clip(lower, upper)
```

---

## 2. Exploratory Data Analysis

EDA uncovers patterns, relationships, and problems in the data before modeling.

### 2.1 Target Variable Distribution

```python
df["target"].value_counts(normalize=True)
```

Check for class imbalance. If one class is under 20%, plan for imbalance handling later.

### 2.2 Feature Distributions

```python
import seaborn as sns
import matplotlib.pyplot as plt

df.hist(figsize=(16, 10), bins=30)
df["loan_intent"].value_counts().plot(kind="bar")
```

### 2.3 Correlation Analysis

```python
df.corr()["target"].sort_values()

sns.heatmap(df.corr(), annot=True, fmt=".2f")
```

### 2.4 Feature vs Target

```python
sns.boxplot(x="target", y="income", data=df)

pd.crosstab(df["home_ownership"], df["target"], normalize="index")
```

### 2.5 Key Questions to Answer

- Which features are most correlated with the target?
- Are there features with too many missing values to be useful?
- Is the target imbalanced?
- Are there collinear features that should be merged or dropped?

---

## 3. Feature Engineering

Transform raw features into signals that are more informative for the model.

### 3.1 Ratio Features

```python
df["debt_to_income"]    = df["debt"]            / df["income"]
df["loan_to_income"]    = df["loan_amount"]      / df["income"]
df["payment_to_income"] = df["monthly_payment"] / (df["income"] / 12)
```

### 3.2 Interaction Features

```python
df["income_x_emp_length"] = df["income"] * df["emp_length"]
```

### 3.3 Binning

```python
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 50, 100],
    labels=["young", "early_career", "mid_career", "senior"]
)
```

### 3.4 Date / Time Features

```python
df["account_age_days"] = (pd.Timestamp.today() - df["account_open_date"]).dt.days
df["loan_month"]       = df["loan_date"].dt.month
df["loan_year"]        = df["loan_date"].dt.year
```

### 3.5 Aggregation Features

```python
income_stats = df.groupby("home_ownership")["income"].agg(["mean", "std"])
df = df.merge(income_stats, on="home_ownership", suffixes=("", "_group"))
```

### 3.6 Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder

# Binary categories
df["default_history"] = LabelEncoder().fit_transform(df["default_history"])

# Multi-class: use OneHotEncoder inside the sklearn pipeline (see Section 4)
```

---

## 4. Preprocessing Pipeline

Build a reproducible, leak-free pipeline using `sklearn.Pipeline`.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numerical_features   = ["income", "loan_amount", "loan_int_rate", "emp_length"]
categorical_features = ["home_ownership", "loan_intent", "loan_grade"]

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline,   numerical_features),
    ("cat", categorical_pipeline, categorical_features),
])
```

Fit the pipeline only on training data. Transform validation and test sets using the fitted pipeline.

---

## 5. Train / Validation / Test Split

### Split Strategy

```python
from sklearn.model_selection import train_test_split

X = df.drop("target", axis=1)
y = df["target"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176, stratify=y_train_val, random_state=42
)
# 0.176 of 85% = 15% of total
```

| Set | Size | Purpose |
|---|---|---|
| Train | 70% | Fit the model |
| Validation | 15% | Tune hyperparameters |
| Test | 15% | Final unbiased evaluation |

### Class Imbalance Handling

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train_processed, y_train)
```

Apply SMOTE only on the training set. Never on validation or test.

---

## 6. Model Selection

Start simple, increase complexity only when needed.

| Stage | Model | When to Use |
|---|---|---|
| Baseline | Logistic Regression | Always start here |
| Intermediate | Random Forest | Non-linear patterns, interpretable |
| Advanced | XGBoost / LightGBM | Best tabular performance |
| Experimental | Neural Network | Large data, complex patterns |

Selection criteria: performance on validation set, training speed, interpretability requirements, and dataset size.

---

## 7. Model Training

### Full Pipeline

```python
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        eval_metric="auc",
        random_state=42,
    )),
])

model_pipeline.fit(X_train, y_train)
```

### Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model_pipeline, X_train, y_train,
    scoring="roc_auc", cv=cv
)

print(f"CV ROC-AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
```

---

## 8. Model Evaluation

### Core Metrics

```python
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

y_pred      = model_pipeline.predict(X_test)
y_pred_prob = model_pipeline.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
print(classification_report(y_test, y_pred))
```

### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"])
```

### Metrics Reference

| Metric | Use When |
|---|---|
| ROC-AUC | Imbalanced classes, comparing models |
| Precision | Cost of false positives is high |
| Recall | Cost of false negatives is high |
| F1-Score | Balance between precision and recall |
| Log Loss | Evaluating probability calibration |

---

## 9. Hyperparameter Tuning

### Option A: Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__n_estimators":  [100, 300, 500],
    "classifier__max_depth":     [3, 6, 9],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
}

grid_search = GridSearchCV(
    model_pipeline, param_grid,
    scoring="roc_auc", cv=5, n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```

### Option B: Optuna (Recommended)

```python
import optuna

def objective(trial):
    params = {
        "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
        "max_depth":     trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
    }
    model = XGBClassifier(**params, random_state=42)
    return cross_val_score(model, X_train_processed, y_train,
                           scoring="roc_auc", cv=5).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

---

## 10. Error Analysis

Understanding where the model fails is as important as knowing where it succeeds.

### Inspect False Negatives

```python
fn_mask  = (y_test == 1) & (y_pred == 0)
fn_cases = X_test[fn_mask]
fn_cases.describe()
```

Questions to answer:
- What income range is most often misclassified?
- What loan grade appears most in false negatives?
- Is there a demographic cluster being systematically missed?

### Threshold Adjustment

```python
threshold = 0.35  # lower = catch more defaults, higher = fewer false alarms

y_pred_adjusted = (y_pred_prob >= threshold).astype(int)
print(classification_report(y_test, y_pred_adjusted))
```

---

## 11. Model Explainability

### SHAP Values

```python
import shap

explainer   = shap.TreeExplainer(model_pipeline["classifier"])
X_processed = model_pipeline["preprocessor"].transform(X_test)
shap_values = explainer.shap_values(X_processed)

# Global importance
shap.summary_plot(shap_values, X_processed, feature_names=feature_names)

# Single prediction
shap.waterfall_plot(explainer(X_processed)[0])
```

### Built-in Feature Importance

```python
importances = model_pipeline["classifier"].feature_importances_

feat_df = pd.DataFrame({
    "feature":    feature_names,
    "importance": importances,
}).sort_values("importance", ascending=False)
```

---

## 12. Pipeline Serialization

Save the entire pipeline — preprocessor and model together — to ensure consistent inference.

```python
import joblib

joblib.dump(model_pipeline, "models/pipeline.pkl")

pipeline = joblib.load("models/pipeline.pkl")
pipeline.predict_proba(X_test[:5])
```

Save metadata alongside the model:

```python
import json

metadata = {
    "model_type":    "XGBClassifier",
    "roc_auc":       round(roc_auc_score(y_test, y_pred_prob), 4),
    "threshold":     0.35,
    "features":      list(X.columns),
    "training_date": str(pd.Timestamp.today().date()),
}

with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

---

## 13. API Development

### Project Structure

```
api/
├── main.py
├── schema.py
├── model.py
└── requirements.txt
```

### Schema

```python
# schema.py
from pydantic import BaseModel

class CustomerInput(BaseModel):
    age:            int
    income:         float
    emp_length:     float
    loan_amount:    float
    loan_int_rate:  float
    home_ownership: str
    loan_intent:    str
    loan_grade:     str

class PredictionOutput(BaseModel):
    risk_probability: float
    risk_level:       str
```

### Endpoint

```python
# main.py
from fastapi import FastAPI
import joblib, json
import pandas as pd
from schema import CustomerInput, PredictionOutput

app      = FastAPI(title="Credit Risk API")
pipeline = joblib.load("models/pipeline.pkl")
metadata = json.load(open("models/metadata.json"))

@app.post("/predict", response_model=PredictionOutput)
def predict(data: CustomerInput):
    df   = pd.DataFrame([data.dict()])
    prob = pipeline.predict_proba(df)[0][1]

    if prob < 0.3:
        level = "Low"
    elif prob < 0.6:
        level = "Medium"
    else:
        level = "High"

    return PredictionOutput(risk_probability=round(prob, 4), risk_level=level)

@app.get("/health")
def health():
    return {"status": "ok", "model": metadata["model_type"]}
```

```bash
uvicorn api.main:app --reload --port 8000
```

---

## 14. Testing

```python
# tests/test_predict.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_returns_valid_structure():
    payload = {
        "age": 35, "income": 8000, "emp_length": 5,
        "loan_amount": 10000, "loan_int_rate": 12.5,
        "home_ownership": "RENT", "loan_intent": "PERSONAL",
        "loan_grade": "C"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert 0 <= data["risk_probability"] <= 1

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

```bash
pytest tests/ -v
```

---

## 15. Deployment

### Option A: Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

### Option B: Render / Railway

1. Push code to GitHub
2. Connect repository to Render or Railway
3. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Set environment variables if needed
5. Deploy

### Option C: Cloud (AWS / GCP / Azure)

```
1. Containerize with Docker
2. Push image to ECR / Artifact Registry / ACR
3. Deploy via ECS / Cloud Run / App Service
4. Attach load balancer and configure auto-scaling
```

---

## 16. Monitoring

### What to Monitor

| Signal | Tool | Action Trigger |
|---|---|---|
| Prediction distribution shift | Custom logging | Alert if mean probability drifts > 10% |
| Feature distribution (data drift) | Evidently AI | Retrain if PSI > 0.2 |
| API latency and error rate | Prometheus / Grafana | Alert on p95 > 200ms |
| Model performance | MLflow | Retrain if ROC-AUC drops > 5% |

### Logging Predictions

```python
import logging

logging.basicConfig(filename="logs/predictions.log", level=logging.INFO)

@app.post("/predict", response_model=PredictionOutput)
def predict(data: CustomerInput):
    prob  = pipeline.predict_proba(pd.DataFrame([data.dict()]))[0][1]
    level = "Low" if prob < 0.3 else "Medium" if prob < 0.6 else "High"

    logging.info({"input": data.dict(), "probability": prob, "risk_level": level})

    return PredictionOutput(risk_probability=round(prob, 4), risk_level=level)
```

### Retraining Triggers

- Scheduled — monthly or quarterly cadence
- Performance-based — metric drops below a defined threshold
- Data-based — significant distribution shift detected in production

---

## Summary

```
Raw Data
  -> Data Cleaning           consistent, complete, valid
  -> EDA                     understand patterns and risks
  -> Feature Engineering     create meaningful signals
  -> Preprocessing Pipeline  reproducible transformations
  -> Train / Val / Test      stratified, no leakage
  -> Model Selection         start simple
  -> Training                full pipeline fit
  -> Evaluation              AUC, precision, recall, F1
  -> Hyperparameter Tuning   Optuna or GridSearch
  -> Error Analysis          understand failures
  -> Explainability          SHAP values
  -> Serialization           save pipeline and metadata
  -> API                     FastAPI endpoint
  -> Testing                 unit and integration
  -> Deployment              Docker or cloud
  -> Monitoring              drift, latency, performance
```