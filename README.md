# Credit Risk Prediction System

A complete machine learning pipeline for predicting loan default risk using tabular financial data.  
The project covers data preprocessing, model training, evaluation, and deployment through a simple Streamlit application.

<img width="2048" height="902" alt="Screenshot from 2026-04-28 23-49-19" src="https://github.com/user-attachments/assets/c492cf12-3a40-489c-87b8-70ddcb355104" />

---

## Project Overview

The goal is to classify whether a loan applicant is likely to **default (1)** or **not default (0)** based on financial and personal attributes.

The system includes:
- Data cleaning and feature engineering
- Multiple ML models training and evaluation
- Model comparison and selection
- Deployment via a Streamlit web app

---

## Dataset

- Total samples: **28,800**
- Features after encoding: **18**
- Train/Test split:
  - Train: 23,040
  - Test: 5,760
- Default rate: ~22%

### Features include:
- Personal: age, income, employment length
- Loan: amount, interest rate, grade, intent
- Credit history: past defaults, history length
- Derived: loan-to-income ratio

---

## Models Trained

The following models were implemented and evaluated:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree
- Random Forest

---

## Model Performance (Test Set)

| Model                  | Accuracy | ROC-AUC | Precision | Recall | F1 Score |
|-----------------------|----------|--------|----------|--------|---------|
| Random Forest         | 0.9109   | 0.9239 | 0.8412   | 0.7328 | 0.7833  |
| SVM                   | 0.8686   | 0.8991 | 0.6835   | 0.7478 | 0.7142  |
| Decision Tree         | 0.8870   | 0.8944 | 0.7452   | 0.7375 | 0.7414  |
| Logistic Regression   | 0.7811   | 0.8558 | 0.5011   | 0.7542 | 0.6021  |
| KNN                   | 0.8825   | 0.8435 | 0.8403   | 0.5739 | 0.6820  |
| Naive Bayes           | 0.7970   | 0.8236 | 0.5308   | 0.6530 | 0.5856  |

### Best Model
- **Random Forest**
- ROC-AUC: **0.9239**

---

## Project Structure
```
├── data/
│ ├── credit_risk_dataset.csv
│ ├── credit_risk_dataset_clean.csv
│ └── FEATURES.md
│
├── models/
│ ├── best_model.pkl
│ ├── random_forest.pkl
│ ├── decision_tree.pkl
│ ├── svm.pkl
│ ├── logistic_regression.pkl
│ ├── k-nearest_neighbors.pkl
│ ├── naive_bayes.pkl
│ ├── scaler.pkl
│ ├── feature_columns.pkl
│ ├── metadata.pkl
│ └── model_results.csv
│
├── notebooks/
│ ├── 01_cleaning.ipynb
│ ├── 02_analysis.ipynb
│ └── 03_training.ipynb
│
├── app.py
├── requirements.txt
├── README.md
├── GUIDE.md
└── PIPELINE.md
```



---

## Pipeline

1. Data Cleaning
2. Feature Engineering
   - Encoding categorical variables
   - Creating loan-to-income ratio
3. Train/Test split
4. Model training
5. Evaluation using:
   - Accuracy
   - Precision / Recall
   - F1 Score
   - ROC-AUC
6. Model selection
7. Saving artifacts

---

## Saved Artifacts

- `best_model.pkl` → selected model (Random Forest)
- `scaler.pkl` → feature scaler
- `feature_columns.pkl` → ordered feature list
- `metadata.pkl` → model configuration
- `model_results.csv` → comparison results

---

## Running the App

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All notebooks inside `notebooks/` folder

### 3. a folder named `/models` will be generated at `/notebooks`, Move it to root directory

### 4. Run this command and copy past the output ip address into your browser

```bash
streamlit run app.py
```

### Usage

- Enter applicant details
- Click "Predict"
- Get:
```
Loan decision (Approved / Denied)
Default probability
Key contributing factors
```

developed with ❤️ by **soufian**.
