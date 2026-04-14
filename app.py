import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Credit Risk Predictor")

# ─────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────
MODELS_DIR = "models"

@st.cache_resource
def load_artifacts():
    required = ["best_model.pkl", "scaler.pkl", "feature_columns.pkl", "metadata.pkl"]
    for f in required:
        if not os.path.exists(os.path.join(MODELS_DIR, f)):
            return None, f"Missing file: {f}"
    return {
        "model":    joblib.load(os.path.join(MODELS_DIR, "best_model.pkl")),
        "scaler":   joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
        "features": joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl")),
        "meta":     joblib.load(os.path.join(MODELS_DIR, "metadata.pkl")),
    }, None

artifacts, err = load_artifacts()

st.title("Credit Risk Predictor")

if err:
    st.error(err)
    st.stop()

meta = artifacts["meta"]
st.write(f"Model: {meta['best_model_name']}")

# ─────────────────────────────
# INPUT FORM
# ─────────────────────────────
st.header("Applicant Information")

person_age = st.number_input("Age", 18, 100, 30)
person_income = st.number_input("Annual Income", 0, 10_000_000, 55000)
person_emp_length = st.number_input("Employment Length (years)", 0, 60, 5)

person_home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

cb_person_default_on_file = st.selectbox(
    "Previous Default",
    ["N", "Y"]
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length (years)", 0, 60, 8
)

st.header("Loan Information")

loan_amnt = st.number_input("Loan Amount", 500, 500_000, 10000)
loan_intent = st.selectbox(
    "Loan Purpose",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

loan_int_rate = st.number_input("Interest Rate (%)", 1.0, 30.0, 11.0)

loan_percent_income = loan_amnt / max(person_income, 1)
st.write(f"Loan-to-Income Ratio: {loan_percent_income:.2f}")

predict_btn = st.button("Predict")

# ─────────────────────────────
# PREDICTION
# ─────────────────────────────
def run_prediction(raw):
    grade_map = meta.get("grade_map", {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7})
    features  = artifacts["features"]
    model     = artifacts["model"]
    scaler    = artifacts["scaler"]

    row = {col: 0 for col in features}

    scalar_fields = [
        "person_age", "person_income", "person_emp_length",
        "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length"
    ]

    for f in scalar_fields:
        if f in row:
            row[f] = raw[f]

    row["loan_grade"] = grade_map[raw["loan_grade"]]
    row["cb_person_default_on_file"] = 1 if raw["cb_person_default_on_file"] == "Y" else 0

    col_own = f"person_home_ownership_{raw['person_home_ownership']}"
    if col_own in row:
        row[col_own] = 1

    col_intent = f"loan_intent_{raw['loan_intent']}"
    if col_intent in row:
        row[col_intent] = 1

    X = pd.DataFrame([row])[features]

    if meta.get("needs_scaling", False):
        X = scaler.transform(X)
    else:
        X = X.values

    label = int(model.predict(X)[0])
    prob  = float(model.predict_proba(X)[0][1])

    return label, prob

# ─────────────────────────────
# RESULT
# ─────────────────────────────
if predict_btn:
    raw_input = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    }

    label, prob = run_prediction(raw_input)

    st.header("Result")

    if label == 0:
        st.success("Loan Approved")
    else:
        st.error("Loan Denied")

    st.write(f"Default probability: {prob:.2%}")

    st.subheader("Key Factors")

    st.write(f"- Loan-to-Income: {loan_percent_income:.2f}")
    st.write(f"- Loan Grade: {loan_grade}")
    st.write(f"- Interest Rate: {loan_int_rate}%")
    st.write(f"- Previous Default: {cb_person_default_on_file}")
    st.write(f"- Employment Length: {person_emp_length} years")
    st.write(f"- Credit History: {cb_person_cred_hist_length} years")