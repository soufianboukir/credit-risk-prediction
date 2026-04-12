# Feature Reference — Credit Risk Dataset

> **Core formula:** `Risk = f(capacity, behavior, history, stability)`

---

## Features Overview

| # | Feature | Type | Risk Direction |
|---|---|---|---|
| 1 | `person_age` | Demographic | Very young or very old = higher risk |
| 2 | `person_income` | Financial | Lower income = higher risk |
| 3 | `person_home_ownership` | Stability | RENT > MORTGAGE > OWN (risk order) |
| 4 | `person_emp_length` | Stability | Shorter employment = higher risk |
| 5 | `loan_intent` | Behavioral | BUSINESS / MEDICAL riskier than EDUCATION |
| 6 | `loan_grade` | Pre-scored | A (safe) → G (risky) |
| 7 | `loan_amnt` | Financial | Larger loan = harder to repay |
| 8 | `loan_int_rate` | Financial | High rate = risky customer signal |
| 9 | `loan_status` | **Target** | `0` = No default · `1` = Default |
| 10 | `loan_percent_income` | Ratio | Closer to 1 = riskier |
| 11 | `cb_person_default_on_file` | History | Past default = strong risk predictor |
| 12 | `cb_person_cred_hist_length` | History | Shorter history = unknown risk |

---

## Feature Details

### 1. `person_age`
Customer's age in years. Risk peaks at extremes — young customers lack financial history; older customers may face income uncertainty near retirement.

### 2. `person_income`
Annual income. Strong **inverse** relationship with default — higher income means better repayment capacity.

### 3. `person_home_ownership`
Proxy for wealth and financial stability.

```
OWN → most stable
MORTGAGE → moderately stable
RENT → least stable
```

### 4. `person_emp_length`
Years of continuous employment. Longer tenure signals stable, reliable income.

### 5. `loan_intent`
The declared purpose of the loan. Adds **behavioral context** about urgency and risk tolerance.

```
BUSINESS  → high risk (uncertain return)
MEDICAL   → high risk (emergency-driven)
EDUCATION → moderate
PERSONAL  → variable
```

### 6. `loan_grade`
A lender-assigned risk score from `A` (safest) to `G` (riskiest). One of the **strongest predictive features** — it already encodes expert risk assessment.

### 7. `loan_amnt`
Total amount borrowed. Risk scales with size, but must be read alongside income.

### 8. `loan_int_rate`
Interest rate on the loan. High rates are both a **cause** and a **symptom** of risk — lenders charge more to risky customers, and high payments make default more likely.

### 9. `loan_status` — Target Variable
```
0 → No default
1 → Default
```
This is what you predict. Not a feature.

### 10. `loan_percent_income`
```python
loan_percent_income = loan_amount / income
```
Measures the **debt burden** relative to income. One of the most important features.

```
0.1 → safe  (loan is small vs income)
0.8 → risky (loan is large vs income)
```

### 11. `cb_person_default_on_file`
Binary past default history (`Y` / `N`). Past behavior is the strongest behavioral predictor of future default.

### 12. `cb_person_cred_hist_length`
Length of credit history in years. Longer history = more data = more trust.

---

## ⚡ Feature Importance Ranking

```
1. loan_percent_income       ████████████  ← most important
2. person_income             ██████████
3. loan_int_rate             █████████
4. cb_person_default_on_file ████████
5. person_emp_length         ██████
```

---

## Key Interaction: Context Matters

The same loan amount carries different risk depending on income:

| Income | Loan | `loan_percent_income` | Risk |
|---|---|---|---|
| $10,000 | $5,000 | 0.50 | Moderate |
| $2,000 | $5,000 | 2.50 | High |

---

## Mental Model

Each feature answers one core question:

| Question | Features |
|---|---|
| Can they pay? | `person_income`, `loan_percent_income`, `loan_amnt` |
| How hard is it to pay? | `loan_int_rate`, `loan_grade` |
| Do they usually pay? | `cb_person_default_on_file`, `cb_person_cred_hist_length` |
| Will income continue? | `person_emp_length`, `person_home_ownership` |
| Why are they borrowing? | `loan_intent` |