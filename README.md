# LoanPredict
A machine learning project that predicts whether a loan applicant is likely to repay a loan or default using a Decision Tree Classifier.

## Live demo
[Try LoanPredict](https://loanpredict-parth.streamlit.app/)

## Objective

The objective of this project is to develop a machine learning model
that predicts whether a loan applicant is likely to repay a loan or
default.

The project also aims to understand how decision-tree based models
make predictions based on financial and credit-related features.

## Dataset

The dataset contains 45,000 loan applicant records.

The target variable is:

- `loan_status`
  - `0` → Default
  - `1` → Repaid

The dataset contains financial, employment, housing, loan, and credit
history information.

## Machine Learning Approach

The following steps were performed:

1. Checked for duplicate records
2. Checked for missing values
3. Selected relevant features
4. Encoded categorical variables
5. Split the dataset into training and testing sets
6. Trained a Decision Tree Classifier
7. Evaluated the model on the test set

## Model

The project uses `DecisionTreeClassifier` from scikit-learn.

### Model Parameters

- Criterion: Entropy
- Max Depth: 5
- Random State: 42

## Project Structure

```text
loan-default-prediction/
│
├── bank loan data/
│   └── loan_data.csv
│
├── Loan analysis & model/
│   └── loan_prediction.ipynb
│   └── model.pkl
│
├── app.py
└── README.md
```

## Limitations

- The dataset is synthetic and may not represent real-world lending data.
- Decision-tree performance depends on the training data and selected parameters.
- Model predictions should not be treated as financial advice or as a real-world lending decision.
- The model's probability estimates may not be perfectly calibrated.
