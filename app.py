import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💰",
    layout="centered"
)


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("Load Analysis & Model/mymodel.pkl")


model = load_model()


# --------------------------------------------------
# 3. TITLE
# --------------------------------------------------

st.title("💰 Loan Default Predictor")

st.write(
    "Enter the applicant and loan information below "
    "to predict whether the loan is likely to be repaid or defaulted."
)


# --------------------------------------------------
# 4. INPUT SECTION
# --------------------------------------------------

st.subheader("Applicant Information")

person_income = st.number_input(
    "Annual Income ($)",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

person_emp_exp = st.number_input(
    "Employment Experience (years)",
    min_value=0,
    value=5,
    step=1
)

person_home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE"]
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length (years)",
    min_value=0.0,
    value=5.0,
    step=1.0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=0,
    max_value=1000,
    value=650,
    step=1
)

previous_loan_defaults_on_file = st.selectbox(
    "Previous Loan Default",
    ["No", "Yes"]
)


# --------------------------------------------------
# 5. LOAN INFORMATION
# --------------------------------------------------

st.subheader("Loan Information")

loan_amnt = st.number_input(
    "Loan Amount ($)",
    min_value=0.0,
    value=10000.0,
    step=500.0
)

loan_int_rate = st.number_input(
    "Loan Interest Rate (%)",
    min_value=0.0,
    value=10.0,
    step=0.1
)

loan_percent_income = st.number_input(
    "Loan Percent of Income",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)


# --------------------------------------------------
# 6. ENCODING
# --------------------------------------------------

# IMPORTANT:
# Replace these mappings with the EXACT mappings
# you used when training your model.

home_ownership_mapping = {
    "RENT": 0,
    "OWN": 1,
    "MORTGAGE": 2
}

previous_default_mapping = {
    "No": 0,
    "Yes": 1
}


encoded_home_ownership = home_ownership_mapping[
    person_home_ownership
]

encoded_previous_default = previous_default_mapping[
    previous_loan_defaults_on_file
]


# --------------------------------------------------
# 7. CREATE INPUT DATAFRAME
# --------------------------------------------------

input_data = pd.DataFrame({
    "person_income": [person_income],
    "person_emp_exp": [person_emp_exp],
    "person_home_ownership": [encoded_home_ownership],
    "loan_amnt": [loan_amnt],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "cb_person_cred_hist_length": [cb_person_cred_hist_length],
    "credit_score": [credit_score],
    "previous_loan_defaults_on_file": [encoded_previous_default]
})


# --------------------------------------------------
# 8. PREDICTION BUTTON
# --------------------------------------------------

if st.button("Predict Loan Status", use_container_width=True):

    prediction = model.predict(input_data)[0]

    st.divider()

    st.subheader("Prediction")

    if prediction == 1:

        st.success("✅ Likely to be Repaid")

    else:

        st.error("⚠️ Potential Default")


    # --------------------------------------------------
    # 9. PREDICTION PROBABILITY
    # --------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]

        default_probability = probabilities[0]
        repaid_probability = probabilities[1]

        st.write(
            f"Default probability: "
            f"**{default_probability * 100:.2f}%**"
        )

        st.write(
            f"Repaid probability: "
            f"**{repaid_probability * 100:.2f}%**"
        )


# --------------------------------------------------
# 10. OPTIONAL: SHOW INPUT DATA
# --------------------------------------------------

with st.expander("Show model input"):

    st.dataframe(input_data)
