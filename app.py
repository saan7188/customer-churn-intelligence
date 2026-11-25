import pandas as pd
import streamlit as st
from churn.data import sample_data, validate
from churn.model import train

st.title("Customer churn intelligence")
file = st.file_uploader("Upload CSV", type=["csv"])
algorithm = st.selectbox("Algorithm", ["logistic", "forest"])
try:
    data = validate(pd.read_csv(file) if file else sample_data())
    if not file:
        st.info("Using synthetic demonstration data.")
    st.write(data.head())
    st.bar_chart(data.groupby("contract").churn.mean())
    if st.button("Train model"):
        st.session_state["result"] = train(data, algorithm)
    if "result" in st.session_state:
        model, metrics, matrix = st.session_state["result"]
        st.write(pd.DataFrame([metrics]))
        st.write("Confusion matrix (actual rows, predicted columns)", matrix)
        tenure = st.number_input("Tenure", 1, 100, 12)
        charge = st.number_input("Monthly charge", 0.0, 1000.0, 60.0)
        calls = st.number_input("Support calls", 0, 100, 2)
        contract = st.selectbox("Contract", ["monthly", "annual"])
        row = pd.DataFrame([dict(tenure=tenure, monthly_charge=charge, support_calls=calls, contract=contract)])
        st.metric("Estimated churn probability", f"{model.predict_proba(row)[0,1]:.1%}")
except Exception as exc:
    st.error(str(exc))
