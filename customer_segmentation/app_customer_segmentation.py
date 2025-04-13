
import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

st.set_page_config(page_title="Customer Segmentation Predictor", layout="centered")
st.title("🔍 Customer Segmentation Predictor")

# === Load model + scaler from GitHub ===
@st.cache_resource
def load_models():
    scaler_url = "https://raw.githubusercontent.com/marymorkos/SalesPlaybookDS5640/main/customer_segmentation/models/minmax_scaler.pkl"
    model_url = "https://raw.githubusercontent.com/marymorkos/SalesPlaybookDS5640/main/customer_segmentation/models/kmeans_model.pkl"
    scaler = joblib.load(BytesIO(requests.get(scaler_url).content))
    model = joblib.load(BytesIO(requests.get(model_url).content))
    return scaler, model

scaler, model = load_models()

# === Input form ===
st.write("Enter customer information:")

annual_revenue = st.number_input("Annual Revenue", min_value=0.0, value=500000.0)
form_submissions = st.number_input("Number of Form Submissions", min_value=0, value=0)
times_contacted = st.number_input("Number of times contacted", min_value=0, value=0)
pageviews = st.number_input("Number of Pageviews", min_value=0, value=100)
company_age = st.number_input("Company Age", min_value=0, value=10)
employees = st.number_input("Number of Employees", min_value=0, value=100)
sessions = st.number_input("Number of Sessions", min_value=0, value=50)

# === Predict segment ===
if st.button("Predict Segment"):
    input_df = pd.DataFrame({
        "Annual Revenue": [annual_revenue],
        "Number of Form Submissions": [form_submissions],
        "Number of times contacted": [times_contacted],
        "Number of Pageviews": [pageviews],
        "Company_Age": [company_age],
        "Number of Employees": [employees],
        "Number of Sessions": [sessions]
    })

    scaled_input = scaler.transform(input_df)
    segment = model.predict(scaled_input)[0]

    st.success(f"🎯 Predicted Customer Segment: {segment}")
    segment_meanings = {
        0: "🟢 Segment 0: Small, high-engagement companies",
        1: "🔵 Segment 1: Large enterprises with moderate traffic",
        2: "🟡 Segment 2: High-revenue, low-engagement firms",
        3: "🟠 Segment 3: Medium-sized firms, high traffic",
        4: "🔴 Segment 4: Low-revenue, low-activity customers"
    }
    st.markdown(f"**Interpretation:** {segment_meanings.get(segment, 'Unknown')}")
