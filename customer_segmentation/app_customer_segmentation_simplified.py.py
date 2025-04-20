#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# === SymTrain logo in sidebar ===
with st.sidebar:
    logo_url = "https://raw.githubusercontent.com/marymorkos/SalesPlaybookDS5640/main/customer_segmentation/LOGO_white_greenWTagline-1.png"
    try:
        logo_img = Image.open(BytesIO(requests.get(logo_url).content))
        st.image(logo_img, width=200)
    except:
        st.warning("⚠️ Logo image could not be loaded.")
    st.markdown("#### SymTrain Sales Playbook")
    st.markdown("---")

# === Load model + scaler from GitHub ===
@st.cache_resource
def load_models():
    scaler_url = "https://raw.githubusercontent.com/marymorkos/SalesPlaybookDS5640/main/customer_segmentation/models/minmax_scaler.joblib"
    model_url = "https://raw.githubusercontent.com/marymorkos/SalesPlaybookDS5640/main/customer_segmentation/models/kmeans_model.joblib"
    scaler = joblib.load(BytesIO(requests.get(scaler_url).content))
    model = joblib.load(BytesIO(requests.get(model_url).content))
    return scaler, model

scaler, model = load_models()

# === Sidebar navigation ===
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to", [
    "Customer Segmentation",
    "Deal Outcome Predictor",
    "Pipeline ETL Overview",
    "Team Insights"
])

if page == "Customer Segmentation":
    st.title("🔍 Customer Segmentation Predictor")
    st.write("### Enter customer information or upload a CSV")

    with st.expander("🧾 Enter One Customer's Info"):
        annual_revenue = st.number_input("Annual Revenue", min_value=0.0, value=500000.0)
        form_submissions = st.number_input("Number of Form Submissions", min_value=0, value=0)
        times_contacted = st.number_input("Number of times contacted", min_value=0, value=0)
        pageviews = st.number_input("Number of Pageviews", min_value=0, value=100)
        company_age = st.number_input("Company Age", min_value=0, value=10)
        employees = st.number_input("Number of Employees", min_value=0, value=100)
        sessions = st.number_input("Number of Sessions", min_value=0, value=50)

    predict = st.button("Predict Segment")

    if predict:
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
            0: "Segment 0: Small, high-engagement companies",
            1: "Segment 1: Large enterprises with moderate traffic",
            2: "Segment 2: High-revenue, low-engagement firms",
            3: "Segment 3: Medium-sized firms, high traffic",
            4: "Segment 4: Low-revenue, low-activity customers"
        }
        st.markdown(f"**Interpretation:** {segment_meanings.get(segment, 'Unknown')}")

        with st.expander("📈 Cluster Visualization"):
            try:
                pca = PCA(n_components=2)
                pca_centroids = pca.fit_transform(model.cluster_centers_)
                input_pca = pca.transform(scaled_input)

                fig, ax = plt.subplots()
                ax.scatter(pca_centroids[:, 0], pca_centroids[:, 1], c='gray', label='Centroids')
                ax.scatter(input_pca[:, 0], input_pca[:, 1], c='red', marker='X', s=100, label='Input')
                ax.legend()
                ax.set_title("PCA Projection of Customer Segment")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Unable to generate visualization: {e}")

    st.write("### 📥 Batch Prediction from CSV")
    uploaded = st.file_uploader("Upload CSV with 7 numeric columns", type=["csv"])
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        try:
            X_scaled = scaler.transform(df_batch)
            preds = model.predict(X_scaled)
            df_batch["Predicted Segment"] = preds
            st.dataframe(df_batch)

            csv = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button("📤 Download Predictions", csv, "segmented_customers.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.caption("🧾 Model trained on 7 numeric features from GitHub-cleaned dataset.")
    st.caption("[View Training Notebook](https://github.com/marymorkos/SalesPlaybookDS5640/blob/main/customer_segmentation/customer_segmentation_for_streamlit%20(1).ipynb)")

else:
    st.title(f"{page} (Coming Soon)")
    st.info("This page is reserved for integration with your teammates' modules.")


# In[ ]:




