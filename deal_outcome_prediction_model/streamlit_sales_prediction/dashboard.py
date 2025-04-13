import streamlit as st
import pandas as pd
import joblib
import numpy as np
from preprocessing import preprocess_data  # ✅ uses correct function name

st.set_page_config(page_title="Deal Outcome Predictor", layout="wide")

# === Load and display logo ===
from PIL import Image

# === Load and display SymTrain logo ===
logo = Image.open("symtrain-logo.png")
st.image(logo, width=180)  # Adjust width if needed


# === Load model and preprocessing assets ===
@st.cache_resource
def load_model_and_assets():
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoding_info = joblib.load("encoding_info.pkl")# match uploaded name
    return model, scaler, encoding_info

model, scaler, encoding_info = load_model_and_assets()

# === Streamlit App ===
st.title("🎯 Deal Outcome Prediction Dashboard")
st.markdown("""
Upload your deal data and this app will predict the likelihood of winning each deal based on our trained Random Forest model.
""")

# Upload file
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully.")
    input_df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Preview of Uploaded Data")
    st.dataframe(input_df.head())

    try:
        # === Preprocess data ===
        processed_df = preprocess_data(input_df, scaler, encoding_info)

        # === Run predictions ===
        predictions = model.predict(processed_df)
        prediction_proba = model.predict_proba(processed_df)[:, 1]

        # === Display predictions ===
        result_df = input_df.copy()
        result_df["Predicted Outcome"] = np.where(predictions == 1, "Won", "Lost")
        result_df["Win Probability"] = prediction_proba.round(3)

        st.write("### 🧠 Prediction Results")
        st.dataframe(result_df[["Predicted Outcome", "Win Probability"]])

        # === Show average probability and chart ===
        st.metric("📈 Average Win Probability", f"{prediction_proba.mean():.2%}")
        st.bar_chart(result_df["Win Probability"])

        # === Highlight the top deal ===
        top_deal_idx = result_df["Win Probability"].idxmax()
        top_deal_prob = result_df.loc[top_deal_idx, "Win Probability"]
        st.metric("🎯 Highest Win Probability", f"{top_deal_prob:.2%}", label_visibility="visible")

        st.write("🔍 Most Promising Deal Based on Prediction:")
        st.dataframe(result_df.loc[[top_deal_idx]].reset_index(drop=True))



        # === Downloadable CSV ===
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", csv, "deal_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error during preprocessing or prediction: {e}")

else:
    st.info(" Please upload a CSV file to begin.")
