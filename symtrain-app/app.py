import streamlit as st
import pandas as pd
import joblib
import numpy as np            
import seaborn as sns
from io import BytesIO
import requests
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === App Config ===
st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")

# === Load Logo ===

logo_url = "https://raw.githubusercontent.com/marymorkos/SalesPlaybookDS5640/main/customer_segmentation/LOGO_white_greenWTagline-1.png"
logo_img = Image.open(BytesIO(requests.get(logo_url).content))
#logo = Image.open("/Users/camdenbibro/Documents/03_machine_learning/SalesPlaybookDS5640-main/deal_outcome_prediction_model/streamlit_sales_prediction/symtrain-logo.png")
st.sidebar.image(logo_img, width=180)
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", [
    "Deal Outcome Predictor",
    "Customer Segmentation",
    "Sales Intelligence Dashboard"
])

# === LOAD MODELS AND ASSETS ===
@st.cache_resource
def load_preprocessing_module():
    # Get the preprocessing.py module from GitHub
    preprocessing_url = "https://raw.githubusercontent.com/marymorkos/SalesPlaybookDS5640/main/customer_segmentation/utils/preprocessing.py"
    preprocessing_content = requests.get(preprocessing_url).text
    
    # Create a module from the content
    spec = importlib.util.spec_from_loader(
        "preprocessing", 
        loader=None, 
        origin=preprocessing_url
    )
    preprocessing = importlib.util.module_from_spec(spec)
    
    # Execute the module code
    exec(preprocessing_content, preprocessing.__dict__)
    
    return preprocessing

@st.cache_resource
def load_deal_model_assets():
    model = joblib.load("models/random_forest_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoding_info = joblib.load("models/encoding_info.pkl")
    return model, scaler, encoding_info

@st.cache_resource
def load_segmentation_assets():
    model_url = "https://raw.githubusercontent.com/marymorkos/SalesPlaybookDS5640/main/customer_segmentation/models/kmeans_model.joblib"
    scaler_url = "https://raw.githubusercontent.com/marymorkos/SalesPlaybookDS5640/main/customer_segmentation/models/minmax_scaler.joblib"
    scaler = joblib.load(BytesIO(requests.get(scaler_url).content))
    model = joblib.load(BytesIO(requests.get(model_url).content))
    return scaler, model

# === Page: Deal Outcome Predictor ===
if page == "Deal Outcome Predictor":
    from preprocessing import preprocess_data

    st.title("🎯 Deal Outcome Prediction Dashboard")
    st.markdown("Upload your deal data to predict win probability.")

    uploaded_file = st.file_uploader("📁 Upload Deal CSV", type=["csv"])

    if uploaded_file:
        model, scaler, encoding_info = load_deal_model_assets()
        input_df = pd.read_csv(uploaded_file)
        st.dataframe(input_df.head())

        try:
            # === Prediction using processed data ===
            processed_df = preprocess_data(input_df, scaler, encoding_info)
            processed_df = processed_df[model.feature_names_in_]

            predictions = model.predict(processed_df)
            prediction_proba = model.predict_proba(processed_df)[:, 1]

            # === Combine predictions with original data for EDA ===
            result_df = input_df.copy()
            result_df["Predicted Outcome"] = np.where(predictions == 1, "Won", "Lost")
            result_df["Win Probability"] = prediction_proba.round(3)

            st.success("✅ Predictions Complete")
            st.metric("📈 Average Win Probability", f"{prediction_proba.mean():.2%}")
            st.dataframe(result_df.head())

            # === INTERACTIVE FILTERING & EDA ===
            st.markdown("---")
            st.header("🔍 Explore and Filter Deals")

            # Dynamically build filters for any categorical columns
            filter_columns = ["Deal Stage", "Deal Type", "ICP Fit"]
            filters = {}
            for col in filter_columns:
                if col in result_df:
                    filters[col] = st.multiselect(f"Filter by {col}", result_df[col].dropna().unique(), default=result_df[col].dropna().unique())

            # Apply filters
            filtered_df = result_df.copy()
            for col, selected_values in filters.items():
                filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

            # === Summary metrics and visualizations ===
            st.subheader("📊 Summary Statistics (Filtered Deals)")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Deals", len(filtered_df))
            with col2:
                st.metric("Avg Win Probability", f"{filtered_df['Win Probability'].mean():.2%}" if not filtered_df.empty else "N/A")

            # Distribution of outcomes
            if "Predicted Outcome" in filtered_df:
                st.bar_chart(filtered_df["Predicted Outcome"].value_counts())

            # Show full filtered deal table
            st.subheader("📄 Matching Deals")
            default_cols = ["Deal Name", "Deal Stage", "Deal Type", "ICP Fit", "Predicted Outcome", "Win Probability"]
            show_cols = [col for col in default_cols if col in filtered_df.columns]
            st.dataframe(filtered_df[show_cols].sort_values("Win Probability", ascending=False).reset_index(drop=True))

            # === Select a deal to inspect ===
            st.subheader("🔬 Drill Down on a Specific Deal")
            deal_id_col = "Deal Name" if "Deal Name" in filtered_df.columns else filtered_df.columns[0]
            if not filtered_df.empty:
                selected_deal = st.selectbox("Choose a Deal", filtered_df[deal_id_col].unique())
                deal_row = filtered_df[filtered_df[deal_id_col] == selected_deal].iloc[0]

                st.markdown(f"**🧾 Deal Name:** {deal_row.get('Deal Name', 'N/A')}")
                st.markdown(f"**🗂️ Stage:** {deal_row.get('Deal Stage', 'N/A')}")
                st.markdown(f"**🏷️ Type:** {deal_row.get('Deal Type', 'N/A')}")
                st.markdown(f"**📌 ICP Fit:** {deal_row.get('ICP Fit', 'N/A')}")
                st.markdown(f"**🎯 Predicted Outcome:** `{deal_row['Predicted Outcome']}`")
                st.markdown(f"**📈 Win Probability:** `{deal_row['Win Probability']:.2%}`")
            else:
                st.warning("No deals match your selected filters.")

            # === Download button for filtered CSV ===
            csv_filtered = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Filtered Deals", csv_filtered, "filtered_deals.csv", "text/csv")

        except Exception as e:
            st.error(f"Error during processing: {e}")
    else:
        st.info("Upload a CSV to begin.")


# === Page: Customer Segmentation ===
elif page == "Customer Segmentation":
    st.title("🔍 Customer Segmentation Predictor")

    scaler, model = load_segmentation_assets()

    st.write("### Enter customer information or upload a CSV")

    # Manual entry
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
        meanings = {
            0: "🟢 Segment 0: Small businesses with high digital engagement",
            1: "🔵 Segment 1: Mid-market companies with moderate engagement",
            2: "🟡 Segment 2: Enterprise clients with high revenue and specialized needs",
            3: "🟠 Segment 3: Growth-stage companies with high potential"
        }
        st.markdown(f"**Interpretation:** {meanings.get(segment, 'Unknown')}")

        with st.expander("📈 Cluster Visualization"):
            try:
                pca = PCA(n_components=2)
                centroids = model.cluster_centers_
                pca_centroids = pca.fit_transform(centroids)
                input_pca = pca.transform(scaler.transform(input_df))

                fig, ax = plt.subplots(figsize=(8, 6))

                # Plot centroids - updated for 4 clusters with distinct colors
                ax.scatter(pca_centroids[:, 0], pca_centroids[:, 1], 
                        s=200, marker='o', c=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], 
                        edgecolors='k', alpha=0.7, label='Cluster Centers')

                # Plot input data point
                ax.scatter(input_pca[:, 0], input_pca[:, 1], 
                        s=300, marker='*', c='purple', 
                        edgecolors='k', label='Your Customer')

                # Add cluster labels
                for i, (x, y) in enumerate(pca_centroids):
                    ax.annotate(f'Cluster {i}', (x, y), 
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center')

                ax.set_title('Customer Position Relative to Cluster Centers', fontsize=14)
                ax.legend(loc='best')
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Plotting failed: {e}")

    # Batch upload
    st.write("### 📥 Batch Prediction from CSV")
    uploaded = st.file_uploader("Upload CSV with 7 numeric columns", type=["csv"], key="segment_csv")
    if uploaded:
        df_batch = pd.read_csv(uploaded)
    
        try:
    
    # 🧠 Simplified Customer Segmentation Model (7 Numeric Features from Clean GitHub Data)

            import pandas as pd
            import datetime
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.cluster import KMeans
            import joblib



            # 🔹 STEP 2: Ensure 'Year Founded' is not null and compute 'Company_Age'
            df_batch = df_batch[df_batch['Year Founded'].notnull()].copy()
            current_year = datetime.datetime.now().year
            df_batch['Company_Age'] = current_year - df_batch['Year Founded']

            # 🔹 STEP 3: Define 7 key numeric features for segmentation
            selected_features = [
                'Annual Revenue',
                'Number of Form Submissions',
                'Number of times contacted',
                'Number of Pageviews',
                'Company_Age',
                'Number of Employees',
                'Number of Sessions'
            ]

            # 🔹 STEP 4: Prepare feature matrix (remove rows with any missing values)
            df_batch = df_batch[selected_features].dropna()

    

            X_scaled = scaler.transform(df_batch)
            preds = model.predict(X_scaled)
            df_batch["Predicted Segment"] = preds
            st.dataframe(df_batch)

            # === PCA Plot of All Clusters ===
            st.subheader("🧬 Cluster Visualization")



            import pacmap
            import plotly.express as px
            from sklearn.decomposition import PCA

            # Consistent color palette
            custom_colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            ]

            # Convert predictions to labeled string format
            segment_labels = ["Segment " + str(x) for x in preds]

            # === PCA Plot ===
            pca = PCA(n_components=2)
            pca_embed = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(pca_embed, columns=["PCA1", "PCA2"])
            pca_df["Segment"] = segment_labels

            fig_pca = px.scatter(
                pca_df,
                x="PCA1",
                y="PCA2",
                color="Segment",
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                title="PCA Projection of Customer Segments",
                width=700,
                height=450
            )
            fig_pca.update_traces(marker=dict(size=10, opacity=0.7))
            fig_pca.update_layout(legend_title_text="Segment")
            st.plotly_chart(fig_pca, use_container_width=True)

            # === PaCMAP Plot ===
            reducer = pacmap.PaCMAP(n_components=2, random_state=42)
            pacmap_embed = reducer.fit_transform(X_scaled)
            pacmap_df = pd.DataFrame(pacmap_embed, columns=["PaCMAP_1", "PaCMAP_2"])
            pacmap_df["Segment"] = segment_labels

            fig_pacmap = px.scatter(
                pacmap_df,
                x="PaCMAP_1",
                y="PaCMAP_2",
                color="Segment",
                color_discrete_sequence=custom_colors,
                title="PaCMAP Projection of Customer Segments",
                width=700,
                height=450
            )
            fig_pacmap.update_traces(marker=dict(size=10, opacity=0.7))
            fig_pacmap.update_layout(legend_title_text="Segment")
            st.plotly_chart(fig_pacmap, use_container_width=True)




            # === Cluster Summary and Filtering ===
            st.subheader("📊 Cluster Summary and Filtering")
            selected_segment = st.selectbox("Select Segment to Explore", sorted(df_batch["Predicted Segment"].unique()))

            segment_df = df_batch[df_batch["Predicted Segment"] == selected_segment]
            st.markdown(f"**Segment Size:** {len(segment_df)} customers")

            # Summary statistics
            st.markdown("**Average Feature Values**")
            st.dataframe(segment_df.drop(columns=["Predicted Segment"]).mean().to_frame("Average Value").round(2))

            st.subheader("📊 Cluster Comparison for Each Feature")

            # Calculate mean of each feature grouped by cluster
            cluster_means = df_batch.groupby("Predicted Segment").mean(numeric_only=True)
            
            import altair as alt

            feature_df = cluster_means.reset_index().melt(id_vars="Predicted Segment", var_name="Feature", value_name="Average")

            for feature in feature_df["Feature"].unique():
                chart = alt.Chart(feature_df[feature_df["Feature"] == feature]).mark_bar().encode(
                    x=alt.X("Predicted Segment:N", title="Segment"),
                    y=alt.Y("Average:Q", title=f"Average {feature}"),
                    tooltip=["Predicted Segment", "Average"]
                ).properties(title=f"{feature} by Segment")
                st.altair_chart(chart, use_container_width=True)

            # Allow download of filtered segment
            csv_segment = segment_df.to_csv(index=False).encode("utf-8")
            st.download_button(f"⬇️ Download Segment {selected_segment} Data", csv_segment, f"segment_{selected_segment}.csv", "text/csv")

            # Download all
            csv = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button("📤 Download All Predictions", csv, "segmented_customers.csv", "text/csv")

       # try:
       #     X_scaled = scaler.transform(df_batch)
       #     preds = model.predict(X_scaled)
       #     df_batch["Predicted Segment"] = preds
       #     st.dataframe(df_batch)

       #     csv = df_batch.to_csv(index=False).encode('utf-8')
       #     st.download_button("📤 Download Predictions", csv, "segmented_customers.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")



