# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

st.set_page_config(page_title="Multi-Disease Prediction", layout="wide")

ART = "artifacts"
MODEL_FILE = os.path.join(ART, "best_rf_randomized.pkl")
LE_FILE = os.path.join(ART, "label_encoder.pkl")
FEATURE_FILE = os.path.join(ART, "feature_list.csv")

# try to find dataset for the "Data Insights" tab
DATA_CANDIDATES = [
    os.path.join("data", "DiseaseAndSymptoms.csv"),
    "DiseaseAndSymptoms.csv",
    "/mnt/data/DiseaseAndSymptoms.csv",
    "/mnt/data/MedAI_transformed.csv",
    r"C:/Users/gande/OneDrive/文档/2ndyear/aiml/DiseaseAndSymptoms.csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)

# load model/artifacts
if not (os.path.exists(MODEL_FILE) and os.path.exists(LE_FILE) and os.path.exists(FEATURE_FILE)):
    st.error("Model artifacts not found. Run train.py first to create ./artifacts/")
    st.stop()

model = joblib.load(MODEL_FILE)
le = joblib.load(LE_FILE)
features = pd.read_csv(FEATURE_FILE, header=None).iloc[:, 0].tolist()

st.title("🔷 Multi-Disease Prediction System")
st.write("Predict probable disease(s) from symptoms.")

# Sidebar symptom selection
selected = st.sidebar.multiselect("Select symptoms you have:", options=features)

col1, col2 = st.columns([2, 3])

with col1:
    st.header("Prediction")
    if st.button("Predict"):
        if not selected:
            st.warning("Select at least one symptom.")
        else:
            # build input row
            input_df = pd.DataFrame(0, index=[0], columns=features)
            for s in selected:
                if s in input_df.columns:
                    input_df.at[0, s] = 1

            pred_idx = model.predict(input_df)[0]
            pred_name = le.inverse_transform([pred_idx])[0]
            probs = model.predict_proba(input_df)[0]
            top_idx = probs.argsort()[-5:][::-1]

            st.success(f"### Predicted Disease: {pred_name}")
            df_probs = pd.DataFrame({
                "Disease": [le.inverse_transform([i])[0] for i in top_idx],
                "Confidence (%)": [round(float(probs[i]) * 100, 2) for i in top_idx]
            })
            fig = px.bar(df_probs, x="Confidence (%)", y="Disease", orientation="h",
                         color="Confidence (%)", color_continuous_scale="blues")
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Model Feature Importances (Top 20)")
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(20)
        fi_df = fi.reset_index()
        fi_df.columns = ["Symptom", "Importance"]
        fig2 = px.bar(fi_df, x="Importance", y="Symptom", orientation="h",
                      color="Importance", color_continuous_scale="blues")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("Model does not expose feature_importances_")

st.markdown("---")
st.header("Dataset preview")
if DATA_PATH:
    try:
        df = pd.read_csv(DATA_PATH)
        st.write(f"Loaded dataset from `{DATA_PATH}`")
        st.dataframe(df.head(10))
        st.write(f"Records: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
    except Exception as e:
        st.warning(f"Failed to load dataset at {DATA_PATH}: {e}")
else:
    st.warning("Dataset not found in expected locations. Place it in ./data or project root.")
