# make_shap.py
import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

ART = "artifacts"
MODEL_FILE = os.path.join(ART, "best_rf_randomized.pkl")
FEATURE_FILE = os.path.join(ART, "feature_list.csv")
DATA_CANDIDATES = [
    os.path.join("data", "DiseaseAndSymptoms.csv"),
    "DiseaseAndSymptoms.csv",
    "/mnt/data/DiseaseAndSymptoms.csv",
    "/mnt/data/MedAI_transformed.csv",
]

DATA_PATH = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("Could not find dataset. Put it in ./data/ or project root.")

model = joblib.load(MODEL_FILE)
features = pd.read_csv(FEATURE_FILE, header=None).iloc[:, 0].tolist()
df = pd.read_csv(DATA_PATH)

symptom_cols = [c for c in df.columns if c.startswith("Symptom")]
X = pd.DataFrame(0, index=df.index, columns=features)

import re
for col in symptom_cols:
    col_vals = df[col].fillna("").astype(str)
    for i, v in col_vals.items():
        if not v or str(v).strip() == "":
            continue
        parts = [p.strip() for p in re.split(r"[;,/]", v) if p.strip()] if "," in v or ";" in v or "/" in v else [v.strip()]
        for p in parts:
            if p in X.columns:
                X.at[i, p] = 1

# SHAP: use TreeExplainer for tree models.
explainer = shap.TreeExplainer(model)
# For sklearn RandomForest multiclass shap_values is a list; shap.summary_plot handles both
print("Computing SHAP values (may take some time)...")
shap_values = explainer.shap_values(X)

OUT_IMG = os.path.join(ART, "shap_summary.png")
plt.figure(figsize=(10, 7))
# shap.summary_plot can accept shap_values as list (multiclass) or array (binary/continuous)
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(OUT_IMG, bbox_inches="tight")
plt.close()
print("Saved SHAP summary to", OUT_IMG)
