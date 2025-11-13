# train.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from scipy.stats import randint as sp_randint
import joblib

# Attempt multiple likely CSV locations (project-friendly)
CANDIDATES = [
    os.path.join("data", "DiseaseAndSymptoms.csv"),
    "DiseaseAndSymptoms.csv",
    r"C:/Users/gande/OneDrive/文档/2ndyear/aiml/DiseaseAndSymptoms.csv",
    "/mnt/data/DiseaseAndSymptoms.csv",
    "/mnt/data/MedAI_transformed.csv",
]

def find_data_path(cands):
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None

DATA_PATH = find_data_path(CANDIDATES)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Could not find dataset. Tried:\n" + "\n".join(CANDIDATES)
    )

OUT_DIR = os.path.join(".", "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Identify symptom columns (columns beginning with "Symptom")
symptom_cols = [c for c in df.columns if str(c).startswith("Symptom")]
if not symptom_cols:
    raise ValueError("No columns beginning with 'Symptom' were found in the CSV.")

# Collect all unique symptom names robustly (handle comma-separated cells)
all_symptoms = []
for c in symptom_cols:
    col_vals = df[c].fillna("").astype(str)
    for v in col_vals:
        if v is None:
            continue
        v = v.strip()
        if v == "":
            continue
        # if multiple symptoms in a cell separated by comma/semicolon, split them
        parts = [p.strip() for p in re.split(r"[;,/]", v) if p.strip()] if "," in v or ";" in v or "/" in v else [v]
        all_symptoms.extend(parts)

# Use sorted unique list
feature_list = sorted(set(all_symptoms))

# Build binary feature DataFrame
X = pd.DataFrame(0, index=df.index, columns=feature_list)

import re
for col in symptom_cols:
    col_vals = df[col].fillna("").astype(str)
    for i, val in col_vals.items():
        if not val or str(val).strip() == "":
            continue
        parts = [p.strip() for p in re.split(r"[;,/]", val) if p.strip()] if "," in val or ";" in val or "/" in val else [val.strip()]
        for p in parts:
            if p in X.columns:
                X.at[i, p] = 1

# Target
if "Disease" not in df.columns:
    # fallback check for possible column name differences
    raise ValueError("No 'Disease' column found in dataset.")
y = df["Disease"].astype(str)

# Label encode
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split (stratify if possible)
if len(np.unique(y_enc)) < 2:
    raise ValueError("Need at least two classes in 'Disease' column to train.")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# Random forest + randomized search
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
param_dist = {
    "n_estimators": sp_randint(100, 500),
    "max_depth": sp_randint(5, 50),
    "min_samples_split": sp_randint(2, 20),
    "min_samples_leaf": sp_randint(1, 20),
    "criterion": ["gini", "entropy"],
}

rs = RandomizedSearchCV(
    rf,
    param_dist,
    n_iter=20,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1,
    random_state=42,
    verbose=2,
)

rs.fit(X_train, y_train)

print("Best params:", rs.best_params_)
best_rf = rs.best_estimator_

# Evaluate
preds = best_rf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, preds))
print("Test macro F1:", f1_score(y_test, preds, average="macro", zero_division=0))
print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))

# Save artifacts
joblib.dump(best_rf, os.path.join(OUT_DIR, "best_rf_randomized.pkl"))
joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.pkl"))
pd.Series(feature_list).to_csv(os.path.join(OUT_DIR, "feature_list.csv"), index=False, header=False)

print("Saved model and artifacts to", OUT_DIR)
