# predict.py
import os
import joblib
import pandas as pd

ART = "artifacts"
MODEL_FILE = os.path.join(ART, "best_rf_randomized.pkl")
LE_FILE = os.path.join(ART, "label_encoder.pkl")
FEATURE_FILE = os.path.join(ART, "feature_list.csv")

model = joblib.load(MODEL_FILE)
le = joblib.load(LE_FILE)
features = pd.read_csv(FEATURE_FILE, header=None).iloc[:, 0].tolist()

def predict_from_symptoms(symptoms):
    """
    symptoms: list of strings (symptom names)
    returns: (predicted_label_str, list_of_top5 (label, prob))
    """
    import numpy as np
    x = [1 if f in symptoms else 0 for f in features]
    pred_idx = model.predict([x])[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    probs = model.predict_proba([x])[0]
    top_idx = np.argsort(probs)[-5:][::-1]
    top = [(le.inverse_transform([i])[0], float(probs[i])) for i in top_idx]
    return pred_label, top

# Example usage
if __name__ == "__main__":
    sample = ["headache", "fever"]
    print("Sample symptoms:", sample)
    label, top5 = predict_from_symptoms(sample)
    print("Predicted:", label)
    print("Top 5:", top5)
