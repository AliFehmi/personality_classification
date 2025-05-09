import joblib
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from scipy import sparse
from scipy.sparse import hstack

# === Load model components (trained on TF-IDF + Empath) ===
vectorizer = joblib.load("../models/tfidf_vectorizer.joblib")
selector = joblib.load("../models/chi2_selector.joblib")
clf = joblib.load("../models/logistic_model_with_empath_and_chi2.joblib")

# === Load test data ===
with open("../outputs/test_texts.json") as f:
    X_text = json.load(f)
with open("../outputs/test_empath_vectors.json") as f:
    X_empath = json.load(f)
with open("../outputs/test_labels.json") as f:
    y_test = json.load(f)

# === Transform text using TF-IDF and chi² ===
X_text_vec = vectorizer.transform(X_text)
X_text_vec = selector.transform(X_text_vec)

# === Convert Empath features to sparse and combine ===
X_empath_sparse = sparse.csr_matrix(X_empath)
X_test_combined = hstack([X_text_vec, X_empath_sparse])

# === Predict with combined model ===
y_pred = clf.predict(X_test_combined)

# === Save predictions ===
with open("../outputs/test_predictions_from_analysis.json", "w") as f:
    json.dump(y_pred.tolist(), f)

# === Save classification report and confusion matrix ===
report = classification_report(y_test, y_pred, target_names=["Not Neurotic", "Neurotic"])
conf_matrix = confusion_matrix(y_test, y_pred)

with open("../outputs/classification_report.txt", "w") as f:
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(conf_matrix))

# === Analyze top features (TF-IDF only) ===
# Empath features are not included in chi2 or vectorizer feature names
chi2_scores = selector.scores_
selected_indices = selector.get_support(indices=True)
feature_names_all = np.array(vectorizer.get_feature_names_out())

# Only selected TF-IDF features will have weights/chi² scores
selected_features = feature_names_all[selected_indices]
chi2_scores = chi2_scores[selected_indices]
weights = clf.coef_[0][:len(selected_features)]  # exclude weights for Empath features

features_info = list(zip(selected_features, chi2_scores, weights))

# Sort and extract
top_neurotic = sorted(features_info, key=lambda x: x[2], reverse=True)[:5]
top_not_neurotic = sorted(features_info, key=lambda x: x[2])[:5]

# Save top features
with open("../outputs/top_features.txt", "w") as f:
    f.write("Top 5 features predicting NEUROTIC:\n")
    for word, chi2, weight in top_neurotic:
        f.write(f"{word:25s} | chi2 = {chi2:.4f} | weight = {weight:.4f}\n")

    f.write("\nTop 5 features predicting NOT NEUROTIC:\n")
    for word, chi2, weight in top_not_neurotic:
        f.write(f"{word:25s} | chi2 = {chi2:.4f} | weight = {weight:.4f}\n")

        
# Get the last 12 weights (these correspond to the Empath features)
empath_weights = clf.coef_[0][-12:]

# Load empath category names
with open("../outputs/empath_categories.json") as f:
    empath_categories = json.load(f)

for name, weight in zip(empath_categories, empath_weights):
    print(f"{name:20s} | weight = {weight:.4f}")
