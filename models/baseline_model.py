import json
import joblib
import numpy as np
from empath import Empath
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix
import os

# === Create model/output directories if they don't exist ===
os.makedirs("../models", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

# === Load user-level data from JSON ===
with open("../data/grouped_users_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["clean_text"] for item in data]  # aggregated user texts
labels = [1 if item["label"] == "y" else 0 for item in data]  # convert labels to binary

# === Extract Empath features (dense emotional/semantic categories) ===
lexicon = Empath()
empath_categories = [ # themes we want to map the user's text to.
    "negative_emotion", "pain", "death", "anxiety", "violence",
    "disappointment", "nervousness", "positive_emotion", "joy",
    "friends", "optimism", "celebration"
]

# For each text, get normalized empath category scores (proportional to word count). See part 4) in baseline.md to understand how it works
empath_vectors = [
    list(lexicon.analyze(text, categories=empath_categories, normalize=True).values())
    for text in texts
]
empath_vectors = np.array(empath_vectors)  # convert to array for later concatenation

# === Split both TF and Empath features into train/test ===
X_text_train, X_text_test, X_empath_train, X_empath_test, y_train, y_test = train_test_split(
    texts, empath_vectors, labels, test_size=0.2, random_state=42, stratify=labels
)

# === TF-IDF Vectorization (1- to 3-grams, max 5000 features) ===
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000) # selects 5000 features (1-3 grams)
X_text_train_vec = vectorizer.fit_transform(X_text_train)  # for each feature calculate the idf, for each user creates a vector with tf-idf values (there are max_features amount of features in each vector)
X_text_test_vec = vectorizer.transform(X_text_test)        # at this point idf values are fixed, only tf values are calculated for each user and creates tf-idf value vectors for test data.

# === Apply Chi-squared feature selection on TF-IDF only ===
selector = SelectKBest(score_func=chi2, k=4000)  # keep top 4000 most class-associated features
X_text_train_vec = selector.fit_transform(X_text_train_vec, y_train)  # fit + transform train
X_text_test_vec = selector.transform(X_text_test_vec)                # transform test

# === Convert Empath features to sparse and combine with TF-IDF features ===
X_empath_train_sparse = csr_matrix(X_empath_train)
X_empath_test_sparse = csr_matrix(X_empath_test)
# The TF-IDF feature matrix: shape e.g., (n_train, 4000) The Empath feature matrix: shape (n_train, 12)
X_train_combined = hstack([X_text_train_vec, X_empath_train_sparse])  # final training set
X_test_combined = hstack([X_text_test_vec, X_empath_test_sparse])     # final test set
# shape (n_train, 4012)
# === Train logistic regression model on combined features ===
clf = LogisticRegression(max_iter=1000, class_weight="balanced")  # handle class imbalance
clf.fit(X_train_combined, y_train)  # learn weights
y_pred = clf.predict(X_test_combined)  # make predictions on test set

# === Save trained model and components for later analysis ===
joblib.dump(vectorizer, "../models/tfidf_vectorizer.joblib")
joblib.dump(selector, "../models/chi2_selector.joblib")
joblib.dump(clf, "../models/logistic_model_with_empath_and_chi2.joblib")
with open("../outputs/empath_categories.json", "w") as f:
    json.dump(empath_categories, f)

# === Save test data and predictions ===
with open("../outputs/test_texts.json", "w") as f:
    json.dump(X_text_test, f)

with open("../outputs/test_empath_vectors.json", "w") as f:
    json.dump(X_empath_test.tolist(), f)

with open("../outputs/test_labels.json", "w") as f:
    json.dump(y_test, f)

with open("../outputs/test_predictions.json", "w") as f:
    json.dump(y_pred.tolist(), f)


