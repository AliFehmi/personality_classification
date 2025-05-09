import json
import numpy as np
from empath import Empath
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm

# === Load user-level data from JSON ===
with open("../data/grouped_users_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["clean_text"] for item in data]
labels = np.array([1 if item["label"] == "y" else 0 for item in data])

# === Extract Empath features (12D semantic vectors) ===
lexicon = Empath()
empath_categories = [
    "negative_emotion", "pain", "death", "anxiety", "violence",
    "disappointment", "nervousness", "positive_emotion", "joy",
    "friends", "optimism", "celebration"
]
empath_vectors = [
    list(lexicon.analyze(text, categories=empath_categories, normalize=True).values())
    for text in texts
]
empath_vectors = np.array(empath_vectors)

# === 5-fold stratified cross-validation ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = []

for train_idx, test_idx in tqdm(kf.split(texts, labels), total=5):
    # Split raw and empath data
    X_train_text = [texts[i] for i in train_idx]
    X_test_text = [texts[i] for i in test_idx]
    X_train_empath = csr_matrix(empath_vectors[train_idx])
    X_test_empath = csr_matrix(empath_vectors[test_idx])
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    # Chi2 feature selection on TF-IDF only
    selector = SelectKBest(score_func=chi2, k=4000)
    X_train_tfidf = selector.fit_transform(X_train_tfidf, y_train)
    X_test_tfidf = selector.transform(X_test_tfidf)

    # Combine TF-IDF + Empath
    X_train_combined = hstack([X_train_tfidf, X_train_empath])
    X_test_combined = hstack([X_test_tfidf, X_test_empath])

    # Train classifier
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_combined, y_train)
    y_pred = clf.predict(X_test_combined)

    # Evaluate
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    metrics.append((acc, precision, recall, f1))

# === Average metrics ===
metrics = np.array(metrics)
avg = metrics.mean(axis=0)

# === Save results ===
with open("../outputs/crossval_metrics.txt", "w") as f:
    f.write("5-Fold Cross-Validation Results (Baseline with Empath + TF-IDF + chi2)\n")
    f.write(f"Accuracy : {avg[0]:.4f}\n")
    f.write(f"Precision: {avg[1]:.4f}\n")
    f.write(f"Recall   : {avg[2]:.4f}\n")
    f.write(f"F1-Score : {avg[3]:.4f}\n")
