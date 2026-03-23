import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DATA_NPZ = os.path.join(PROJECT_ROOT, "data", "speech", "speech_dataset_crema.npz")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "speech", "speech_svm_crema_balanced.joblib")

data = np.load(DATA_NPZ, allow_pickle=True)
X = data["X"]
y = data["y"]

print("Loaded dataset:", X.shape, y.shape)

labels = np.unique(y)
print("Original label distribution:")
for lbl in labels:
    print(lbl, ":", np.sum(y == lbl))

# ---------- BALANCING ----------
rng = np.random.default_rng(42)
indices_balanced = []

min_count = min(np.sum(y == lbl) for lbl in labels)
target_per_class = min(1000, min_count)  # cap at 1000 per class

print(f"\nBalancing to {target_per_class} samples per class.")

for lbl in labels:
    idx = np.where(y == lbl)[0]
    if len(idx) > target_per_class:
        chosen = rng.choice(idx, size=target_per_class, replace=False)
    else:
        chosen = idx
    indices_balanced.append(chosen)

indices_balanced = np.concatenate(indices_balanced)

X_bal = X[indices_balanced]
y_bal = y[indices_balanced]

print("Balanced dataset shape:", X_bal.shape, y_bal.shape)
for lbl in labels:
    print(lbl, ":", np.sum(y_bal == lbl))

# ---------- TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# ---------- MODEL: StandardScaler + SVM (RBF) ----------
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        C=10.0,
        gamma="scale",
        random_state=42,
    ))
])

clf.fit(X_train, y_train)

# ---------- EVALUATION ----------
y_pred = clf.predict(X_test)

print("\nClassification report (SVM, balanced training, 26-dim MFCC features):")
print(classification_report(y_test, y_pred, digits=3))

print("Confusion matrix (rows=true, cols=pred) with order [positive, neutral, negative]:")
order = ["positive", "neutral", "negative"]
print(confusion_matrix(y_test, y_pred, labels=order))

# ---------- SAVE MODEL ----------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print("\nSaved SVM model to", MODEL_PATH)
