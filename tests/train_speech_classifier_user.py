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

DATA_NPZ = os.path.join(PROJECT_ROOT, "data", "speech", "speech_dataset_user_ser.npz")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "speech", "speech_svm_user_ser.joblib")

data = np.load(DATA_NPZ, allow_pickle=True)
X = data["X"]
y = data["y"]

print("Loaded user dataset:", X.shape, y.shape)
labels = np.unique(y)
print("Label distribution:")
for lbl in labels:
    print(lbl, ":", np.sum(y == lbl))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

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

y_pred = clf.predict(X_test)
print("\nUser-specific SVM report:")
print(classification_report(y_test, y_pred, digits=3))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"]))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print("\nSaved user SER model to:", MODEL_PATH)
