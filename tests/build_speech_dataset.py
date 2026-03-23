import os
import numpy as np
from perception.speech_features import extract_mfcc_features

RAW_DIR = "data/speech/raw"
OUT_NPZ = "data/speech/speech_dataset.npz"

X = []
y = []

for fname in os.listdir(RAW_DIR):
    if not fname.lower().endswith(".wav"):
        continue
    path = os.path.join(RAW_DIR, fname)

    # label from filename prefix: positive_*, neutral_*, negative_*
    label = fname.split("_")[0].lower()
    if label not in ["positive", "neutral", "negative"]:
        print("Skipping file with unknown label pattern:", fname)
        continue

    try:
        feat_vec = extract_mfcc_features(path)
        X.append(feat_vec)
        y.append(label)
        print("Processed:", fname, "->", label)
    except Exception as e:
        print("Error processing", fname, ":", e)

X = np.array(X)  # shape: (N, 13)
y = np.array(y)

print("Dataset shape:", X.shape, y.shape)
os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
np.savez(OUT_NPZ, X=X, y=y)
print("Saved dataset to", OUT_NPZ)
