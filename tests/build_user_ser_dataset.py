import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from perception.speech_features import extract_mfcc_features

BASE_DIR = os.path.join(PROJECT_ROOT, "data", "user_ser")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "speech", "speech_dataset_user_ser.npz")


def main():
    X = []
    y = []

    labels = ["positive", "neutral", "negative"]

    for lbl in labels:
        dir_lbl = os.path.join(BASE_DIR, lbl)
        if not os.path.isdir(dir_lbl):
            continue

        for fname in os.listdir(dir_lbl):
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(dir_lbl, fname)
            feat_vec = extract_mfcc_features(path)
            X.append(feat_vec)
            y.append(lbl)

    X = np.array(X)
    y = np.array(y)

    print("User SER dataset shape:", X.shape, y.shape)
    unique, counts = np.unique(y, return_counts=True)
    for lbl, c in zip(unique, counts):
        print(lbl, ":", c)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, X=X, y=y)
    print("Saved user dataset to:", OUT_PATH)


if __name__ == "__main__":
    main()
