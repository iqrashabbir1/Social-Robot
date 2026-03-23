import os
import sys
import numpy as np

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from perception.speech_features import extract_mfcc_features

# 👇 Adjust this path if your CREMA-D folder name is different
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "CREMA-D", "AudioWAV")
OUT_NPZ = os.path.join(PROJECT_ROOT, "data", "speech", "speech_dataset_crema.npz")

print("Using RAW_DIR:", RAW_DIR)

X = []
y = []

# Emotion code → 3-class mapping
def map_emotion_code_to_label(code: str) -> str:
    """
    CREMA-D codes:
      ANG, DIS, FEA, HAP, NEU, SAD
    Map to: positive / neutral / negative
    """
    code = code.upper()
    if code == "HAP":
        return "positive"
    elif code == "NEU":
        return "neutral"
    elif code in ["ANG", "DIS", "FEA", "SAD"]:
        return "negative"
    else:
        return None  # unknown

files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".wav")]
files.sort()

for fname in files:
    path = os.path.join(RAW_DIR, fname)

    # CREMA-D filename example: 1001_DFA_ANG_XX.wav
    parts = fname.split("_")
    if len(parts) < 3:
        print("Skipping unexpected filename:", fname)
        continue

    emotion_code = parts[2]  # 'ANG', 'HAP', 'NEU', etc.
    label = map_emotion_code_to_label(emotion_code)
    if label is None:
        print("Skipping unknown emotion code:", emotion_code, "in", fname)
        continue

    try:
        feat_vec = extract_mfcc_features(path)
        X.append(feat_vec)
        y.append(label)
        # comment this if too verbose:
        # print("Processed:", fname, "->", label)
    except Exception as e:
        print("Error processing", fname, ":", e)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape, y.shape)
unique, counts = np.unique(y, return_counts=True)
print("Label distribution:")
for lbl, cnt in zip(unique, counts):
    print(f"  {lbl}: {cnt}")

os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
np.savez(OUT_NPZ, X=X, y=y)
print("Saved dataset to", OUT_NPZ)
