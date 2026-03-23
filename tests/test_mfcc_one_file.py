import os
import sys

# Ensure project root (D:\emotion_assistant) is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from perception.speech_features import extract_mfcc_features
import numpy as np

# ⬅️ put an actual filename you recorded earlier
wav_path = r"data/speech/raw/positive_3c4257a3.wav"

feat = extract_mfcc_features(wav_path)
print("Feature shape:", feat.shape)
print("Feature vector:", feat)
