import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from perception.speech_emotion import predict_speech_emotion

# Pick any CREMA-D wav file path
wav_path = os.path.join(PROJECT_ROOT, "data", "CREMA-D", "AudioWAV", "1001_DFA_HAP_XX.wav")  # adjust name if needed

print("Testing file:", wav_path)
emotion = predict_speech_emotion(wav_path)
print("Predicted speech emotion:", emotion)
