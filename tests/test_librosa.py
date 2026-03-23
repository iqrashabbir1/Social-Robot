import librosa
import numpy as np

y, sr = librosa.load("tests/sample_audio.wav", sr=None)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print("MFCC shape:", mfcc.shape)
print("Mean vector:", np.mean(mfcc, axis=1))
