import numpy as np
import librosa

def extract_mfcc_features(path: str, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC-based features from an audio file.
    We compute:
      - 13 MFCC coefficients over time
      - Mean and standard deviation of each coefficient
    Final feature vector has length 26.
    """
    y, sr = librosa.load(path, sr=None)

    # Optional: trim leading/trailing silence to focus on voiced part
    y, _ = librosa.effects.trim(y)

    # Compute MFCCs: shape (n_mfcc, T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Mean and std over time axis
    mfcc_mean = mfcc.mean(axis=1)  # shape (n_mfcc,)
    mfcc_std = mfcc.std(axis=1)    # shape (n_mfcc,)

    # Concatenate mean and std → 26-dim vector
    feat_vec = np.concatenate([mfcc_mean, mfcc_std], axis=0)

    return feat_vec
