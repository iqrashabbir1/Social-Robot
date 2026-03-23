from __future__ import annotations
from typing import Optional


def normalize_face_emotion(raw_face: Optional[str]) -> Optional[str]:
    """
    Map DeepFace-style labels to canonical labels used in the robot:
      - happy
      - sad
      - angry
      - fear
      - neutral
    Anything unknown returns None.
    """
    if raw_face is None:
        return None

    raw = raw_face.strip().lower()

    # Common DeepFace labels: angry, disgust, fear, happy, sad, surprise, neutral
    if raw in ["happy"]:
        return "happy"
    if raw in ["sad"]:
        return "sad"
    if raw in ["angry"]:
        return "angry"
    if raw in ["fear", "fearful"]:
        return "fear"
    if raw in ["neutral"]:
        return "neutral"

    # Map 'disgust' and 'surprise' to neutral for now (could refine later)
    if raw in ["disgust", "surprise"]:
        return "neutral"

    # Unknown / unsupported label
    return None


def normalize_speech_emotion(raw_speech: Optional[str]) -> Optional[str]:
    """
    Ensure speech emotion is one of:
      - 'positive'
      - 'neutral'
      - 'negative'
    or None if unknown.
    """
    if raw_speech is None:
        return None

    raw = raw_speech.strip().lower()
    if raw in ["positive", "pos"]:
        return "positive"
    if raw in ["neutral", "neu"]:
        return "neutral"
    if raw in ["negative", "neg"]:
        return "negative"

    return None


def fuse_emotions(face_emotion: Optional[str],
                  speech_emotion: Optional[str]) -> str:
    """
    Fuse face and speech into a single high-level label.

    Rules:
      1) Normalize inputs.
      2) If face is clearly non-neutral (happy/sad/angry/fear), trust face.
      3) If face is neutral:
           - speech negative  -> 'sad'
           - speech positive  -> 'happy'
           - speech neutral/None -> 'neutral'
      4) If no face:
           - speech negative  -> 'sad'
           - speech positive  -> 'happy'
           - speech neutral   -> 'neutral'
           - speech None      -> 'unknown'
      5) Fallback: 'unknown'.
    """
    f = normalize_face_emotion(face_emotion)
    s = normalize_speech_emotion(speech_emotion)

    # Case 1: face gives a strong non-neutral emotion → trust face
    if f in ["happy", "sad", "angry", "fear"]:
        return f

    # Case 2: face is explicitly neutral
    if f == "neutral":
        if s == "negative":
            return "sad"
        elif s == "positive":
            return "happy"
        elif s == "neutral" or s is None:
            return "neutral"

    # Case 3: no usable face, rely on speech only
    if f is None:
        if s == "negative":
            return "sad"
        elif s == "positive":
            return "happy"
        elif s == "neutral":
            return "neutral"
        else:
            return "unknown"

    # Fallback (should rarely happen)
    return "unknown"
