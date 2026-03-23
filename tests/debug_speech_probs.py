import os
import sys
import uuid
import speech_recognition as sr

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from perception.speech_emotion import predict_speech_emotion_with_proba

TMP_AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "speech", "tmp_debug")


def record_one_clip(label_hint: str = "") -> str:
    os.makedirs(TMP_AUDIO_DIR, exist_ok=True)
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print(f"\n[{label_hint}] Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        print(f"[{label_hint}] Recording... speak for 3–5 seconds.")
        audio = r.listen(source, timeout=6, phrase_time_limit=5)

    fname = f"debug_{label_hint}_{uuid.uuid4().hex[:6]}.wav"
    wav_path = os.path.join(TMP_AUDIO_DIR, fname)
    with open(wav_path, "wb") as f:
        f.write(audio.get_wav_data())

    print(f"[{label_hint}] Saved to: {wav_path}")
    return wav_path


def main():
    print("Debug speech probabilities. We'll record 3 clips: positive, neutral, negative.")
    print("Try to REALLY exaggerate the emotion.\n")

    for label_hint in ["POS", "NEU", "NEG"]:
        wav = record_one_clip(label_hint)
        pred_label, proba = predict_speech_emotion_with_proba(wav)
        print(f"\n[{label_hint}] Predicted label: {pred_label}")
        print(f"[{label_hint}] Probabilities: {proba}")
        print("-" * 70)


if __name__ == "__main__":
    main()
