import os
import sys
import uuid
import time
import speech_recognition as sr

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

BASE_DIR = os.path.join(PROJECT_ROOT, "data", "user_ser")


def record_samples_for_class(label: str, n_samples: int = 20):
    target_dir = os.path.join(BASE_DIR, label)
    os.makedirs(target_dir, exist_ok=True)

    r = sr.Recognizer()

    print(f"\n=== Recording {n_samples} samples for class: {label} ===")

    for i in range(1, n_samples + 1):
        input(f"\n[{label} {i}/{n_samples}] Press Enter when you are ready to record...")

        with sr.Microphone() as source:
            print("  Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("  Recording... speak for about 3–4 seconds in the target emotion.")
            audio = r.listen(source, timeout=6, phrase_time_limit=4)

        fname = f"user_{label}_{uuid.uuid4().hex[:8]}.wav"
        wav_path = os.path.join(target_dir, fname)
        with open(wav_path, "wb") as f:
            f.write(audio.get_wav_data())

        print(f"  Saved: {wav_path}")
        time.sleep(0.5)


def main():
    print("User SER dataset recording.")
    print("You will record your own positive / neutral / negative samples.")
    print("Try to exaggerate the emotional expression so the model can learn clearly.\n")

    # You can change n_samples per class here
    n_pos = 20
    n_neu = 20
    n_neg = 20

    record_samples_for_class("positive", n_pos)
    record_samples_for_class("neutral", n_neu)
    record_samples_for_class("negative", n_neg)

    print("\nDone. Your files are in:", BASE_DIR)


if __name__ == "__main__":
    main()
