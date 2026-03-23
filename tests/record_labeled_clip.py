import os
import uuid
import speech_recognition as sr

BASE_DIR = "data/speech/raw"

os.makedirs(BASE_DIR, exist_ok=True)

r = sr.Recognizer()

print("Label options (type one of these): positive, neutral, negative")
label = input("Enter label for this recording: ").strip().lower()

if label not in ["positive", "neutral", "negative"]:
    print("Invalid label. Use positive/neutral/negative.")
    exit()

# Unique filename
fname = f"{label}_{uuid.uuid4().hex[:8]}.wav"
path = os.path.join(BASE_DIR, fname)

with sr.Microphone() as source:
    print("Adjusting for ambient noise...")
    r.adjust_for_ambient_noise(source, duration=1)
    print("Recording... Speak now (3–5 seconds).")
    audio = r.listen(source, timeout=6, phrase_time_limit=5)

# Save wav
with open(path, "wb") as f:
    f.write(audio.get_wav_data())

print(f"Saved: {path}")
