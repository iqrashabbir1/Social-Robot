import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Say something (5 seconds)...")
    r.adjust_for_ambient_noise(source, duration=1)
    audio = r.listen(source, timeout=5, phrase_time_limit=5)

with open("tests/sample_audio.wav", "wb") as f:
    f.write(audio.get_wav_data())

print("Saved to tests/sample_audio.wav")
