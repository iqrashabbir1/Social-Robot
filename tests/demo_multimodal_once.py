import os
import sys
from datetime import datetime
import uuid
import cv2
import speech_recognition as sr

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from perception.face_emotion import detect_face_emotion_from_frame
from perception.speech_emotion import predict_speech_emotion_robust_3class
from fusion.fusion_logic import fuse_emotions
from response.response_rules import get_response_for_emotion
from response.tts_engine import speak


def capture_face_emotion() -> tuple[str | None, float | None]:
    """
    Capture one frame from webcam and detect face emotion.
    Returns (face_emotion, confidence).
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None, None

    print("Opening webcam... please look at the camera and hold your expression.")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to capture frame from webcam.")
        return None, None

    # Optional: show captured frame for debugging (press a key to close)
    cv2.imshow("Captured Frame (press any key to close)", frame)
    cv2.waitKey(1000)  # show for 1 second
    cv2.destroyAllWindows()

    face_emotion, confidence = detect_face_emotion_from_frame(frame, enforce_detection=False)
    print(f"[Vision] Raw face_emotion={face_emotion}, confidence={confidence}")
    return face_emotion, confidence


def capture_speech_and_predict(tmp_dir: str = None) -> str | None:
    """
    Record 3-5 seconds of audio from microphone, save to a temp wav file,
    and run predict_speech_emotion on it.
    Returns predicted speech emotion ('positive'/'neutral'/'negative') or None on failure.
    """
    if tmp_dir is None:
        tmp_dir = os.path.join(PROJECT_ROOT, "data", "speech", "tmp")

    os.makedirs(tmp_dir, exist_ok=True)

    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        print("Recording... please speak for about 3–5 seconds.")
        try:
            audio = r.listen(source, timeout=6, phrase_time_limit=5)
        except Exception as e:
            print("Error recording audio:", e)
            return None

    # Save temporary wav
    fname = f"demo_{uuid.uuid4().hex[:8]}.wav"
    wav_path = os.path.join(tmp_dir, fname)
    with open(wav_path, "wb") as f:
        f.write(audio.get_wav_data())

    print(f"[Speech] Saved temporary audio to: {wav_path}")

    try:
        speech_emotion = predict_speech_emotion_robust_3class(wav_path)
        print(f"[Speech] Robust 3-class speech emotion: {speech_emotion}")
    except Exception as e:
        print("Error in predict_speech_emotion:", e)
        speech_emotion = None

    return speech_emotion


def main():
    print("=== Multimodal Emotion Demo (One-Shot) ===")

    # 1) Capture face emotion
    face_emotion, face_conf = capture_face_emotion()

    # 2) Capture speech emotion
    speech_emotion = capture_speech_and_predict()

    # 3) Fuse both
    fused = fuse_emotions(face_emotion, speech_emotion)

    # 4) Generate empathetic response
    response_text = get_response_for_emotion(fused)

    # Log to console
    ts = datetime.now().isoformat(timespec="seconds")
    print("\n=== Result ===")
    print(f"Time:           {ts}")
    print(f"Face emotion:   {face_emotion} (conf={face_conf})")
    print(f"Speech emotion: {speech_emotion}")
    print(f"Fused emotion:  {fused}")
    print(f"Response text:  {response_text}")

    # 5) Speak response
    print("\nRobot speaking...")
    speak(response_text)


if __name__ == "__main__":
    main()
