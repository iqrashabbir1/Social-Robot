import os
import sys
import csv
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


LOG_PATH = os.path.join(CURRENT_DIR, "multimodal_log.csv")
TMP_AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "speech", "tmp")


def capture_face_emotion() -> tuple[str | None, float | None]:
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

    # Optional: brief preview
    cv2.imshow("Captured Frame (press any key to close)", frame)
    cv2.waitKey(500)  # show for 0.5 sec
    cv2.destroyAllWindows()

    face_emotion, confidence = detect_face_emotion_from_frame(frame, enforce_detection=False)
    print(f"[Vision] Raw face_emotion={face_emotion}, confidence={confidence}")
    return face_emotion, confidence


def capture_speech_and_predict(tmp_dir: str = TMP_AUDIO_DIR) -> tuple[str | None, str | None]:
    """
    Record audio and predict speech emotion.
    Returns (speech_emotion, wav_path).
    """
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
            return None, None

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

    return speech_emotion, wav_path


def ensure_log_header():
    """
    Create the CSV log with header if it does not exist yet.
    """
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "face_emotion",
                "face_confidence",
                "speech_emotion",
                "fused_emotion",
                "response_text",
                "audio_file",
            ])
        print(f"Created new log file with header at: {LOG_PATH}")


def append_log_row(row: list[str]):
    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    ensure_log_header()

    print("=== Multimodal Emotion Demo (Interactive Loop) ===")
    print("Instructions:")
    print(" - Press Enter to run a new interaction (face + speech + response).")
    print(" - Type 'q' and press Enter to quit.\n")

    while True:
        cmd = input("Press Enter to start a new interaction, or 'q' then Enter to quit: ").strip().lower()
        if cmd == "q":
            print("Exiting.")
            break

        # 1) Capture face emotion
        face_emotion, face_conf = capture_face_emotion()

        # 2) Capture speech emotion
        speech_emotion, wav_path = capture_speech_and_predict()

        # 3) Fuse
        fused = fuse_emotions(face_emotion, speech_emotion)

        # 4) Generate response
        response_text = get_response_for_emotion(fused)

        # 5) Log
        ts = datetime.now().isoformat(timespec="seconds")
        row = [
            ts,
            str(face_emotion) if face_emotion is not None else "",
            str(face_conf) if face_conf is not None else "",
            str(speech_emotion) if speech_emotion is not None else "",
            fused,
            response_text,
            wav_path if wav_path is not None else "",
        ]
        append_log_row(row)

        # 6) Print summary
        print("\n=== Interaction Result ===")
        print(f"Time:           {ts}")
        print(f"Face emotion:   {face_emotion} (conf={face_conf})")
        print(f"Speech emotion: {speech_emotion}")
        print(f"Fused emotion:  {fused}")
        print(f"Response text:  {response_text}")

        # 7) Speak response
        print("Robot speaking...")
        speak(response_text)
        print("Interaction complete.\n")


if __name__ == "__main__":
    main()

