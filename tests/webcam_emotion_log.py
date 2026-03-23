import cv2
from deepface import DeepFace
import csv
from datetime import datetime
import os

log_path = "tests/emotion_log.csv"
log_exists = os.path.exists(log_path)

log_file = open(log_path, mode="a", newline="", encoding="utf-8")
csv_writer = csv.writer(log_file)

if not log_exists:
    csv_writer.writerow(["timestamp", "emotion", "confidence"])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    log_file.close()
    exit()

frame_count = 0
last_emotion = "unknown"
last_confidence = 0.0
last_region = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        frame_count += 1

        if frame_count % 5 == 0:
            try:
                result = DeepFace.analyze(
                    img_path=frame,
                    actions=['emotion'],
                    enforce_detection=False
                )

                if isinstance(result, list):
                    result = result[0]

                last_emotion = result.get("dominant_emotion", "unknown")
                emotions_dict = result.get("emotion", {})
                if emotions_dict:
                    last_confidence = emotions_dict.get(last_emotion, 0.0)
                else:
                    last_confidence = 0.0

                last_region = result.get("region", None)

                ts = datetime.now().isoformat(timespec="seconds")
                csv_writer.writerow([ts, last_emotion, f"{last_confidence:.4f}"])
                log_file.flush()

            except Exception as e:
                print("DeepFace error:", e)

        # Draw bounding box if available
        if last_region is not None:
            x, y, w, h = last_region["x"], last_region["y"], last_region["w"], last_region["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text = f"{last_emotion} ({last_confidence:.2f})"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Webcam Emotion - Logging", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
