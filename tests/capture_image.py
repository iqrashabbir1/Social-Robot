import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Failed to grab frame")
    exit()

cv2.imwrite("tests/img.jpg", frame)
print("Saved tests/img.jpg")
