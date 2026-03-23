from deepface import DeepFace
import pprint

img_path = "tests/img.jpg"

result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)

# DeepFace may return a list in some versions
if isinstance(result, list):
    result = result[0]

print("\n--- Raw DeepFace result ---")
pprint.pprint(result)

print("\n--- Key fields ---")
print("Dominant emotion:", result.get("dominant_emotion"))
print("Emotion scores:", result.get("emotion"))
print("Face region:", result.get("region"))
