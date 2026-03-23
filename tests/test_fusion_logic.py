import os
import sys

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from fusion.fusion_logic import fuse_emotions


def run_tests():
    """
    Each test case:
      - name: description
      - face: raw face emotion (as DeepFace might output)
      - speech: raw speech emotion (as your speech module outputs)
      - expected: expected fused emotion
    """
    test_cases = [
        {
            "name": "face_happy_speech_negative_trust_face",
            "face": "happy",
            "speech": "negative",
            "expected": "happy",
        },
        {
            "name": "face_sad_speech_positive_trust_face",
            "face": "sad",
            "speech": "positive",
            "expected": "sad",
        },
        {
            "name": "face_angry_speech_negative_trust_face",
            "face": "angry",
            "speech": "negative",
            "expected": "angry",
        },
        {
            "name": "face_neutral_speech_negative_become_sad",
            "face": "neutral",
            "speech": "negative",
            "expected": "sad",
        },
        {
            "name": "face_neutral_speech_positive_become_happy",
            "face": "neutral",
            "speech": "positive",
            "expected": "happy",
        },
        {
            "name": "face_neutral_speech_neutral_stay_neutral",
            "face": "neutral",
            "speech": "neutral",
            "expected": "neutral",
        },
        {
            "name": "no_face_speech_negative_sad",
            "face": None,
            "speech": "negative",
            "expected": "sad",
        },
        {
            "name": "no_face_speech_positive_happy",
            "face": None,
            "speech": "positive",
            "expected": "happy",
        },
        {
            "name": "no_face_speech_neutral_neutral",
            "face": None,
            "speech": "neutral",
            "expected": "neutral",
        },
        {
            "name": "no_face_no_speech_unknown",
            "face": None,
            "speech": None,
            "expected": "unknown",
        },
        {
            "name": "face_disgust_speech_negative_map_disgust_to_neutral_then_negative_to_sad",
            "face": "disgust",
            "speech": "negative",
            "expected": "sad",
        },
        {
            "name": "face_surprise_speech_positive_map_surprise_to_neutral_then_positive_to_happy",
            "face": "surprise",
            "speech": "positive",
            "expected": "happy",
        },
    ]

    passed = 0
    failed = 0

    print("Running fusion logic tests...\n")

    for case in test_cases:
        name = case["name"]
        face = case["face"]
        speech = case["speech"]
        expected = case["expected"]

        result = fuse_emotions(face, speech)

        if result == expected:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        print(
            f"[{status}] {name}: "
            f"face={face!r}, speech={speech!r} -> fused={result!r}, expected={expected!r}"
        )

    print("\nSummary:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    total = passed + failed
    if total > 0:
        print(f"  Success rate: {passed}/{total} ({passed/total*100:.1f}%)")


if __name__ == "__main__":
    run_tests()
