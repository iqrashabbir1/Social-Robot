import os
import sys

# Ensure project root on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from response.response_rules import get_response_for_emotion, normalize_fused_emotion


def run_tests():
    test_labels = [
        "happy",
        "sad",
        "angry",
        "fear",
        "neutral",
        "unknown",
        None,
        "HAPPY",         # uppercase
        "Fearful",       # alias
        "something_else" # completely unknown
    ]

    print("Testing response rules...\n")

    for lbl in test_labels:
        norm = normalize_fused_emotion(lbl) if lbl is not None else "None→unknown"
        resp = get_response_for_emotion(lbl)
        print(f"Input label: {lbl!r}  | normalized: {norm!r}")
        print(f"Response: {resp}")
        print("-" * 70)

    print("Done.")


if __name__ == "__main__":
    run_tests()
