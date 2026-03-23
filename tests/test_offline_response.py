import os
import sys
import time

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from response.response_rules import get_response_for_emotion
from response.tts_engine import speak

def demo_offline_responses():
    # Simulated fused emotions coming from fusion.fusion_logic
    fused_emotions = [
        "happy",
        "sad",
        "angry",
        "fear",
        "neutral",
        "unknown",
        None,            # should map to 'unknown'
        "HAPPY",         # test case-insensitivity via normalize
        "something_else" # completely unknown, goes to 'unknown'
    ]

    print("Offline response + TTS demo.")
    print("The robot will speak one response for each fused emotion.\n")

    for i, fused in enumerate(fused_emotions, start=1):
        print(f"--- Case {i} ---")
        print(f"Fused emotion input: {fused!r}")

        resp = get_response_for_emotion(fused)
        print(f"Selected response text: {resp}")

        # Speak the text
        speak(resp)

        # Small pause between cases (optional)
        time.sleep(0.5)

    print("\nDemo completed.")

if __name__ == "__main__":
    demo_offline_responses()
