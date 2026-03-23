import os
import sys

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from response.tts_engine import speak

def main():
    print("Testing TTS... You should hear the robot speak.")
    speak("Hello, I am your emotion-aware assistant. I am ready to talk to you.")
    speak("This is a second sentence to test my voice.")

if __name__ == "__main__":
    main()
