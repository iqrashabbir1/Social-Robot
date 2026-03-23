from __future__ import annotations
import pyttsx3
from typing import Optional


def _create_engine(rate: int = 180, volume: float = 0.9) -> pyttsx3.Engine:
    """
    Create and configure a new TTS engine instance.
    """
    engine = pyttsx3.init()  # On Windows, uses SAPI5 by default

    # Set speech rate
    engine.setProperty("rate", rate)

    # Set volume
    engine.setProperty("volume", volume)

    # Optional: choose an English voice if available
    voices = engine.getProperty("voices")
    if voices:
        chosen_voice: Optional[str] = None
        for v in voices:
            name = v.name.lower()
            if "english" in name:
                chosen_voice = v.id
                break
        if chosen_voice is None:
            chosen_voice = voices[0].id
        engine.setProperty("voice", chosen_voice)

    return engine


def speak(text: str, rate: int = 180, volume: float = 0.9) -> None:
    """
    Speak the given text.
    This function:
      - creates a fresh engine,
      - speaks the text,
      - blocks until finished,
      - stops and discards the engine.
    """
    if not text:
        return

    engine = _create_engine(rate=rate, volume=volume)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
