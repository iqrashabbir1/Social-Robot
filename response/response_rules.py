from __future__ import annotations
from typing import Dict, List
import random

# Library of empathetic responses per fused emotion.
# You can tune wording later for your target users (elderly, kids, etc.).
RESPONSES: Dict[str, List[str]] = {
    "happy": [
        "You look happy right now. That makes me glad!",
        "I can sense some joy in you. Do you want to share what made you happy?",
        "You seem in a good mood. It's nice to see you like this.",
        "You sound cheerful! I’m happy to be here with you.",
    ],
    "sad": [
        "You seem a bit down. I’m here for you if you’d like to talk.",
        "I sense some sadness. Do you want to tell me what’s bothering you?",
        "It’s okay to feel sad sometimes. You are not alone.",
        "You look upset. Let’s take things slowly together.",
    ],
    "angry": [
        "I sense some frustration. Do you want to tell me what happened?",
        "You seem angry. It might help to talk about it.",
        "It’s okay to feel annoyed. I’m listening if you want to share.",
        "You look tense. We can pause for a moment and take a deep breath.",
    ],
    "fear": [
        "You seem worried. I’m here with you.",
        "I sense some anxiety. Do you want to tell me what is making you nervous?",
        "It’s okay to feel scared. We can take it one step at a time.",
        "You look concerned. I’m here to support you.",
    ],
    "neutral": [
        "I’m here if you want to talk or do something together.",
        "You seem calm. Is there anything you would like to do now?",
        "I don’t sense a strong emotion right now. How are you feeling?",
        "We can chat, relax, or do an activity whenever you like.",
    ],
    "unknown": [
        "I’m not sure how you feel right now, but I’m here with you.",
        "I’m having trouble understanding your emotion, but you can always talk to me.",
        "Even if I can’t read your feelings clearly, I’m here to support you.",
        "I may not fully understand your mood, but you are not alone.",
    ],
}


def normalize_fused_emotion(label: str | None) -> str:
    """
    Ensure the fused emotion is one of the known keys in RESPONSES.
    Anything unexpected maps to 'unknown'.
    """
    if label is None:
        return "unknown"

    raw = label.strip().lower()
    if raw in RESPONSES:
        return raw

    # Accept some aliases if you ever use them
    if raw in ["fearful", "scared"]:
        return "fear"

    return "unknown"


def get_response_for_emotion(fused_emotion: str | None) -> str:
    """
    Given a fused emotion label, return one empathetic response sentence.
    If the label is unknown or None, use 'unknown' category.
    """
    key = normalize_fused_emotion(fused_emotion)
    options = RESPONSES.get(key, RESPONSES["unknown"])

    if not options:
        # Should not happen, but be safe
        return "I’m here for you, even if I cannot understand your emotion right now."

    # Randomly select one response to avoid repetition
    return random.choice(options)
