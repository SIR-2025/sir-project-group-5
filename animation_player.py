"""
SIC NAO Animation Player (Posture-Aware)
- Plays gestures + BodyTalk animations
- Automatically switches between Stand / Sit postures
- Ignores failures
"""

import time

from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_motion import (
    NaoPostureRequest,
    NaoqiAnimationRequest,
)
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import (
    NaoqiTextToSpeechRequest,
)


def safe_say(nao, text):
    try:
        nao.tts.request(NaoqiTextToSpeechRequest(text))
    except Exception as e:
        print(f"  [TTS failed] {e}")


def safe_anim(nao, anim):
    """Try to play animation and report success/failure."""
    name = anim.split("/")[-1]
    print(f"  -> {name}")
    safe_say(nao, name)

    try:
        nao.motion.request(NaoqiAnimationRequest(anim))
        print("     ✔ OK")
    except Exception as e:
        print(f"     ✘ Failed: {e}")


def play_group(nao, posture, animations):
    """Set posture once and play all animations in the group."""
    print(f"\n=== Switching posture to {posture} ===")
    nao.motion.request(NaoPostureRequest(posture, 0.8))
    time.sleep(2.0)

    for anim in animations:
        safe_anim(nao, anim)
        time.sleep(2)


def main():
    nao_ip = "10.0.0.242"
    nao = Nao(ip=nao_ip)

    print("Connected to NAO.")

    # --- SIT BodyTalk ---
    bodytalk_sit = [
        "animations/Sit/BodyTalk/BodyTalk_1",
        "animations/Sit/BodyTalk/BodyTalk_10",
        "animations/Sit/BodyTalk/BodyTalk_11",
        "animations/Sit/BodyTalk/BodyTalk_12",
        "animations/Sit/BodyTalk/BodyTalk_2",
        "animations/Sit/BodyTalk/BodyTalk_3",
        "animations/Sit/BodyTalk/BodyTalk_4",
        "animations/Sit/BodyTalk/BodyTalk_5",
        "animations/Sit/BodyTalk/BodyTalk_6",
        "animations/Sit/BodyTalk/BodyTalk_7",
        "animations/Sit/BodyTalk/BodyTalk_8",
        "animations/Sit/BodyTalk/BodyTalk_9",
    ]

    # --- STAND BodyTalk ---
    bodytalk_stand = [
        "animations/Stand/BodyTalk/BodyTalk_1",
        "animations/Stand/BodyTalk/BodyTalk_10",
        "animations/Stand/BodyTalk/BodyTalk_11",
        "animations/Stand/BodyTalk/BodyTalk_12",
        "animations/Stand/BodyTalk/BodyTalk_13",
        "animations/Stand/BodyTalk/BodyTalk_14",
        "animations/Stand/BodyTalk/BodyTalk_15",
        "animations/Stand/BodyTalk/BodyTalk_16",
        "animations/Stand/BodyTalk/BodyTalk_17",
        "animations/Stand/BodyTalk/BodyTalk_18",
        "animations/Stand/BodyTalk/BodyTalk_19",
        "animations/Stand/BodyTalk/BodyTalk_2",
        "animations/Stand/BodyTalk/BodyTalk_20",
        "animations/Stand/BodyTalk/BodyTalk_21",
        "animations/Stand/BodyTalk/BodyTalk_22",
        "animations/Stand/BodyTalk/BodyTalk_3",
        "animations/Stand/BodyTalk/BodyTalk_4",
        "animations/Stand/BodyTalk/BodyTalk_5",
        "animations/Stand/BodyTalk/BodyTalk_6",
        "animations/Stand/BodyTalk/BodyTalk_7",
        "animations/Stand/BodyTalk/BodyTalk_8",
        "animations/Stand/BodyTalk/BodyTalk_9",
    ]

    # --- Gestures ---
    gestures = [
        "animations/Stand/Gestures/BowShort_1",
        "animations/Stand/Gestures/Enthusiastic_4",
        "animations/Stand/Gestures/Enthusiastic_5",
        "animations/Stand/Gestures/Explain_1",
        "animations/Stand/Gestures/Explain_10",
        "animations/Stand/Gestures/Explain_11",
        "animations/Stand/Gestures/Explain_2",
        "animations/Stand/Gestures/Explain_3",
        "animations/Stand/Gestures/Explain_4",
        "animations/Stand/Gestures/Explain_5",
        "animations/Stand/Gestures/Explain_6",
        "animations/Stand/Gestures/Explain_7",
        "animations/Stand/Gestures/Explain_8",
        "animations/Stand/Gestures/Hey_1",
        "animations/Stand/Gestures/Hey_6",
        "animations/Stand/Gestures/IDontKnow_1",
        "animations/Stand/Gestures/IDontKnow_2",
        "animations/Stand/Gestures/Me_1",
        "animations/Stand/Gestures/Me_2",
        "animations/Stand/Gestures/No_3",
        "animations/Stand/Gestures/No_8",
        "animations/Stand/Gestures/No_9",
        "animations/Stand/Gestures/Please_1",
        "animations/Stand/Gestures/Yes_1",
        "animations/Stand/Gestures/Yes_2",
        "animations/Stand/Gestures/Yes_3",
        "animations/Stand/Gestures/YouKnowWhat_1",
        "animations/Stand/Gestures/YouKnowWhat_5",
        "animations/Stand/Gestures/You_1",
        "animations/Stand/Gestures/You_4",
    ]

    # --- Play groups ---
    # play_group(nao, "Sit", bodytalk_sit)
    # play_group(nao, "Stand", bodytalk_stand)
    play_group(nao, "Stand", gestures)

    print("\nAll done!")


if __name__ == "__main__":
    main()