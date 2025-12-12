"""
NAO Voice Tester (SIC + NAOqi setVoice), interactive via stdin.

Commands:
  l  -> list available voices
  v  -> show current voice
  n  -> next voice (from test list)
  p  -> previous voice
  s  -> speak sample line
  q  -> quit
"""

from __future__ import annotations

import time
import qi

from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import (
    NaoRestRequest,
    NaoSetAutonomousLifeRequest,
    NaoWakeUpRequest,
)


class NaoVoiceTester(SICApplication):
    def __init__(self):
        super().__init__()
        self.set_log_level(sic_logging.INFO)

        self.nao_ip = "10.0.0.183"  # <- your NAO
        self.nao: Nao | None = None

        self._qi_session: qi.Session | None = None
        self._naoqi_tts = None

        self.available_voices: list[str] = []
        self.test_voices: list[str] = []
        self.voice_idx: int = 0

        self.setup()

    def setup(self):
        self.logger.info("Initializing NAO (SIC)...")
        self.nao = Nao(ip=self.nao_ip)

        try:
            self.nao.autonomous.request(NaoWakeUpRequest())
            time.sleep(0.2)
            self.nao.autonomous.request(NaoSetAutonomousLifeRequest("solitary"))
            time.sleep(0.2)
        except Exception:
            self.logger.exception("Could not wake up / set autonomous life (continuing).")

        self.logger.info("Connecting to NAOqi (qi.Session)...")
        self._qi_session = qi.Session()
        self._qi_session.connect(f"tcp://{self.nao_ip}:9559")
        self._naoqi_tts = self._qi_session.service("ALTextToSpeech")

        self.available_voices = list(self._naoqi_tts.getAvailableVoices())
        self.logger.info("NAOqi reports %d available voice(s). Type 'l' to list them.", len(self.available_voices))

        # Pick a small default test set (fallback to first 3 available)
        preferred = self.available_voices
        self.test_voices = [v for v in preferred if v in self.available_voices]
        if not self.test_voices:
            self.test_voices = self.available_voices[:3]

        if self.test_voices:
            self.voice_idx = 0
            self._apply_voice()

    def _apply_voice(self):
        voice = self.test_voices[self.voice_idx]
        self._naoqi_tts.setVoice(voice)
        self.logger.info("Selected voice: %s", voice)

    def _speak_sample(self):
        voice = self.test_voices[self.voice_idx]
        text = f"Testing voice {voice}. The quick brown fox jumps over the lazy dog."
        self.nao.tts.request(NaoqiTextToSpeechRequest(text))

    def shutdown_nao(self):
        self.logger.info("Shutting down...")
        try:
            self.nao.autonomous.request(NaoRestRequest())
        except Exception:
            self.logger.exception("NaoRestRequest failed.")

    def run(self):
        self.logger.info("Voice tester ready. Commands: l v n p s q")

        for v in self.available_voices:
            print(v)
            self._speak_sample()
            self.voice_idx = (self.voice_idx + 1) % len(self.test_voices)
            self._apply_voice()

        # try:
        #     while not self.shutdown_event.is_set():
        #         cmd = input("> ").strip().lower()
        #
        #         if cmd == "q":
        #             break
        #         elif cmd == "l":
        #             for v in self.available_voices:
        #                 print(v)
        #         elif cmd == "v":
        #             if self.test_voices:
        #                 print(self.test_voices[self.voice_idx])
        #             else:
        #                 print("No test voices configured.")
        #         elif cmd == "n":
        #             if self.test_voices:
        #                 self.voice_idx = (self.voice_idx + 1) % len(self.test_voices)
        #                 self._apply_voice()
        #         elif cmd == "p":
        #             if self.test_voices:
        #                 self.voice_idx = (self.voice_idx - 1) % len(self.test_voices)
        #                 self._apply_voice()
        #         elif cmd == "s":
        #             if self.test_voices:
        #                 self._speak_sample()
        #         else:
        #             print("Commands: l v n p s q")




if __name__ == "__main__":
    app = NaoVoiceTester()
    app.run()