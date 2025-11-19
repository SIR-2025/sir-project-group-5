#!/usr/bin/env python3
"""
NAO Teach Mode with Dialogflow CX (v3)
- Listens for commands ("start teaching", "stop teaching", "play music")
- Plays music and records movements
- Uses threads so NAO can listen, talk, and act simultaneously
"""

import threading
import json
import time
from os.path import abspath, join
import numpy as np

from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
from sic_framework.core.message_python2 import AudioRequest


from sic_framework.services.dialogflow_cx.dialogflow_cx import (
    DialogflowCX,
    DialogflowCXConf,
    DetectIntentRequest,
)

from nao_teacher_runner import run_teacher

class NaoTeachMode(SICApplication):
    def __init__(self):
        super(NaoTeachMode, self).__init__()
        self.set_log_level(sic_logging.INFO)
        self.nao_ip = "10.0.0.243"
        self.keyfile_path = abspath(join("conf", "google", "google-key.json"))

        self.nao = None
        self.dialogflow = None
        self.session_id = np.random.randint(10000)
        self.is_teaching = False
        self.is_playing = False
        self._lock = threading.Lock()

        self.setup()

    def setup(self):
        self.logger.info("Initializing NAO...")
        self.nao = Nao(ip=self.nao_ip)
        nao_mic = self.nao.mic

        self.logger.info("Initializing Dialogflow CX...")
        with open(self.keyfile_path) as f:
            keyfile_json = json.load(f)

        agent_id = "957eb40b-16d8-46c9-8eb7-1c6df539a161"
        location = "europe-west4"

        conf = DialogflowCXConf(
            keyfile_json=keyfile_json,
            agent_id=agent_id,
            location=location,
            sample_rate_hertz=16000,
            language="en"
        )

        self.dialogflow = DialogflowCX(conf=conf, input_source=nao_mic)
        self.dialogflow.register_callback(self.on_recognition)

        self.logger.info("Setup complete. Say 'start teaching' to begin.")

    # -------------------------------------------------------------------
    # DIALOGFLOW CALLBACK
    # -------------------------------------------------------------------
    def on_recognition(self, message):
        """Log speech transcript as it’s recognized."""
        if message.response:
            rr = getattr(message.response, "recognition_result", None)
            if rr and getattr(rr, "is_final", False):
                self.logger.info(f"User said: {rr.transcript}")

    # -------------------------------------------------------------------
    # INTENT HANDLER
    # -------------------------------------------------------------------
    def handle_intent(self, reply):
        """Act based on detected intent."""
        intent = reply.intent or ""
        text = reply.fulfillment_message or ""
        self.logger.info(f"Detected intent: {intent}")
        if text:
            self.nao.tts.request(NaoqiTextToSpeechRequest(text))

        if intent == "start_teaching":
            self.start_teaching_mode()

        elif intent == "stop_teaching":
            self.stop_teaching_mode()

        elif intent == "play_music":
            self.play_music()

    # -------------------------------------------------------------------
    # BEHAVIORS
    # -------------------------------------------------------------------
    def start_teaching_mode(self):
        with self._lock:
            if self.is_teaching:
                self.nao.tts.request(NaoqiTextToSpeechRequest("I'm already learning."))
                return
            self.is_teaching = True

        self.nao.tts.request(NaoqiTextToSpeechRequest("Okay, I'm watching you dance!"))
        threading.Thread(
            target=run_teacher,
            args=(self.nao, self.nao_ip),
            daemon=True
        ).start()


    def stop_teaching_mode(self):
        with self._lock:
            if not self.is_teaching:
                self.nao.tts.request(NaoqiTextToSpeechRequest("I'm not learning right now."))
                return
            self.is_teaching = False
        self.nao.tts.request(NaoqiTextToSpeechRequest("Got it! I learned your dance."))

    def play_music(self):
        if self.is_playing:
            return
        self.is_playing = True
        threading.Thread(target=self._play_music_thread, daemon=True).start()

    # -------------------------------------------------------------------
    # THREAD FUNCTIONS
    # -------------------------------------------------------------------

    def _play_music_thread(self):
        try:
            self.nao.tts.request(NaoqiTextToSpeechRequest("Starting the music!"))
            with open("dance_song.wav", "rb") as f:
                sound = f.read()
            self.nao.speaker.request(AudioRequest(sample_rate=44100, waveform=sound))
        finally:
            time.sleep(3)
            self.is_playing = False

    # -------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------
    def run(self):
        try:
            self.nao.tts.request(NaoqiTextToSpeechRequest("Hello, I'm ready to learn!"))
            while not self.shutdown_event.is_set():
                reply = self.dialogflow.request(DetectIntentRequest(self.session_id))
                if reply.intent:
                    self.handle_intent(reply)
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        finally:
            self.nao.autonomous.request(NaoRestRequest())
            self.shutdown()


if __name__ == "__main__":
    app = NaoTeachMode()
    app.run()
