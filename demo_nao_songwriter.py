
from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.core.message_python2 import AudioRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
from sic_framework.devices.common_naoqi.naoqi_leds import NaoLEDRequest
from sic_framework.devices.nao_stub import NaoStub
import time


from sic_framework.devices.common_naoqi.naoqi_motion import (
    NaoPostureRequest,
    NaoqiAnimationRequest,
)

import wave
import threading

from song_generation import instrumental_gen, download_song

class NaoSongwriterDemo(SICApplication):
    """Song gen and movements in paralllel"""

    def __init__(self, nao_ip="10.0.0.181", style="Latin Dance. 20 seconds."):
        super(NaoSongwriterDemo, self).__init__()
        self.nao_ip = nao_ip
        self.style = style
        self.nao = None
        self.wavefile = None
        self.samplerate = None
        self.audio_file = None
        self.stop_event = threading.Event()

        self.set_log_level(sic_logging.INFO)
        self.setup()

    def setup(self):
        """generate the song and feed it to the robot"""
        self.logger.info("generating instrumental")
        track = instrumental_gen(self.style)
        self.logger.info("downloading and conversion")
        self.audio_file = download_song(track)

        self.wavefile = wave.open(self.audio_file, "rb")
        self.samplerate = self.wavefile.getframerate()
        self.nao = Nao(ip=self.nao_ip)

    def play_audio(self):
        try:
            self.logger.info("Passing audio to nao")
            self.wavefile.rewind()
            data = self.wavefile.readframes(self.wavefile.getnframes())
            msg = AudioRequest(sample_rate=self.samplerate, waveform=bytes(data))
            self.nao.speaker.request(msg)
        except Exception as e:
            self.logger.error(f"ERROR!")

    def motion_routine(self, duration_sec=22):
        """Motion routine for demo of parrallelisation"""
        try:
            self.nao.motion.request(NaoPostureRequest("Stand", 0.7))
            t0 = time.time()
            while not self.stop_event.is_set() and (time.time() - t0) < duration_sec:
                self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Hey_1"))
                time.sleep(2.5)
                self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/You_1"))
                time.sleep(2.5)

            self.logger.info("Motion finished")
        except Exception as e:
            self.logger.error(f"Motion error: {e}")

    def led_routine(self, duration_sec=22):
        """Changes the leds"""
        try:
            self.logger.info("Starts LED")
            t0 = time.time()
            on = True
            while not self.stop_event.is_set() and (time.time() - t0) < duration_sec:
                self.nao.leds.request(NaoLEDRequest(group="FaceLeds", intensity=1.0 if on else 0.2, duration=0.2))
                on = not on
                time.sleep(0.3)
            self.logger.info("Finished LED")
        except Exception as e:
            self.logger.error(f"LED Error: {e}")

    def run(self):
        """Run with threading"""
        audio_t = threading.Thread(target=self.play_audio, name="audio-thread", daemon=True)
        motion_t = threading.Thread(target=self.motion_routine, name="motion-thread", daemon=True)
        leds_t  = threading.Thread(target=self.led_routine,  name="leds-thread",  daemon=True)

        try:
            self.logger.info("Starting run")
            audio_t.start()
            motion_t.start()
            leds_t.start()
            audio_t.join(timeout=30)
            motion_t.join(timeout=30)
            leds_t.join(timeout=30)
            
        finally:
            #stop background 
            self.stop_event.set()
            
            try:
                self.logger.info("Requesting safe posture and rest…")
                self.nao.motion.request(NaoPostureRequest("Stand", 0.5))
                self.nao.autonomous.request(NaoRestRequest())
            except Exception as e:
                self.logger.error(f"Error during shutdown posture: {e}")

            if self.wavefile:
                self.wavefile.close()
            self.logger.info("Shutting down application")
            self.shutdown()


if __name__ == "__main__":
    demo = NaoSongwriterDemo(
        nao_ip="10.0.0.181",
        style="Latin Dance. 20 seconds."
    )
    demo.run()
