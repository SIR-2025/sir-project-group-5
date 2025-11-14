import os
import json
import time

from sic_framework.core import sic_logging
from sic_framework.core.sic_application import SICApplication
from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiAnimationRequest, NaoPostureRequest

from replicate_json_pose import Pose, replicate_pose  # adjust import if needed

class NaoTeacher(SICApplication):
    """
    NAO teacher demo application.
    Load a sequence of poses, make them and wait for the user to repeat before going to the next one.
    """

    def __init__(self):
        super(NaoTeacher, self).__init__()

        self.nao_ip = "10.0.0.181"
        self.nao = None

        self.set_log_level(sic_logging.INFO)

        self.setup()

    def setup(self):
        """Initialize and configure the NAO robot."""
        self.nao = Nao(ip=self.nao_ip)

    def run(self):
        """Main application logic."""
        try:
            self.nao.motion.request(NaoPostureRequest("Stand", 0.5))
            time.sleep(1)

            # Eyes red
            self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0, 0, 0.5))

            dir = os.path.dirname(__file__)
            pose_path = os.path.join(dir, "poses", "MyPose.json")

            with open(pose_path, "r") as f:
                data = json.load(f)

            pose = Pose(kp_img_norm=data["kp_img_norm"])

            replicate_pose(pose, self.nao_ip, mirror=True, duration=2.0)
            time.sleep(3)

            # Eyes green
            self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 1, 0, 0.5))
            time.sleep(1)

            self.nao.autonomous.request(NaoRestRequest())
        except Exception as e:
            self.logger.exception("Error in motion demo: %s", e)
        finally:
            self.logger.info("Shutting down application")
            try:
                self.nao.autonomous.request(NaoRestRequest())
            except Exception:
                self.logger.exception("Error while sending NaoRestRequest in finally")
            self.shutdown()

if __name__ == "__main__":
    app = NaoTeacher()
    app.run()