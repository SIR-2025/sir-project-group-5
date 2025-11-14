import json
import os
import time

import cv2
import numpy as np
from sic_framework.core import sic_logging
from sic_framework.core.sic_application import SICApplication
from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest
from sic_framework.devices.common_naoqi.naoqi_motion import (
    NaoPostureRequest,
)

from replicate_json_pose import Pose, replicate_pose

BODY_IDXS = list(range(11, 33))
BODY_CONNS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]


def _draw_pose_skeleton(frame: np.ndarray, kp_img_norm: np.ndarray) -> None:
    """Draw a simple white skeleton using normalized keypoints on a BGR frame."""
    h, w = frame.shape[:2]
    # bones
    for a, b in BODY_CONNS:
        xa, ya = int(kp_img_norm[a, 0] * w), int(kp_img_norm[a, 1] * h)
        xb, yb = int(kp_img_norm[b, 0] * w), int(kp_img_norm[b, 1] * h)
        cv2.line(frame, (xa, ya), (xb, yb), (255, 255, 255), 2)
    # joints
    for i in BODY_IDXS:
        x, y = int(kp_img_norm[i, 0] * w), int(kp_img_norm[i, 1] * h)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)


def _show_pose_window(pose: Pose, window_name: str = "NAO Current Pose") -> None:
    """Render the given pose into a window on the laptop."""
    h, w = 480, 360
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    _draw_pose_skeleton(frame, pose.kp_img_norm)
    cv2.imshow(window_name, frame)
    # small wait to refresh window; ignore key presses
    cv2.waitKey(1)


class NaoTeacher(SICApplication):
    """
    NAO teacher demo application.
    Load a sequence of poses, execute them on NAO,
    and show the current target pose in a window on the laptop.
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
        window_name = "NAO Current Pose"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            self.nao.motion.request(NaoPostureRequest("Stand", 0.5))
            time.sleep(1)

            # Eyes blue at start
            self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 0, 1, 0))

            poses = []
            pose_dir = os.path.join(os.path.dirname(__file__), "poses")
            self.logger.info(f"Loading poses from: {pose_dir}")

            for fname in sorted(os.listdir(pose_dir)):
                if fname.endswith(".json"):
                    path = os.path.join(pose_dir, fname)
                    with open(path, "r") as f:
                        data = json.load(f)
                    poses.append(Pose(kp_img_norm=data["kp_img_norm"]))
                    self.logger.info(f"Loaded pose: {fname}")

            for pose in poses:
                # Show the pose on the laptop
                _show_pose_window(pose, window_name=window_name)

                # Eyes red while moving to pose
                self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0, 0, 0))
                replicate_pose(pose, self.nao_ip, mirror=True, duration=2.0)

                # Hold briefly with pose and preview visible
                time.sleep(1)

                # Eyes green to indicate pose reached
                self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 1, 0, 0))
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
            cv2.destroyAllWindows()
            self.shutdown()


if __name__ == "__main__":
    app = NaoTeacher()
    app.run()