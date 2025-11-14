import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

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

mp_pose = mp.solutions.pose


def _landmarks_to_array(landmarks) -> np.ndarray:
    """Convert MediaPipe landmarks to (33,3) normalized array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)


def _draw_pose_skeleton(frame: np.ndarray, kp_img_norm: np.ndarray) -> None:
    """Draw a simple white skeleton using normalized keypoints on a BGR frame."""
    h, w = frame.shape[:2]
    for a, b in BODY_CONNS:
        xa, ya = int(kp_img_norm[a, 0] * w), int(kp_img_norm[a, 1] * h)
        xb, yb = int(kp_img_norm[b, 0] * w), int(kp_img_norm[b, 1] * h)
        cv2.line(frame, (xa, ya), (xb, yb), (255, 255, 255), 2)
    for i in BODY_IDXS:
        x, y = int(kp_img_norm[i, 0] * w), int(kp_img_norm[i, 1] * h)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)


def _show_pose_window(pose: Pose, window_name: str = "NAO Current Pose") -> None:
    """Render the given pose into a window on the laptop."""
    h, w = 480, 360
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    _draw_pose_skeleton(frame, pose.kp_img_norm)
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _auto_pose_path(base_dir: str, idx: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"pose_{ts}_{idx:02d}.json")


def _record_poses_with_camera(
    logger,
    out_dir: str,
    duration: float = 30.0,
    sample_interval: float = 5.0,
    countdown: int = 3,
    camera_index: int = 0,
) -> list[Pose]:
    """
    Open the laptop camera, wait for SPACE, show a countdown, then record poses for `duration` seconds.
    Every `sample_interval` seconds, capture a pose and save it to JSON (only kp_img_norm).
    Returns a list of Pose objects corresponding to the saved poses.
    """
    _ensure_dir(out_dir)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Cannot open camera index {camera_index}")
        return []

    window_name = "Camera / Pose Recorder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    recorded_poses: list[Pose] = []
    logger.info("Press SPACE to start recording (30s). ESC to cancel.")

    try:
        recording = False
        countdown_end = None
        record_start = None
        next_sample = None
        pose_idx = 0

        with mp_pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as pose_detector:

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                disp = frame.copy()

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    logger.info("Recording aborted by user.")
                    break

                now = time.perf_counter()

                if not recording:
                    cv2.putText(
                        disp,
                        "SPACE: start recording (30s) | ESC: quit",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if key == 32:  # SPACE
                        recording = True
                        countdown_end = now + countdown
                        record_start = None
                        logger.info("Countdown started.")
                else:
                    if record_start is None:
                        remaining = max(0, int(countdown_end - now) + 1)
                        cv2.putText(
                            disp,
                            str(remaining),
                            (disp.shape[1] // 2 - 20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.5,
                            (0, 255, 255),
                            6,
                            cv2.LINE_AA,
                        )
                        if now >= countdown_end:
                            record_start = time.perf_counter()
                            next_sample = record_start
                            logger.info("Recording poses for 30 seconds...")
                    else:
                        elapsed = now - record_start
                        remaining = max(0.0, duration - elapsed)
                        cv2.putText(
                            disp,
                            f"REC {remaining:04.1f}s",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb.flags.writeable = False
                        res = pose_detector.process(rgb)
                        rgb.flags.writeable = True

                        if res.pose_landmarks:
                            kp = _landmarks_to_array(res.pose_landmarks)
                            _draw_pose_skeleton(disp, kp)

                            if now >= next_sample and elapsed <= duration:
                                pose_obj = Pose(kp_img_norm=kp)
                                recorded_poses.append(pose_obj)
                                out_path = _auto_pose_path(out_dir, pose_idx)
                                with open(out_path, "w") as f:
                                    json.dump({"kp_img_norm": kp.tolist()}, f, indent=2)
                                logger.info(f"Saved pose #{pose_idx} → {out_path}")
                                pose_idx += 1
                                next_sample += sample_interval

                        if elapsed > duration:
                            logger.info("Recording finished.")
                            break

                cv2.imshow(window_name, disp)

    finally:
        cap.release()
        cv2.destroyWindow(window_name)

    logger.info(f"Total recorded poses: {len(recorded_poses)}")
    return recorded_poses


class NaoTeacher(SICApplication):
    """
    NAO teacher demo application.
    1. NAO stands and opens the laptop camera.
    2. On SPACE: countdown + 30s recording; every 5s a pose is captured and saved.
    3. After recording, NAO replicates the recorded poses and the current target pose
       is displayed on the laptop.
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
        nao_pose_window = "NAO Current Pose"
        cv2.namedWindow(nao_pose_window, cv2.WINDOW_NORMAL)

        try:
            self.nao.motion.request(NaoPostureRequest("Stand", 0.5))
            time.sleep(1)

            # Eyes blue at start (ready to record)
            self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 0, 1, 0))

            pose_dir = os.path.join(os.path.dirname(__file__), "poses")
            self.logger.info(f"Recording poses into: {pose_dir}")

            recorded_poses = _record_poses_with_camera(
                logger=self.logger,
                out_dir=pose_dir,
                duration=30.0,
                sample_interval=5.0,
                countdown=3,
                camera_index=0,
            )

            if not recorded_poses:
                self.logger.info("No poses recorded; going to rest.")
                self.nao.autonomous.request(NaoRestRequest())
                return

            # After recording, change eyes color to indicate playback phase
            self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0.5, 0, 0.5))

            for pose in recorded_poses:
                _show_pose_window(pose, window_name=nao_pose_window)

                # Eyes red while moving to pose
                self.nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0, 0, 0))
                replicate_pose(pose, self.nao_ip, mirror=True, duration=2.0)

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