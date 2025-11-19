from __future__ import annotations

import json
import threading
import time
from os.path import abspath, join

import cv2
import numpy as np

from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
from sic_framework.core.message_python2 import CompressedImageMessage

from sic_framework.services.dialogflow_cx.dialogflow_cx import (
    DialogflowCX,
    DialogflowCXConf,
    DetectIntentRequest,
)

from runners.teacher_runner import run_teacher
from modules.replicate_json_pose import Pose

BODY_IDXS = list(range(11, 33))
BODY_CONNS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]


class NaoTeachMode(SICApplication):
    def __init__(self):
        super(NaoTeachMode, self).__init__()
        self.set_log_level(sic_logging.INFO)

        self.nao_ip = "10.0.0.245"  # <- adjust to your NAO IP
        self.keyfile_path = abspath(join("conf", "google", "google-key.json"))

        self.nao: Nao | None = None
        self.dialogflow: DialogflowCX | None = None
        self.session_id = int(np.random.randint(10000))

        self.is_teaching = False
        self._lock = threading.Lock()

        # latest NAO camera frame (stored as BGR for OpenCV)
        self._last_frame: np.ndarray | None = None

        # dialogflow thread control
        self._df_thread: threading.Thread | None = None
        self._df_stop = threading.Event()

        # overlay: recorded poses (list of kp_img_norm arrays) + active index
        self._overlay_poses: list[np.ndarray] = []
        self._active_pose_idx: int | None = None

        # live mediapipe skeleton while learning
        self._learning_kp: np.ndarray | None = None
        self._learning_active: bool = False

        self._overlay_lock = threading.Lock()

        self.setup()

    # ----------------------------------------------------------
    # Setup: NAO + camera + Dialogflow
    # ----------------------------------------------------------
    def setup(self):
        """Initialize NAO, register camera callback, and set up Dialogflow."""
        self.logger.info("Initializing NAO...")
        self.nao = Nao(ip=self.nao_ip)

        # register camera callback ONCE; SIC starts the component
        self.nao.top_camera.register_callback(self._on_image)

        self.logger.info("Initializing Dialogflow CX...")
        with open(self.keyfile_path) as f:
            keyfile_json = json.load(f)

        agent_id = "957eb40b-16d8-46c9-8eb7-1c6df539a161"  # your agent ID
        location = "europe-west4"

        conf_df = DialogflowCXConf(
            keyfile_json=keyfile_json,
            agent_id=agent_id,
            location=location,
            sample_rate_hertz=16000,
            language="en",
        )

        self.dialogflow = DialogflowCX(conf=conf_df, input_source=self.nao.mic)
        self.dialogflow.register_callback(self.on_recognition)

        self.logger.info("Setup complete. Say 'start teaching' to begin.")

    # ----------------------------------------------------------
    # Camera handling
    # ----------------------------------------------------------
    def _on_image(self, msg: CompressedImageMessage):
        """
        Camera callback: keep only latest frame.

        SIC / NAO give RGB; we convert here once to BGR and store that,
        we treat frames as standard OpenCV BGR.
        """
        img = msg.image
        if img is None:
            return
        self._last_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def get_latest_frame(self) -> np.ndarray | None:
        """Return latest camera frame as BGR, or None."""
        return self._last_frame

    # ----------------------------------------------------------
    # Overlay helpers (poses + live learning skeleton)
    # ----------------------------------------------------------
    def _on_pose_saved(self, pose: Pose, idx: int) -> None:
        """Called by teacher module each time a pose is recorded."""
        with self._overlay_lock:
            self._overlay_poses.append(pose.kp_img_norm.copy())

    def _on_pose_start(self, idx: int) -> None:
        """Called by teacher module when a pose is about to be executed."""
        with self._overlay_lock:
            self._active_pose_idx = idx

    def _on_learning_frame_kp(self, kp: np.ndarray | None) -> None:
        """Called by teacher module each processed frame during recording."""
        with self._overlay_lock:
            self._learning_kp = kp

    def _draw_pose_skeleton_in_roi(
        self,
        roi: np.ndarray,
        kp_img_norm: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        """Draw skeleton into a ROI using normalized image coordinates."""
        h, w = roi.shape[:2]
        for a, b in BODY_CONNS:
            xa, ya = int(kp_img_norm[a, 0] * w), int(kp_img_norm[a, 1] * h)
            xb, yb = int(kp_img_norm[b, 0] * w), int(kp_img_norm[b, 1] * h)
            cv2.line(roi, (xa, ya), (xb, yb), color, 2)
        for i in BODY_IDXS:
            x, y = int(kp_img_norm[i, 0] * w), int(kp_img_norm[i, 1] * h)
            cv2.circle(roi, (x, y), 3, color, -1)

    def _draw_pose_thumbnails(self, frame: np.ndarray) -> None:
        """
        Draw saved poses as small overlays at the bottom of the camera view.
        The currently executed pose is drawn in green.
        """
        h, w = frame.shape[:2]

        with self._overlay_lock:
            poses = list(self._overlay_poses)
            active_idx = self._active_pose_idx

        if not poses:
            return

        thumb_h = int(h * 0.20)
        thumb_w = int(w * 0.16)
        margin = 8
        max_per_row = max(1, (w - margin) // (thumb_w + margin))
        max_rows = 2

        for idx, kp in enumerate(poses):
            row = idx // max_per_row
            col = idx % max_per_row
            if row >= max_rows:
                break

            x0 = margin + col * (thumb_w + margin)
            y0 = h - (row + 1) * (thumb_h + margin)
            y1 = y0 + thumb_h
            x1 = x0 + thumb_w
            if y0 < 0 or x1 > w:
                continue

            roi = frame[y0:y1, x0:x1]

            overlay = roi.copy()
            overlay[:] = (0, 0, 0)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

            color = (0, 255, 0) if active_idx == idx else (255, 255, 255)
            self._draw_pose_skeleton_in_roi(roi, kp, color)

            label = f"P{idx + 1}"
            cv2.putText(
                frame,
                label,
                (x0 + 5, y0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_learning_skeleton(self, frame: np.ndarray) -> None:
        """
        Draw the live MediaPipe skeleton for the last processed frame
        during the learning phase over the camera frame.
        """
        with self._overlay_lock:
            kp = self._learning_kp

        if kp is None:
            return

        kp_draw = kp.copy()
        kp_draw[:, 0] = 1.0 - kp_draw[:, 0]

        self._draw_pose_skeleton_in_roi(frame, kp_draw, (255, 255, 255))

    # ----------------------------------------------------------
    # Dialogflow recognition callback (partial results)
    # ----------------------------------------------------------
    def on_recognition(self, message):
        if message.response:
            rr = getattr(message.response, "recognition_result", None)
            if rr and getattr(rr, "is_final", False):
                self.logger.info(f"User said: {rr.transcript}")

    # ----------------------------------------------------------
    # Dialogflow intent polling loop (runs in background thread)
    # ----------------------------------------------------------
    def _dialogflow_loop(self):
        """Continuously call Dialogflow without blocking the GUI/camera."""
        try:
            while not self._df_stop.is_set():
                reply = self.dialogflow.request(DetectIntentRequest(self.session_id))
                if reply and reply.intent:
                    self.handle_intent(reply)
                time.sleep(0.05)
        except Exception as e:
            self.logger.exception("Dialogflow loop crashed: %s", e)

    # ----------------------------------------------------------
    # Intent handler
    # ----------------------------------------------------------
    def handle_intent(self, reply):
        intent = reply.intent or ""
        text = reply.fulfillment_message or ""
        self.logger.info(f"Detected intent: {intent}")
        if text:
            self.nao.tts.request(NaoqiTextToSpeechRequest(text))

        if intent == "start_teaching":
            self.start_teaching_mode()
        elif intent == "stop_teaching":
            self.stop_teaching_mode()
        # Add other intents here (play_music, etc.)

    # ----------------------------------------------------------
    # Behaviors
    # ----------------------------------------------------------
    def start_teaching_mode(self):
        with self._lock:
            if self.is_teaching:
                self.nao.tts.request(
                    NaoqiTextToSpeechRequest("I'm already learning.")
                )
                return
            self.is_teaching = True

        self.nao.tts.request(
            NaoqiTextToSpeechRequest("Okay, I'm watching you dance!")
        )

        def _teacher_thread():
            try:
                with self._overlay_lock:
                    self._overlay_poses = []
                    self._active_pose_idx = None
                    self._learning_kp = None
                    self._learning_active = True

                run_teacher(
                    nao=self.nao,
                    nao_ip=self.nao_ip,
                    frame_provider=self.get_latest_frame,
                    logger=self.logger,
                    on_pose_saved=self._on_pose_saved,
                    on_pose_start=self._on_pose_start,
                    on_kp_frame=self._on_learning_frame_kp,
                )
            finally:
                with self._overlay_lock:
                    self._learning_active = False
                    self._learning_kp = None
                    self._active_pose_idx = None
                with self._lock:
                    self.is_teaching = False

        threading.Thread(target=_teacher_thread, daemon=True).start()

    def stop_teaching_mode(self):
        # todo
        self.nao.tts.request(
            NaoqiTextToSpeechRequest("Got it! I learned your dance.")
        )

    # ----------------------------------------------------------
    # Main loop: show camera + run DF thread
    # ----------------------------------------------------------
    def run(self):
        camera_window = "NAO Camera"
        cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(camera_window, 720, 720)

        # Start Dialogflow in background thread
        self._df_stop.clear()
        self._df_thread = threading.Thread(
            target=self._dialogflow_loop,
            daemon=True,
        )
        self._df_thread.start()

        try:
            self.nao.tts.request(
                NaoqiTextToSpeechRequest("Hello, I'm ready to learn!")
            )

            while not self.shutdown_event.is_set():
                frame = self.get_latest_frame()
                if frame is not None:
                    frame_bgr = np.ascontiguousarray(frame)
                    frame_bgr = cv2.flip(frame_bgr, 1)

                    if self._learning_active:
                        self._draw_learning_skeleton(frame_bgr)

                    self._draw_pose_thumbnails(frame_bgr)

                    cv2.imshow(camera_window, frame_bgr)
                    cv2.waitKey(1)

                time.sleep(0.01)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        finally:
            self._df_stop.set()
            if self._df_thread is not None:
                self._df_thread.join(timeout=2.0)

            try:
                self.nao.autonomous.request(NaoRestRequest())
            except Exception:
                self.logger.exception("Error while sending NaoRestRequest in finally")

            cv2.destroyAllWindows()
            self.shutdown()


if __name__ == "__main__":
    app = NaoTeachMode()
    app.run()