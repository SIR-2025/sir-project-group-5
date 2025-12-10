"""
Interactive NAO teaching application.

This script runs a SICApplication that:

- Connects to a NAO robot and its camera.
- Streams audio to Dialogflow CX and reacts to detected intents.
- Runs a "teaching" routine on intent, recording poses via
  MediaPipe and replaying them on NAO.
- Runs a "learning" routine on intent, where NAO demonstrates
  stored poses and waits up to 15 seconds for imitation before
  going to the next pose.
- Shows a live OpenCV window with:
    * NAO camera view.
    * Live learning skeleton overlay.
    * Thumbnails of saved poses, highlighting the one currently replayed.
"""
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
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import (
    NaoRestRequest,
    NaoSetAutonomousLifeRequest,
    NaoWakeUpRequest,
)
from sic_framework.core.message_python2 import CompressedImageMessage
from sic_framework.devices.common_naoqi.naoqi_motion import (
    NaoqiAnimationRequest,
    NaoPostureRequest,
)
from sic_framework.services.dialogflow_cx.dialogflow_cx import (
    DialogflowCX,
    DialogflowCXConf,
    DetectIntentRequest,
)
from sic_framework.devices.common_naoqi.naoqi_microphone import (
    NaoqiMicrophone,  # Keep this for regular mic usage
)
import qi  # Add this import for direct NAOqi access

from runners.teacher_runner import run_teacher
from runners.learner_runner import run_learner
from runners.song_runner import run_song
from modules.replicate_json_pose import Pose

INTENTS = [
    "shut-down",
    "Default Welcome Intent",
    "generate_a_song",
    "dance_with_us",
    "nao_wants_to_learn",
    "nao_learning_completed",
    "user_wants_to_learn",
    "nao_conversation_repeat",
    "user_learns_dance",
    "nao_check_dance",
    "nao_dance_completed_check",
    "user_thanks",
    "nao_bye",
]

# MediaPipe-style body indices and connections for drawing skeletons.
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
    """SICApplication that lets NAO learn and replay dance poses."""

    def __init__(self):
        """Create the application and perform basic setup."""
        super(NaoTeachMode, self).__init__()
        self.set_log_level(sic_logging.INFO)

        self.nao_ip = "10.0.0.181"  # <- adjust to your NAO IP
        self.keyfile_path = abspath(join("conf", "google", "google-key.json"))

        self.nao: Nao | None = None
        self.dialogflow: DialogflowCX | None = None
        self.session_id = int(np.random.randint(10000))

        self.is_teaching = False
        self.is_learning = False
        self._lock = threading.Lock()

        # latest NAO camera frame (stored as BGR for OpenCV)
        self._last_frame: np.ndarray | None = None

        # dialogflow thread control
        self._df_thread: threading.Thread | None = None
        self._df_stop = threading.Event()

        # main loop control
        self._should_exit = threading.Event()

        # overlay: recorded poses (list of kp_img_norm arrays) + active index
        self._overlay_poses: list[np.ndarray] = []
        self._active_pose_idx: int | None = None

        # poses taught in the current session
        self._taught_poses: list[Pose] = []

        # live mediapipe skeleton while teaching/learning
        self._learning_kp: np.ndarray | None = None
        self._learning_active: bool = False

        self._overlay_lock = threading.Lock()

        self.last_intent = None

        # TTS lock to prevent microphone interference
        self._tts_lock = threading.Lock()

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

        # Put NAO in alive mode for human-like behavior
        try:
            self.logger.info("Setting NAO to alive mode...")
            self.nao.autonomous.request(NaoWakeUpRequest())
            time.sleep(0.3)
            # Set autonomous life to "solitary" for basic alive behavior
            self.nao.autonomous.request(NaoSetAutonomousLifeRequest("solitary"))
            time.sleep(0.5)
        except Exception as e:
            self.logger.exception("Error setting alive mode: %s", e)

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

        self.logger.info(
            "Setup complete. Say 'start teaching' to record a dance, "
            "or 'start learning' to practice the dance together."
        )

    # ----------------------------------------------------------
    # Camera handling
    # ----------------------------------------------------------
    def _on_image(self, msg: CompressedImageMessage):
        """Camera callback: store latest frame as BGR for OpenCV.

        SIC / NAO provide RGB images. We convert once to BGR and keep only
        the latest frame, which is then used by the GUI and teacher module.
        """
        img = msg.image
        if img is None:
            return
        self._last_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def get_latest_frame(self) -> np.ndarray | None:
        """Return the latest camera frame as BGR, or None if unavailable."""
        return self._last_frame

    # ----------------------------------------------------------
    # Overlay helpers (poses + live learning skeleton)
    # ----------------------------------------------------------
    def _on_pose_saved(self, pose: Pose, idx: int) -> None:
        """Store kp_img_norm for overlay every time a pose is recorded."""
        with self._overlay_lock:
            self._taught_poses.append(pose)  # keep Pose objects
            self._overlay_poses.append(pose.kp_img_norm.copy())

    def _on_pose_start(self, idx: int) -> None:
        """Update index of the pose that is currently being replayed / demonstrated."""
        with self._overlay_lock:
            self._active_pose_idx = idx

    def _on_pose_learned(self, idx: int) -> None:
        """Optional callback when a pose is successfully imitated."""
        try:
            self.nao.tts.request(
                NaoqiTextToSpeechRequest(f"Nice! You matched pose number {idx + 1}.")
            )
        except Exception:
            self.logger.exception("Error while giving TTS feedback on pose learned")

    def _on_learning_frame_kp(self, kp: np.ndarray | None) -> None:
        """Update the live learning skeleton for the last processed frame."""
        with self._overlay_lock:
            self._learning_kp = kp

    def _draw_pose_skeleton_in_roi(
        self,
        roi: np.ndarray,
        kp_img_norm: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a pose skeleton into a given ROI using normalized coordinates.

        Args:
            roi: Image region (BGR) to draw into.
            kp_img_norm: (33, 2/3) normalized keypoints in [0, 1].
            color: BGR color for lines and joints.
        """
        h, w = roi.shape[:2]
        for a, b in BODY_CONNS:
            xa, ya = int(kp_img_norm[a, 0] * w), int(kp_img_norm[a, 1] * h)
            xb, yb = int(kp_img_norm[b, 0] * w), int(kp_img_norm[b, 1] * h)
            cv2.line(roi, (xa, ya), (xb, yb), color, 2)
        for i in BODY_IDXS:
            x, y = int(kp_img_norm[i, 0] * w), int(kp_img_norm[i, 1] * h)
            cv2.circle(roi, (x, y), 3, color, -1)

    def _draw_pose_thumbnails(self, frame: np.ndarray) -> None:
        """Draw saved poses as small overlays at the bottom of the camera view.

        The currently executed pose (if any) is highlighted in green.
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

    # ---------- Skeleton drawing: teaching vs learning modes ----------

    def _draw_learning_skeleton_white(self, frame: np.ndarray) -> None:
        """Draw live skeleton while NAO is LEARNING (you teach NAO).

        Everything is white, like the original behavior.
        """
        with self._overlay_lock:
            kp = self._learning_kp

        if kp is None:
            return

        # Mirror horizontally to match flipped camera view
        kp_draw = kp.copy()
        kp_draw[:, 0] = 1.0 - kp_draw[:, 0]

        self._draw_pose_skeleton_in_roi(frame, kp_draw, (255, 255, 255))

    def _draw_teaching_skeleton_feedback(self, frame: np.ndarray) -> None:
        """Draw live skeleton while NAO is TEACHING (you imitate NAO).

        Joints close to the current target pose are drawn in green,
        others in white.
        """
        with self._overlay_lock:
            kp = self._learning_kp
            active_idx = self._active_pose_idx
            taught_poses = list(self._taught_poses)

        if kp is None:
            return

        if active_idx is None or active_idx < 0 or active_idx >= len(taught_poses):
            # Fallback: plain white if something is off
            kp_draw = kp.copy()
            kp_draw[:, 0] = 1.0 - kp_draw[:, 0]
            self._draw_pose_skeleton_in_roi(frame, kp_draw, (255, 255, 255))
            return

        live_kp = kp.copy()
        target_kp = taught_poses[active_idx].kp_img_norm

        # Per-joint distance in normalized xy space
        diffs = np.linalg.norm(live_kp[:, :2] - target_kp[:, :2], axis=1)
        joint_threshold = 0.10  # tweak as needed
        joint_match = diffs < joint_threshold  # bool array (33,)

        # Mirror for drawing (camera is flipped)
        kp_draw = live_kp.copy()
        kp_draw[:, 0] = 1.0 - kp_draw[:, 0]

        h, w = frame.shape[:2]

        # Bones in light gray
        for a, b in BODY_CONNS:
            xa, ya = int(kp_draw[a, 0] * w), int(kp_draw[a, 1] * h)
            xb, yb = int(kp_draw[b, 0] * w), int(kp_draw[b, 1] * h)
            cv2.line(frame, (xa, ya), (xb, yb), (200, 200, 200), 1)

        # Joints: green if matched, white otherwise
        for i in BODY_IDXS:
            x, y = int(kp_draw[i, 0] * w), int(kp_draw[i, 1] * h)
            color = (0, 255, 0) if joint_match[i] else (255, 255, 255)
            cv2.circle(frame, (x, y), 4, color, -1)

    # ----------------------------------------------------------
    # Dialogflow recognition callback (partial results)
    # ----------------------------------------------------------
    def on_recognition(self, message):
        """Handle streaming recognition results from Dialogflow (speech)."""
        if message.response:
            rr = getattr(message.response, "recognition_result", None)
            if rr and getattr(rr, "is_final", False):
                self.logger.info(f"[{self.last_intent}] User said: {rr.transcript}")

    # ----------------------------------------------------------
    # Dialogflow intent polling loop (runs in background thread)
    # ----------------------------------------------------------
    def _dialogflow_loop(self):
        """Continuously poll Dialogflow CX for final intents in a thread."""
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
        """Handle detected intent by triggering behaviors and TTS."""
        intent = getattr(reply, "intent", "") or ""
        text = getattr(reply, "fulfillment_message", "") or ""

        self.logger.info(f"Fulfillment message: {repr(text)}")

        # Dialogflow handles TTS via fulfillment messages
        if text:
            self.nao.tts.request(
                NaoqiTextToSpeechRequest(text)
            )

        match intent:
            case "generate_a_song":
                self.start_generating_song_mode()
            case "nao_wants_to_learn":
                self.start_teaching_mode()
            case "user_learns_dance":
                self.start_learning_mode()
            case "nao_bye":
                self.nao.motion.request(
                    NaoqiAnimationRequest("animations/Stand/Gestures/BowShort_3"),
                    block=True,
                )
            case _:
                self.logger.info(f"Intent '{intent}' not mapped to any behavior.")

        if intent in INTENTS:
            self.last_intent = intent

    # ----------------------------------------------------------
    # Behaviors
    # ----------------------------------------------------------
    def start_teaching_mode(self):
        """Start the teaching routine in a background thread if not active.

        In this mode the HUMAN teaches NAO a dance.
        Overlay skeleton: white (no per-joint feedback).
        """
        with self._lock:
            if self.is_teaching or self.is_learning:
                self.nao.tts.request(
                    NaoqiTextToSpeechRequest("I'm already busy with a dance.")
                )
                return
            self.is_teaching = True

        def _teacher_thread():
            """Worker thread that runs the teacher pipeline and updates overlays."""
            try:
                with self._overlay_lock:
                    self._overlay_poses = []
                    self._taught_poses = []  # clear poses of this session
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
                # Return to alive mode after teaching
                try:
                    self.logger.info("Returning NAO to alive mode...")
                    self.nao.motion.request(NaoPostureRequest("Stand", 0.5))
                    time.sleep(0.5)
                    self.nao.autonomous.request(
                        NaoSetAutonomousLifeRequest("solitary")
                    )
                except Exception as e:
                    self.logger.exception("Error returning to alive mode: %s", e)

                with self._overlay_lock:
                    self._learning_active = False
                    self._learning_kp = None
                    self._active_pose_idx = None
                with self._lock:
                    self.is_teaching = False

        threading.Thread(target=_teacher_thread, daemon=True).start()

    def start_generating_song_mode(self):
        """Start the song generation routine in a background thread if not active."""
        with self._lock:
            if self.is_teaching or self.is_learning:
                self.nao.tts.request(
                    NaoqiTextToSpeechRequest("I'm already busy with a dance.")
                )
                return
            self.is_learning = True  # mark busy (reuse flag)

        def _song_thread():
            """Worker thread that runs the song generator pipeline."""
            try:
                # Clear overlays; this mode does not need skeleton drawing
                with self._overlay_lock:
                    self._active_pose_idx = None
                    self._learning_kp = None
                    self._learning_active = False

                run_song(
                    nao=self.nao,
                    dialogflow_cx=self.dialogflow,
                    session_id=int(self.session_id),
                    logger=self.logger,
                    nao_ip=self.nao_ip,
                )

                self.nao.tts.request(
                    NaoqiTextToSpeechRequest(
                        "Your song is ready! I hope you enjoyed it!"
                    )
                )

            finally:
                with self._lock:
                    self.is_learning = False

        threading.Thread(target=_song_thread, daemon=True).start()

    def start_learning_mode(self):
        """Start the learning routine in a background thread if not active.

        In this mode NAO teaches the HUMAN a dance.
        Overlay skeleton: green joints when matching the current pose.
        """
        with self._lock:
            if self.is_teaching or self.is_learning:
                self.nao.tts.request(
                    NaoqiTextToSpeechRequest("I'm already busy with a dance.")
                )
                return
            self.is_learning = True

        with self._overlay_lock:
            poses_to_use = list(self._taught_poses)

        if not poses_to_use:
            self.nao.tts.request(
                NaoqiTextToSpeechRequest(
                    "You need to teach me a dance first, then we can practice it."
                )
            )
            with self._lock:
                self.is_learning = False
            return

        def _learner_thread():
            """Worker thread that runs the learner pipeline and updates overlays."""
            try:
                with self._overlay_lock:
                    self._active_pose_idx = None
                    self._learning_kp = None
                    self._learning_active = True

                def _learner_on_kp_frame(kp, dist, idx):
                    self._on_learning_frame_kp(kp)

                run_learner(
                    nao=self.nao,
                    nao_ip=self.nao_ip,
                    frame_provider=self.get_latest_frame,
                    logger=self.logger,
                    poses=poses_to_use,
                    on_pose_demo_start=self._on_pose_start,
                    on_pose_learned=self._on_pose_learned,
                    on_kp_frame=_learner_on_kp_frame,
                    on_pose_start=self._on_pose_start,
                )
            finally:
                with self._overlay_lock:
                    self._learning_active = False
                    self._learning_kp = None
                    self._active_pose_idx = None
                with self._lock:
                    self.is_learning = False

        threading.Thread(target=_learner_thread, daemon=True).start()

    def stop_teaching_mode(self):
        """Handle stop_teaching intent (placeholder for more advanced logic)."""
        self.nao.tts.request(
            NaoqiTextToSpeechRequest("Got it! Let's stop for now.")
        )

    # ----------------------------------------------------------
    # Clean shutdown
    # ----------------------------------------------------------
    def shutdown_nao(self):
        """Clean shutdown: stop teaching, rest NAO, stop all threads and exit."""
        self.logger.info("Shutdown intent received. Initiating clean shutdown...")

        try:
            self.nao.tts.request(
                NaoqiTextToSpeechRequest("Goodbye! Shutting down now.")
            )
            time.sleep(1.5)
        except Exception:
            pass

        with self._lock:
            self.is_teaching = False
            self.is_learning = False

        try:
            self.logger.info("Putting NAO to rest position...")
            self.nao.autonomous.request(NaoRestRequest())
            time.sleep(0.5)
        except Exception as e:
            self.logger.exception("Error while sending NaoRestRequest: %s", e)

        try:
            self.logger.info("Stopping camera...")
            self.nao.top_camera.unregister_callback(self._on_image)
        except Exception as e:
            self.logger.exception("Error stopping camera: %s", e)

        self._should_exit.set()
        self.shutdown_event.set()

    def exit_handler(self, *args, **kwargs):
        """Override to prevent sys.exit(0) during atexit."""
        try:
            super().exit_handler(*args, **kwargs)
        except SystemExit:
            pass

    # ----------------------------------------------------------
    # Main loop: show camera + run DF thread
    # ----------------------------------------------------------
    def run(self):
        """Run the main event loop: camera UI + Dialogflow background thread."""
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
            while not self.shutdown_event.is_set() and not self._should_exit.is_set():
                frame = self.get_latest_frame()
                if frame is not None:
                    frame_bgr = np.ascontiguousarray(frame)
                    frame_bgr = cv2.flip(frame_bgr, 1)
                    scale = 2.0
                    h, w = frame_bgr.shape[:2]
                    frame_bgr = cv2.resize(
                        frame_bgr, (int(w * scale), int(h * scale))
                    )

                    if self._learning_active:
                        # NAO is learning from human → white skeleton
                        if self.is_teaching:
                            self._draw_learning_skeleton_white(frame_bgr)
                        # NAO is teaching human → green/white feedback
                        elif self.is_learning:
                            self._draw_teaching_skeleton_feedback(frame_bgr)

                    self._draw_pose_thumbnails(frame_bgr)

                    cv2.imshow(camera_window, frame_bgr)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.logger.info("'q' pressed - initiating shutdown")
                        self.shutdown_nao()
                        break

                time.sleep(0.01)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        finally:
            self.logger.info("Cleaning up resources...")

            self._df_stop.set()
            if self._df_thread is not None:
                self._df_thread.join(timeout=2.0)

            try:
                self.nao.top_camera.unregister_callback(self._on_image)
            except Exception:
                pass

            try:
                self.nao.autonomous.request(NaoRestRequest())
            except Exception:
                pass

            cv2.destroyAllWindows()
            self.shutdown()

            self.logger.info("Shutdown complete.")


if __name__ == "__main__":
    app = NaoTeachMode()
    app.run()