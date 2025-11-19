from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Callable, List, Optional, Dict

import cv2
import mediapipe as mp
import numpy as np

from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest

from modules.replicate_json_pose import Pose, replicate_pose

mp_pose = mp.solutions.pose

BODY_IDXS = list(range(11, 33))
BODY_CONNS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]


def _log(logger, msg: str) -> None:
    """Small logger helper that falls back to print()."""
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _auto_pose_path(base_dir: str, idx: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"pose_{ts}_{idx:02d}.json")


def _landmarks_to_array(landmarks) -> np.ndarray:
    """Convert MediaPipe landmarks to (33, 3) normalized array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)


def _draw_pose_skeleton(frame: np.ndarray, kp_img_norm: np.ndarray) -> None:
    """Draw a simple white skeleton (utility, not used by main app)."""
    h, w = frame.shape[:2]
    for a, b in BODY_CONNS:
        xa, ya = int(kp_img_norm[a, 0] * w), int(kp_img_norm[a, 1] * h)
        xb, yb = int(kp_img_norm[b, 0] * w), int(kp_img_norm[b, 1] * h)
        cv2.line(frame, (xa, ya), (xb, yb), (255, 255, 255), 2)
    for i in BODY_IDXS:
        x, y = int(kp_img_norm[i, 0] * w), int(kp_img_norm[i, 1] * h)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)


def record_poses(
    logger,
    frame_provider: Callable[[], np.ndarray | None],
    out_dir: str,
    duration: float = 30.0,
    sample_interval: float = 5.0,
    countdown: int = 3,
    on_pose_saved: Callable[[Pose, int], None] | None = None,
    on_kp_frame: Callable[[np.ndarray | None], None] | None = None,
) -> List[Pose]:
    """
    Record poses from frames provided by `frame_provider`.

    - Waits `countdown` seconds (time-based).
    - For `duration` seconds, every `sample_interval` seconds:
        - grabs a frame (BGR)
        - runs MediaPipe Pose
        - if pose detected, saves kp_img_norm to JSON and returns Pose objects.
        - calls `on_pose_saved(pose, index)` each time, if provided.
    - On each processed frame, if `on_kp_frame` is provided, it is called with
      the current kp_img_norm (or None if no pose was detected).

    No windows, no OpenCV UI here. Completely headless.
    """
    _ensure_dir(out_dir)
    _log(logger, "Starting pose recording...")

    poses: List[Pose] = []
    pose_idx = 0

    # Countdown
    start_count = time.perf_counter()
    while time.perf_counter() - start_count < countdown:
        time.sleep(0.05)

    _log(logger, "Countdown finished, recording started.")

    with mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose_detector:

        record_start = time.perf_counter()
        next_sample = record_start

        while True:
            now = time.perf_counter()
            elapsed = now - record_start
            if elapsed > duration:
                break

            frame = frame_provider()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_bgr = np.ascontiguousarray(frame)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose_detector.process(rgb)
            rgb.flags.writeable = True

            kp = None
            if res.pose_landmarks:
                kp = _landmarks_to_array(res.pose_landmarks)

                if now >= next_sample:
                    pose_obj = Pose(kp_img_norm=kp)
                    poses.append(pose_obj)
                    out_path = _auto_pose_path(out_dir, pose_idx)
                    with open(out_path, "w") as f:
                        json.dump({"kp_img_norm": kp.tolist()}, f, indent=2)
                    _log(logger, f"Saved pose #{pose_idx} → {out_path}")

                    if on_pose_saved is not None:
                        on_pose_saved(pose_obj, pose_idx)

                    pose_idx += 1
                    next_sample += sample_interval

            if on_kp_frame is not None:
                on_kp_frame(kp)

    _log(logger, f"Recording finished. Total poses: {len(poses)}")
    return poses


def playback_poses(
    nao,
    nao_ip: str,
    poses: List[Pose],
    logger=None,
    sleep_between: float = 1.0,
    on_pose_start: Callable[[int], None] | None = None,
) -> None:
    """
    Playback a list of poses on NAO.
    No camera usage; just motions.

    If `on_pose_start` is provided, it is called with the pose index
    right before that pose is executed.
    """
    if not poses:
        _log(logger, "No poses to playback.")
        return

    for idx, pose in enumerate(poses):
        _log(logger, f"Playing back pose {idx+1}/{len(poses)}")

        if on_pose_start is not None:
            on_pose_start(idx)

        nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0, 0, 0))      # red
        replicate_pose(pose, nao_ip, mirror=True, duration=2.0)
        nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 1, 0, 0))      # green
        time.sleep(sleep_between)


def teach_sequence(
    nao,
    nao_ip: str,
    frame_provider: Callable[[], np.ndarray | None],
    logger=None,
    pose_dir: str = "poses",
    on_pose_saved: Callable[[Pose, int], None] | None = None,
    on_pose_start: Callable[[int], None] | None = None,
    on_kp_frame: Callable[[np.ndarray | None], None] | None = None,
) -> None:
    """
    Full teacher sequence (headless):

    1. NAO to Stand.
    2. Eyes blue: recording phase.
    3. Record poses from NAO camera (via frame_provider).
       - `on_pose_saved(pose, idx)` is called whenever a pose is captured.
       - `on_kp_frame(kp)` is called for each processed frame (or None).
    4. Eyes orange: playback phase.
    5. Playback poses on NAO.
       - `on_pose_start(idx)` is called when each pose starts.
    6. NAO to rest.
    """
    _ensure_dir(pose_dir)

    nao.motion.request(NaoPostureRequest("Stand", 0.5))
    time.sleep(1.0)

    # Eyes blue – ready to record
    nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 0, 1, 0))

    poses = record_poses(
        logger=logger,
        frame_provider=frame_provider,
        out_dir=pose_dir,
        duration=30.0,
        sample_interval=5.0,
        countdown=3,
        on_pose_saved=on_pose_saved,
        on_kp_frame=on_kp_frame,
    )

    if not poses:
        _log(logger, "No poses recorded; going to rest.")
        nao.autonomous.request(NaoRestRequest())
        return

    # Eyes orange – playback phase
    nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0.5, 0, 0.5))

    playback_poses(
        nao=nao,
        nao_ip=nao_ip,
        poses=poses,
        logger=logger,
        on_pose_start=on_pose_start,
    )
    nao.autonomous.request(NaoRestRequest())