"""Utilities for having NAO teach a pose sequence to a human.

- NAO goes to `Stand` posture.
- For each target pose in the sequence:
    1. NAO executes the pose (demonstration).
    2. NAO "looks" at the human and waits until the human imitates the pose.
       The current human pose from the camera is compared to the target pose.
    3. When the human pose matches the target, NAO
       confirms and moves on to the next pose.
- At the end, NAO goes to rest.

It is headless: no OpenCV windows or UI are opened.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, List

import cv2
import mediapipe as mp
import numpy as np
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest

from modules.replicate_json_pose import Pose, replicate_pose

mp_pose = mp.solutions.pose

BODY_IDXS = list(range(11, 33))


def _log(logger, msg: str) -> None:
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _landmarks_to_array(landmarks) -> np.ndarray:
    """Convert MediaPipe pose landmarks to a (33, 3) float32 array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)


def _normalize_kp_for_comparison(kp: np.ndarray) -> np.ndarray:
    """Normalize keypoints for pose comparison.

    Steps:
        - Take only BODY_IDXS joints.
        - Use x,y coordinates only.
        - Center them around the mean of body joints.
        - Scale them by the overall body size (bounding box diagonal).
    """
    body_xy = kp[BODY_IDXS, :2]

    # Center
    center = body_xy.mean(axis=0, keepdims=True)
    centered = body_xy - center

    # Scale
    min_xy = centered.min(axis=0)
    max_xy = centered.max(axis=0)
    diag = np.linalg.norm(max_xy - min_xy)
    scale = diag if diag > 1e-6 else 1e-6

    normalized = centered / scale
    return normalized  # shape


def _pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a scale/translation-invariant distance between two poses.

    Args:
        a: (33, 3) array of keypoints (target pose).
        b: (33, 3) array of keypoints (current human pose).

    Returns:
        Mean L2 distance between normalized body joints.
    """
    na = _normalize_kp_for_comparison(a)
    nb = _normalize_kp_for_comparison(b)
    return float(np.linalg.norm(na - nb, axis=1).mean())


def load_poses_from_dir(pose_dir: str) -> List[Pose]:
    """Load Pose objects from a directory of JSON files.

    Args:
        pose_dir: Directory containing JSON files with {"kp_img_norm": ...}.

    Returns:
        List of Pose objects sorted by filename.
    """
    if not os.path.isdir(pose_dir):
        return []

    files = sorted(
        f for f in os.listdir(pose_dir)
        if f.lower().endswith(".json")
    )

    poses: List[Pose] = []
    for fname in files:
        path = os.path.join(pose_dir, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            kp = np.array(data["kp_img_norm"], dtype=np.float32)
            poses.append(Pose(kp_img_norm=kp))
        except Exception as e:
            print(f"Failed to load pose from {path}: {e}")

    return poses


def wait_for_imitation(
    target_pose: Pose,
    frame_provider: Callable[[], np.ndarray | None],
    logger=None,
    max_wait: float = 10.0,
    match_threshold: float = 0.12,
    stable_frames: int = 5,
    on_kp_frame: Callable[[np.ndarray | None, float], None] | None = None,
) -> bool:
    """Wait until the human imitates `target_pose` or timeout occurs.

    The camera is read through `frame_provider`. For each frame:
        - MediaPipe Pose is run.
        - If a pose is detected, its distance to `target_pose` is computed.
        - If the distance is below `match_threshold` for `stable_frames`
          consecutive frames, we consider the pose correctly imitated.

    Args:
        target_pose: The Pose the human should imitate.
        frame_provider: Callable returning a BGR frame or None.
        logger: Logger-like object or None.
        max_wait: Maximum time to wait for a correct imitation.
        match_threshold: Distance threshold below which poses are considered
            matching.
        stable_frames: Number of consecutive frames below threshold required.
        on_kp_frame: Optional callback `on_kp_frame(kp, dist)` called on each
            processed frame with:
              - kp: (33, 3) keypoint array or None if no pose,
              - dist: float distance to target or float("inf") if no pose.

    Returns:
        True if imitation succeeded in time, False if timeout.
    """
    target_kp = target_pose.kp_img_norm

    _log(logger, "Waiting for human imitation...")

    with mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose_detector:

        start = time.perf_counter()
        consecutive = 0

        while True:
            now = time.perf_counter()
            if now - start > max_wait:
                _log(logger, "Imitation timed out.")
                return False

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
            dist = float("inf")

            if res.pose_landmarks:
                kp = _landmarks_to_array(res.pose_landmarks)
                dist = _pose_distance(target_kp, kp)

                if dist < match_threshold:
                    consecutive += 1
                    if consecutive >= stable_frames:
                        _log(logger, f"Imitation successful (dist={dist:.3f}).")
                        if on_kp_frame is not None:
                            on_kp_frame(kp, dist)
                        return True
                else:
                    consecutive = 0
            else:
                consecutive = 0

            if on_kp_frame is not None:
                on_kp_frame(kp, dist)

            time.sleep(0.01)


def learn_sequence(
    nao,
    nao_ip: str,
    poses: List[Pose],
    frame_provider: Callable[[], np.ndarray | None],
    logger=None,
    demo_duration: float = 2.5,
    wait_max: float = 15.0,          # <-- now default 15 seconds
    match_threshold: float = 0.02,
    stable_frames: int = 5,
    on_pose_demo_start: Callable[[int], None] | None = None,
    on_pose_learned: Callable[[int], None] | None = None,
    on_kp_frame: Callable[[np.ndarray | None, float, int], None] | None = None,
) -> None:
    """Run the 'learning' routine: NAO demonstrates, human imitates.

    For each pose in `poses`:
        1. NAO executes the pose (demonstration) and holds it.
        2. NAO waits (up to `wait_max` seconds) for the human imitation,
           using the camera to compare the human pose with the target pose.
        3. If imitation is successful, NAO confirms and moves on.
           If timeout occurs, NAO gives up on that pose and still moves on.

    LED convention (you can tweak these if you want):
        - Blue: NAO is demonstrating a pose.
        - Yellow: Waiting for imitation.
        - Green: Imitation successful.
        - Red: Imitation timed out / failed.
    """
    if not poses:
        _log(logger, "No poses provided to learn_sequence.")
        return

    # Put NAO in a standing posture
    _log(logger, "Starting learning sequence. Going to Stand posture.")
    nao.motion.request(NaoPostureRequest("Stand", 0.5))
    time.sleep(1.0)

    try:
        for idx, pose in enumerate(poses):
            _log(logger, f"Pose {idx+1}/{len(poses)} – demonstration phase.")

            if on_pose_demo_start is not None:
                on_pose_demo_start(idx)

            # Eyes blue – demonstrating
            nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 0, 1, 0))

            # NAO demonstrates the pose (this moves into pose A)
            replicate_pose(pose, nao_ip, mirror=True, duration=demo_duration)
            time.sleep(0.5)

            # Eyes yellow – waiting for imitation (stay in pose A)
            nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 1, 0, 0))

            def _wrapped_on_kp_frame(kp, dist):
                if on_kp_frame is not None:
                    on_kp_frame(kp, dist, idx)

            success = wait_for_imitation(
                target_pose=pose,
                frame_provider=frame_provider,
                logger=logger,
                max_wait=wait_max,            # <-- wait up to 15s
                match_threshold=match_threshold,
                stable_frames=stable_frames,
                on_kp_frame=_wrapped_on_kp_frame,
            )

            if success:
                # Eyes green – success
                nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 1, 0, 0))
                _log(logger, f"Pose {idx+1} learned successfully.")
                if on_pose_learned is not None:
                    on_pose_learned(idx)
            else:
                # Eyes red – failed / timeout
                nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0, 0, 0))
                _log(logger, f"Pose {idx+1} imitation failed or timed out.")

            # Small pause between poses
            time.sleep(1.0)

        nao.tts.request(
            NaoqiTextToSpeechRequest("Okay that was the lesson, can you show me now if you remember all the steps")
        )

    finally:
        _log(logger, "Learning sequence finished. Going to rest.")
        nao.autonomous.request(NaoRestRequest())