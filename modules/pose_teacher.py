"""Utilities to record human poses with MediaPipe and replay them on a NAO robot.

This module provides three main operations:

- `record_poses`: capture poses from a video stream and save them as JSON files.
- `playback_poses`: execute a sequence of poses on a NAO robot.
- `teach_sequence`: full teaching routine that records poses and replays them.

It is headless: no OpenCV windows or UI are opened.
"""

from __future__ import annotations

import json
import os
import threading
import time
import wave
from datetime import datetime
from typing import Callable, List, Optional, Dict

import cv2
import mediapipe as mp
import numpy as np
from sic_framework import AudioRequest

from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import (
    NaoRestRequest,
    NaoSetAutonomousLifeRequest,  # Changed to enable/disable autonomous life
)
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest

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
    """Log a message using the provided logger, or print it if logger is None.

    Args:
        logger: Logger-like object with an `.info()` method, or None.
        msg: Message string to log.
    """
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _ensure_dir(path: str) -> None:
    """Create the given directory path if it does not already exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def _auto_pose_path(base_dir: str, idx: int) -> str:
    """Generate a timestamped JSON file path for a stored pose.

    Args:
        base_dir: Base directory where pose files are stored.
        idx: Pose index used in the filename.

    Returns:
        Full path to the JSON file for this pose.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"pose_{ts}_{idx:02d}.json")


def _landmarks_to_array(landmarks) -> np.ndarray:
    """Convert MediaPipe pose landmarks to a (33, 3) float32 array.

    Args:
        landmarks: A MediaPipe `NormalizedLandmarkList` with 33 landmarks.

    Returns:
        Numpy array of shape (33, 3) with (x, y, z) coordinates in normalized
        image coordinates.
    """
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)


def _draw_pose_skeleton(frame: np.ndarray, kp_img_norm: np.ndarray) -> None:
    """Draw a simple white pose skeleton on an image.

    Args:
        frame: BGR image on which to draw (modified in place).
        kp_img_norm: Array of shape (33, 3) with normalized (x, y, z) keypoints.
    """
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
    """Record a sequence of poses from frames provided by `frame_provider`.

    The function runs headless.

    Workflow:
        1. Wait `countdown` seconds.
        2. For up to `duration` seconds, repeatedly:
           - grab a frame from `frame_provider` (BGR, or None),
           - run MediaPipe Pose detection,
           - every `sample_interval` seconds, if a pose is detected:
               * store the keypoints to a JSON file under `out_dir`,
               * append a `Pose` object to the result list,
               * call `on_pose_saved(pose, index)` if provided.
        3. For every processed frame, call `on_kp_frame(kp)` if provided, where
           `kp` is the current keypoint array or None if no pose was detected.

    Args:
        logger: Logger-like object or None.
        frame_provider: Callable that returns a BGR frame (np.ndarray) or None.
        out_dir: Directory where JSON pose files will be stored.
        duration: Total recording duration in seconds.
        sample_interval: Time in seconds between saved poses.
        countdown: Time in seconds to wait before starting recording.
        on_pose_saved: Optional callback called as `on_pose_saved(pose, idx)`
            whenever a new pose is saved.
        on_kp_frame: Optional callback called as `on_kp_frame(kp)` on each
            processed frame; `kp` is a (33, 3) array or None.

    Returns:
        List of `Pose` objects in the order they were captured.
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
                # No frame available; wait a bit and retry.
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
    sleep_between: float = 0.5,
    on_pose_start: Callable[[int], None] | None = None,
) -> None:
    """Replay a sequence of poses on a NAO robot (assumes already in StandInit)."""
    if not poses:
        _log(logger, "No poses to playback.")
        return

    for idx, pose in enumerate(poses):
        _log(logger, f"Playing back pose {idx+1}/{len(poses)}")

        if on_pose_start is not None:
            on_pose_start(idx)

        # Never reset - flow smoothly from pose to pose
        replicate_pose(pose, nao_ip, mirror=True, duration=2.5, reset_to_standinit=False)
        
        time.sleep(sleep_between)

def play_audio(nao, wav_path: str, logger=None):
    """Play a WAV file through NAO's speaker."""
    log = logger.info if logger else print
    log(f"Playing WAV on NAO: {wav_path}")

    wf = wave.open(wav_path, "rb")
    samplerate = wf.getframerate()
    wf.rewind()
    data = wf.readframes(wf.getnframes())
    msg = AudioRequest(sample_rate=samplerate, waveform=bytes(data))
    nao.speaker.request(msg)

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
    """Run the full pose teaching sequence on NAO (record + playback).

    High-level flow:
        1. Put NAO in `Stand` posture.
        2. Set face LEDs to blue (recording phase).
        3. Call `record_poses` to capture poses from `frame_provider` and
           store them under `pose_dir`.
           - `on_pose_saved(pose, idx)` is forwarded to `record_poses`.
           - `on_kp_frame(kp)` is forwarded to `record_poses`.
        4. If no poses are recorded, send NAO to rest and return.
        5. Set face LEDs to orange (playback phase).
        6. Call `playback_poses` to execute the recorded poses on NAO.
           - `on_pose_start(idx)` is forwarded to `playback_poses`.
        7. Send NAO to rest.

    Args:
        nao: NAO device handle with `.motion`, `.leds`, and `.autonomous`.
        nao_ip: IP address of the NAO robot.
        frame_provider: Callable that returns a BGR frame (np.ndarray) or None.
        logger: Logger-like object or None.
        pose_dir: Directory where pose JSON files will be saved.
        on_pose_saved: Optional callback forwarded to `record_poses`.
        on_pose_start: Optional callback forwarded to `playback_poses`.
        on_kp_frame: Optional callback forwarded to `record_poses`.
    """
    _ensure_dir(pose_dir)

    nao.motion.request(NaoPostureRequest("Stand", 0.5))
    time.sleep(1.0)

    # Eyes blue – ready to record
    nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 0, 1, 0))

    song_path = "music/song.wav"
    song_thread = threading.Thread(
        target=play_audio,
        args=(nao, song_path, logger),
        daemon=True,
    )
    song_thread.start()
    time.sleep(0.5)

    poses = record_poses(
        frame_provider=frame_provider,
        logger=logger,
        out_dir=pose_dir,
        duration=18.0,
        sample_interval=3.0,
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

    nao.tts.request(
        NaoqiTextToSpeechRequest("Hmm, fancy moves detected. let me try that...")
    )

    playback_poses(
        nao=nao,
        nao_ip=nao_ip,
        poses=poses,
        logger=logger,
        on_pose_start=on_pose_start,
    )
