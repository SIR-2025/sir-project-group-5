"""
Utilities for having NAO teach a pose sequence to a human.

This module is HEADLESS:
- No OpenCV windows
- No drawing
- No UI logic

It only:
- runs MediaPipe Pose
- computes pose distance
- reports pose state to main.py via callbacks

Main.py is the ONLY place that draws anything.
"""

from __future__ import annotations

import json
import os
import threading
import time
import wave
from typing import Callable, List

import cv2
import mediapipe as mp
import numpy as np
from sic_framework import AudioRequest
from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import (
    NaoqiTextToSpeechRequest,
)

from modules.pose_teacher import playback_poses
from modules.replicate_json_pose import Pose, replicate_pose

mp_pose = mp.solutions.pose

# Body joints used for comparison (MediaPipe indices)
BODY_IDXS = list(range(11, 33))

def load_poses_from_dir(pose_dir: str) -> List[Pose]:
    """Load Pose objects from a directory of JSON files.

    Each file must contain:
        {
            "kp_img_norm": [[x, y, z], ...]  # length 33
        }
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

def _log(logger, msg: str) -> None:
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _landmarks_to_array(landmarks) -> np.ndarray:
    """Convert MediaPipe pose landmarks to (33, 3) float32 array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32
    )


def _normalize_kp_for_comparison(kp: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints for pose comparison.

    - Uses BODY_IDXS only
    - Uses x,y only
    - Translation invariant (centered)
    - Scale invariant (bounding box diagonal)
    """
    body_xy = kp[BODY_IDXS, :2]

    center = body_xy.mean(axis=0, keepdims=True)
    centered = body_xy - center

    min_xy = centered.min(axis=0)
    max_xy = centered.max(axis=0)
    diag = np.linalg.norm(max_xy - min_xy)

    scale = diag if diag > 1e-6 else 1e-6
    return centered / scale


def _pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute scale/translation-invariant distance between two poses.

    Returns:
        Mean L2 distance across BODY_IDXS joints.
    """
    na = _normalize_kp_for_comparison(a)
    nb = _normalize_kp_for_comparison(b)
    return float(np.linalg.norm(na - nb, axis=1).mean())

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

def wait_for_imitation(
    target_pose: Pose,
    frame_provider: Callable[[], np.ndarray | None],
    logger=None,
    max_wait: float = 15.0,
    match_threshold: float = 0.08,
    stable_frames: int = 3,
    pose_idx: int | None = None,
    on_kp_frame: Callable[[np.ndarray | None, float, int | None], None] | None = None,
) -> bool:
    """
    Wait until the human imitates `target_pose` or timeout occurs.

    Callback contract:
        on_kp_frame(kp, dist, pose_idx)

    No drawing is done here.
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
            if time.perf_counter() - start > max_wait:
                _log(logger, "Imitation timed out.")
                return False

            frame = frame_provider()
            if frame is None:
                time.sleep(0.01)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                        if on_kp_frame:
                            on_kp_frame(kp, dist, pose_idx)
                        return True
                else:
                    consecutive = 0
            else:
                consecutive = 0

            if on_kp_frame:
                on_kp_frame(kp, dist, pose_idx)

            time.sleep(0.01)

def learn_sequence(
    nao,
    nao_ip: str,
    poses: List[Pose],
    frame_provider: Callable[[], np.ndarray | None],
    logger=None,
    demo_duration: float = 2.5,
    wait_max: float = 15.0,
    match_threshold: float = 0.08,
    stable_frames: int = 3,
    on_pose_demo_start: Callable[[int], None] | None = None,
    on_pose_learned: Callable[[int], None] | None = None,
    on_kp_frame: Callable[[np.ndarray | None, float, int | None], None] | None = None,
    on_pose_start: Callable[[int], None] | None = None,
) -> None:
    """
    NAO demonstrates each pose, human imitates.

    LEDs:
        Blue   – demonstrating
        Yellow – waiting for imitation
        Green  – success
        Red    – timeout/failure
    """
    if not poses:
        _log(logger, "No poses provided to learn_sequence.")
        return

    _log(logger, "Starting learning sequence. Going to Stand posture.")
    nao.motion.request(NaoPostureRequest("Stand", 0.5))
    time.sleep(1.0)

    try:
        for idx, pose in enumerate(poses):
            _log(logger, f"Pose {idx+1}/{len(poses)} – demonstration phase.")

            if on_pose_demo_start:
                on_pose_demo_start(idx)

            # Blue – demonstrating
            nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 0, 1, 0))

            replicate_pose(
                pose,
                nao_ip,
                mirror=True,
                duration=demo_duration,
            )
            time.sleep(0.5)

            # Yellow – waiting
            nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 1, 0, 0))

            success = wait_for_imitation(
                target_pose=pose,
                frame_provider=frame_provider,
                logger=logger,
                max_wait=wait_max,
                match_threshold=match_threshold,
                stable_frames=stable_frames,
                pose_idx=idx,
                on_kp_frame=on_kp_frame,
            )

            if success:
                nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 1, 0, 0))
                _log(logger, f"Pose {idx+1} learned successfully.")
                if on_pose_learned:
                    on_pose_learned(idx)
            else:
                nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0, 0, 0))
                _log(logger, f"Pose {idx+1} imitation failed or timed out.")

            time.sleep(1.0)

        nao.tts.request(
            NaoqiTextToSpeechRequest(
                "Okay, that was the lesson. Can you show me if you remember all the steps?"
            )
        )

        song_thread = threading.Thread(
            target=play_audio,
            args=(nao, "music/song.wav", logger),
            daemon=True,
        )
        song_thread.start()

        playback_poses(
            nao=nao,
            nao_ip=nao_ip,
            poses=poses,
            logger=logger,
            on_pose_start=on_pose_start,
        )

    finally:
        pass