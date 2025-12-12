"""
High-level entry point for the NAO pose learner pipeline.

This module wires NAO, a frame provider (camera), and optional callbacks
into the `learn_sequence` routine from `modules.pose_learner`.

If `poses` is provided, only those poses are used. Otherwise it will load
poses from the default `poses` directory.
"""

from __future__ import annotations

import os
from typing import Callable, Sequence

import numpy as np

from modules.pose_learner import (
    learn_sequence,
    load_poses_from_dir,
)
from modules.replicate_json_pose import Pose


def run_learner(
    nao,
    nao_ip: str,
    frame_provider: Callable[[], np.ndarray | None],
    logger=None,
    poses: Sequence[Pose] | None = None,
    on_pose_demo_start: Callable[[int], None] | None = None,
    on_pose_learned: Callable[[int], None] | None = None,
    on_kp_frame: Callable[[np.ndarray | None, float, int], None] | None = None,
    on_pose_start: Callable[[np.ndarray | None, float, int], None] | None = None,
) -> None:
    """Run the synchronous learner pipeline on a NAO robot.

    This is a convenience wrapper around `learn_sequence` that:
      - Uses the given `poses` if provided and non-empty.
      - Otherwise resolves the default pose directory and loads all poses.
      - Forwards NAO handle, IP, frame provider and optional callbacks.

    Args:
        nao: NAO device handle with `.motion`, `.leds`, `.autonomous`.
        nao_ip: IP address of the NAO robot.
        frame_provider: Callable returning a BGR frame (np.ndarray) or None.
        logger: Optional logger-like object.
        poses: Optional sequence of Pose objects to use. If None or empty,
            poses are loaded from disk.
        on_pose_demo_start: Optional callback `on_pose_demo_start(idx)`.
        on_pose_learned: Optional callback `on_pose_learned(idx)`.
        on_kp_frame: Optional callback `on_kp_frame(kp, dist, idx)`.
        on_pose_start: Optional callback `on_pose_start(idx)`.
    """
    if poses is None or len(poses) == 0:
        pose_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "poses")
        )
        poses_list: list[Pose] = load_poses_from_dir(pose_dir)

        if not poses_list:
            msg = (
                f"No poses provided and no poses found in directory '{pose_dir}'. "
                "Aborting learner pipeline."
            )
            if logger is not None:
                logger.warning(msg)
            else:
                print(f"[run_learner] {msg}")
            return
    else:
        poses_list = list(poses)

    learn_sequence(
        nao=nao,
        nao_ip=nao_ip,
        poses=poses_list,
        frame_provider=frame_provider,
        logger=logger,
        on_pose_demo_start=on_pose_demo_start,
        on_pose_learned=on_pose_learned,
        on_kp_frame=on_kp_frame,
        on_pose_start=on_pose_start
    )