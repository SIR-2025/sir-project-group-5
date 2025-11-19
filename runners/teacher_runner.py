"""
High-level entry point for the NAO pose teacher pipeline.

This module wires NAO, a frame provider (camera), and optional GUI callbacks
into the `teach_sequence` routine from `modules.pose_teacher`.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from modules.pose_teacher import teach_sequence
from modules.replicate_json_pose import Pose


def run_teacher(
    nao,
    nao_ip: str,
    frame_provider: Callable[[], np.ndarray | None],
    logger=None,
    on_pose_saved: Callable[[Pose, int], None] | None = None,
    on_pose_start: Callable[[int], None] | None = None,
    on_kp_frame: Callable[[np.ndarray | None], None] | None = None,
) -> None:
    """Run the synchronous teacher pipeline on a NAO robot.

    This is a convenience wrapper around `teach_sequence` that:
      - Resolves a default pose directory.
      - Forwards NAO handle, IP, frame provider and optional callbacks.

    Args:
        nao: NAO device handle with `.motion`, `.leds`, `.autonomous`.
        nao_ip: IP address of the NAO robot.
        frame_provider: Callable returning a BGR frame (np.ndarray) or None.
        logger: Optional logger-like object.
        on_pose_saved: Optional callback called as `on_pose_saved(pose, idx)`
            whenever a pose is captured and saved.
        on_pose_start: Optional callback called as `on_pose_start(idx)` right
            before a pose is replayed on the robot.
        on_kp_frame: Optional callback called as `on_kp_frame(kp)` for every
            processed frame during recording, where `kp` is a (33, 3) array or
            None if no pose was detected.
    """
    pose_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "poses")
    )

    teach_sequence(
        nao=nao,
        nao_ip=nao_ip,
        frame_provider=frame_provider,
        logger=logger,
        pose_dir=pose_dir,
        on_pose_saved=on_pose_saved,
        on_pose_start=on_pose_start,
        on_kp_frame=on_kp_frame,
    )