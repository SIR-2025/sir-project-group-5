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
    """
    Synchronous teacher pipeline.

    - Reuses NAO's camera stream via `frame_provider`.
    - Calls GUI callbacks `on_pose_saved`, `on_pose_start`, `on_kp_frame` if provided.
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