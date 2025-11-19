"""
Load a pose template and coach live.

Args:
    --camera: Camera index (default: 0)
    --alpha: EMA smoothing factor (0,1] (default: 0.4)
    --mc: Model complexity (default: 1)
    --min-det: Min detection confidence (default: 0.6)
    --min-track: Min tracking confidence (default: 0.6)
    --template: Path to saved pose JSON (required)

Usage:
  python PoseEvaluator.py --template Poses/MyPose.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Tuple

import cv2
import numpy as np

from PoseCore import (
    PoseTracker, PoseTrackerConfig,
    render_target_thumbnail,
    compare_angles, good_joints_from_flags,
    draw_body, DEFAULT_TOLERANCES
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coach live pose against a saved template.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--mc", type=int, default=1)
    p.add_argument("--min-det", type=float, default=0.6)
    p.add_argument("--min-track", type=float, default=0.6)
    p.add_argument("--template", type=str, required=True, help="Path to saved pose JSON")
    return p.parse_args()

def load_pose(path: str) -> tuple[Dict[str, float], np.ndarray, Dict[str, float]]:
    with open(path, "r") as f:
        data = json.load(f)
    angles = data["angles"]
    kp_img_norm = np.array(data["kp_img_norm"], dtype=np.float32)
    tolerances = data.get("tolerances", DEFAULT_TOLERANCES)
    return angles, kp_img_norm, tolerances

def main() -> None:
    args = parse_args()
    target_angles, target_kp, tolerances = load_pose(args.template)

    cfg = PoseTrackerConfig(
        model_complexity=args.mc,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_track,
        smoother_alpha=args.alpha,
    )
    tracker = PoseTracker(cfg)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            annotated, kp_img_norm, live_angles = tracker.process(frame)

            flags = compare_angles(live_angles, target_angles, tolerances) if live_angles else {}
            good_joints = good_joints_from_flags(flags)

            if kp_img_norm is not None:
                joint_colors: Dict[int, Tuple[int,int,int]] = {j: (0,255,0) for j in good_joints}
                annotated[:] = frame
                draw_body(
                    annotated, kp_img_norm,
                    joint_colors=joint_colors,
                    bone_color=(180,180,180),
                    bone_color_if=(0,255,0),
                    good_joints=good_joints
                )

            total_tracked = len(tolerances)
            ok_count = sum(flags.values()) if flags else 0
            cv2.putText(annotated, f"Within tol: {ok_count}/{total_tracked}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            render_target_thumbnail(annotated, target_kp, label="TARGET")

            cv2.imshow("Pose Coach (ESC to quit)", annotated)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()