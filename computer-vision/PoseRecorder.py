"""
Record a static pose with a 3-2-1-0 countdown.
Press SPACE to start the countdown; at 0 the pose is saved as JSON.
The last saved pose is displayed as a transparent overlay in the bottom-right corner.
Press SPACE again to re-record or ESC to exit.

Args:
    --camera: Camera index (default: 0)
    --alpha: EMA smoothing factor (0,1] (default: 0.4
    --mc: Model complexity (default: 1)
    --min-det: Min detection confidence (default: 0.6)
    --min-track: Min tracking confidence (default: 0.6)
    --out: Output JSON path (default: ./Poses/Pose_TIMESTAMP.json)

Usage:
    python PoseRecorder.py --out Poses/MyPose.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from PoseCore import PoseTracker, PoseTrackerConfig, DEFAULT_TOLERANCES, draw_body, render_target_thumbnail


def save_pose(path: str, angles: Dict[str, float], kp_img_norm: np.ndarray, tolerances: Dict[str, float]) -> None:
    """Save a recorded pose to JSON."""
    data = {"angles": angles, "kp_img_norm": kp_img_norm.tolist(), "tolerances": tolerances}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved pose → {path}")


def auto_path(base_dir: str = "Poses") -> str:
    """Return an automatic filename with timestamp."""
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(base_dir, exist_ok=True)
    return str(Path(base_dir) / f"Pose_{ts}.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Record a static pose with a 3-2-1-0 countdown.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--mc", type=int, default=1)
    p.add_argument("--min-det", type=float, default=0.6)
    p.add_argument("--min-track", type=float, default=0.6)
    p.add_argument("--out", type=str, default="", help="Output JSON path (default: ./poses/pose_TIMESTAMP.json)")
    return p.parse_args()


def overlay_transparent(frame: np.ndarray, overlay: np.ndarray, x: int, y: int, alpha: float = 0.4) -> None:
    """Blend an overlay image onto the frame at (x, y) with transparency alpha."""
    h, w = overlay.shape[:2]
    if x + w > frame.shape[1] or y + h > frame.shape[0]:
        return
    roi = frame[y:y + h, x:x + w]
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

def main() -> None:
    """Main entry point."""
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    cfg = PoseTrackerConfig(
        model_complexity=args.mc,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_track,
        smoother_alpha=args.alpha,
    )
    tracker = PoseTracker(cfg)

    last_saved_kp: Optional[np.ndarray] = None
    countdown_active = False
    countdown_start_t = 0.0
    countdown_secs = 3

    print("RECORD: Press SPACE to start 3-2-1-0 capture. ESC to exit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            annotated, kp_img_norm, angles = tracker.process(frame)

            if countdown_active:
                elapsed = time.perf_counter() - countdown_start_t
                remaining = max(0, countdown_secs - int(elapsed))
                cv2.putText(
                    annotated,
                    str(remaining),
                    (annotated.shape[1] // 2 - 20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.5,
                    (0, 255, 255),
                    6,
                    cv2.LINE_AA,
                )
                if elapsed >= countdown_secs + 0.2:
                    countdown_active = False
                    if kp_img_norm is not None and angles:
                        path = args.out if args.out else auto_path()
                        save_pose(path, angles, kp_img_norm, DEFAULT_TOLERANCES)
                        last_saved_kp = kp_img_norm.copy()
                    else:
                        cv2.putText(
                            annotated,
                            "No pose detected—try again.",
                            (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

            if last_saved_kp is not None:
                render_target_thumbnail(annotated, last_saved_kp)

            cv2.putText(
                annotated,
                "SPACE: record | ESC: quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Pose Recorder", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32 and not countdown_active:
                countdown_active = True
                countdown_start_t = time.perf_counter()
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()