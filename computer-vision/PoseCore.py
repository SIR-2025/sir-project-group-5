from __future__ import annotations

import contextlib
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable

import cv2
import mediapipe as mp
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

mp_pose = mp.solutions.pose

BODY_IDXS = list(range(11, 33))
BODY_CONNS: list[Tuple[int, int]] = [
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32),
]

ANGLE_VERTEX_MAP: Dict[str, int] = {
    "shoulder_L": 11, "shoulder_R": 12,
    "elbow_L": 13,    "elbow_R": 14,
    "wrist_L": 15,    "wrist_R": 16,
    "hip_L": 23,      "hip_R": 24,
}

DEFAULT_TOLERANCES: Dict[str, float] = {
    "elbow_L": 10.0, "elbow_R": 10.0,
    "shoulder_L": 15.0, "shoulder_R": 15.0,
    "wrist_L": 15.0, "wrist_R": 15.0,
    "hip_L": 20.0, "hip_R": 20.0,
}


class EMASmoother:
    """Exponential moving average smoother for keypoints."""
    def __init__(self, alpha: float = 0.4):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0,1].")
        self.alpha = alpha
        self.prev: Optional[np.ndarray] = None

    def __call__(self, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            self.prev = None
            return None
        if self.prev is None:
            self.prev = arr
            return arr
        self.prev = self.alpha * arr + (1.0 - self.alpha) * self.prev
        return self.prev


def landmarks_to_array(landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> np.ndarray:
    """Return (33,3) array of normalized image coordinates."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)


def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    """Center on pelvis and scale by shoulder width in xy."""
    L_HIP, R_HIP, L_SHOULDER, R_SHOULDER = 23, 24, 11, 12
    pelvis = (kp[L_HIP] + kp[R_HIP]) / 2.0
    kp_centered = kp - pelvis
    shoulder_width = np.linalg.norm(kp[L_SHOULDER][:2] - kp[R_SHOULDER][:2]) + 1e-6
    return kp_centered / shoulder_width


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the planar angle at vertex b (degrees)."""
    v1, v2 = (a - b)[:2], (c - b)[:2]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def compute_joint_angles(kp_norm: np.ndarray) -> Dict[str, float]:
    """Return elbow, shoulder, wrist, and hip angles in degrees."""
    idx = {
        "L_SHOULDER": 11, "R_SHOULDER": 12,
        "L_ELBOW": 13,    "R_ELBOW": 14,
        "L_WRIST": 15,    "R_WRIST": 16,
        "L_INDEX": 19,    "R_INDEX": 20,
        "L_HIP": 23,      "R_HIP": 24,
        "L_KNEE": 25,     "R_KNEE": 26,
    }
    A = angle_deg(kp_norm[idx["L_SHOULDER"]], kp_norm[idx["L_ELBOW"]], kp_norm[idx["L_WRIST"]])
    B = angle_deg(kp_norm[idx["R_SHOULDER"]], kp_norm[idx["R_ELBOW"]], kp_norm[idx["R_WRIST"]])
    C = angle_deg(kp_norm[idx["L_ELBOW"]],    kp_norm[idx["L_SHOULDER"]], kp_norm[idx["L_HIP"]])
    D = angle_deg(kp_norm[idx["R_ELBOW"]],    kp_norm[idx["R_SHOULDER"]], kp_norm[idx["R_HIP"]])
    E = angle_deg(kp_norm[idx["L_SHOULDER"]], kp_norm[idx["L_HIP"]],     kp_norm[idx["L_KNEE"]])
    F = angle_deg(kp_norm[idx["R_SHOULDER"]], kp_norm[idx["R_HIP"]],     kp_norm[idx["R_KNEE"]])
    G = angle_deg(kp_norm[idx["L_ELBOW"]],    kp_norm[idx["L_WRIST"]],   kp_norm[idx["L_INDEX"]])
    H = angle_deg(kp_norm[idx["R_ELBOW"]],    kp_norm[idx["R_WRIST"]],   kp_norm[idx["R_INDEX"]])
    return {
        "elbow_L": A, "elbow_R": B,
        "shoulder_L": C, "shoulder_R": D,
        "hip_L": E, "hip_R": F,
        "wrist_L": G, "wrist_R": H,
    }


def draw_body(
    frame_bgr: np.ndarray,
    kp_img_norm: np.ndarray,
    joint_colors: Optional[Dict[int, Tuple[int,int,int]]] = None,
    bone_color: Tuple[int,int,int] = (255,255,255),
    bone_color_if: Optional[Tuple[int,int,int]] = None,
    good_joints: Optional[Iterable[int]] = None,
) -> None:
    """Draw body skeleton; color bones if both endpoints are good and joints via joint_colors."""
    h, w = frame_bgr.shape[:2]
    gj = set(good_joints) if good_joints else set()
    for a, b in BODY_CONNS:
        xa, ya = int(kp_img_norm[a,0]*w), int(kp_img_norm[a,1]*h)
        xb, yb = int(kp_img_norm[b,0]*w), int(kp_img_norm[b,1]*h)
        col = bone_color
        if bone_color_if is not None and a in gj and b in gj:
            col = bone_color_if
        cv2.line(frame_bgr, (xa, ya), (xb, yb), col, 2)
    for i in BODY_IDXS:
        x, y = int(kp_img_norm[i,0]*w), int(kp_img_norm[i,1]*h)
        col = (255,255,255)
        if joint_colors and i in joint_colors:
            col = joint_colors[i]
        cv2.circle(frame_bgr, (x,y), 5, col, -1)


def render_target_thumbnail(
    frame_bgr: np.ndarray,
    target_kp_img_norm: np.ndarray,
    scale: float = 0.28,
    margin: int = 12,
    label: str = "TARGET",
    alpha: float = 0.4,
) -> None:
    """Render a small transparent target thumbnail at bottom-right using normalized coordinates."""
    h, w = frame_bgr.shape[:2]
    thumb_h = int(h * scale)
    thumb_w = int(w * scale)
    x0 = w - thumb_w - margin
    y0 = h - thumb_h - margin

    overlay = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    draw_body(overlay, target_kp_img_norm)

    roi = frame_bgr[y0:y0 + thumb_h, x0:x0 + thumb_w]
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

    cv2.rectangle(frame_bgr, (x0, y0), (x0 + thumb_w, y0 + thumb_h), (80, 80, 80), 1)
    cv2.putText(
        frame_bgr,
        label,
        (x0 + 8, y0 + thumb_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

@dataclass
class PoseTrackerConfig:
    """Configuration for the pose tracker."""
    model_complexity: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    smoother_alpha: float = 0.4


class PoseTracker:
    """MediaPipe Pose wrapper with smoothing and drawing helpers."""
    def __init__(self, cfg: PoseTrackerConfig):
        self.smoother = EMASmoother(alpha=cfg.smoother_alpha)
        devnull = open(os.devnull, "w")
        with contextlib.redirect_stderr(devnull):
            self.pose = mp_pose.Pose(
                model_complexity=cfg.model_complexity,
                enable_segmentation=False,
                min_detection_confidence=cfg.min_detection_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence,
            )
            _ = self.pose.process(np.zeros((480,640,3), dtype=np.uint8))
        devnull.close()

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
        """Return annotated frame, smoothed normalized keypoints, and computed angles."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.pose.process(rgb)
        rgb.flags.writeable = True
        angles: Dict[str, float] = {}
        kp_img_norm_smooth: Optional[np.ndarray] = None
        if res.pose_landmarks:
            kp_img_norm = landmarks_to_array(res.pose_landmarks)
            kp_img_norm_smooth = self.smoother(kp_img_norm)
            if kp_img_norm_smooth is not None:
                draw_body(frame_bgr, kp_img_norm_smooth)
                kp_norm = normalize_keypoints(kp_img_norm_smooth)
                angles = compute_joint_angles(kp_norm)
        return frame_bgr, kp_img_norm_smooth, angles

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.pose.close()


def compare_angles(live: Dict[str, float], target: Dict[str, float], tol: Dict[str, float]) -> Dict[str, bool]:
    """Return per-angle within-tolerance flags."""
    flags: Dict[str, bool] = {}
    for k, tval in target.items():
        v = live.get(k, float("nan"))
        if np.isnan(v):
            flags[k] = False
        else:
            flags[k] = abs(v - tval) <= tol.get(k, 9999.0)
    return flags


def good_joints_from_flags(flags: Dict[str, bool]) -> set[int]:
    """Translate angle flags to the set of vertex joint indices that are within tolerance."""
    good: set[int] = set()
    for name, ok in flags.items():
        if ok and name in ANGLE_VERTEX_MAP:
            good.add(ANGLE_VERTEX_MAP[name])
    return good