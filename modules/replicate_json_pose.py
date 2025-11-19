"""
Control a NAO robot to reproduce a static pose recorded with MediaPipe.

Main entry points:

- replicate_pose_from_kp(kp_img_norm, nao_ip, ...)
    Directly use the (33×2/3) list or NumPy array of normalized keypoints.
- replicate_pose(pose, nao_ip, ...)
    Use the minimal Pose dataclass wrapper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List

import numpy as np
import qi


@dataclass(frozen=True)
class Pose:
    """In-memory pose with 33 normalized keypoints (x, y, [z])."""
    kp_img_norm: np.ndarray

    def __post_init__(self):
        """Validate and normalize the stored keypoints array."""
        kp = np.asarray(self.kp_img_norm, dtype=np.float32)
        if kp.ndim != 2 or kp.shape[0] != 33 or kp.shape[1] < 2:
            raise ValueError(f"kp_img_norm must be (33, 2/3), got {kp.shape}")
        object.__setattr__(self, "kp_img_norm", kp)


# NAO arm joint names (left and right).
ARM_JOINTS = [
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll",
]

# Joint limits for safety (in radians).
LIMITS: Dict[str, Tuple[float, float]] = {
    "LShoulderPitch": (-2.08,  2.08),
    "LShoulderRoll":  (-0.31,  1.33),
    "LElbowYaw":      (-2.08,  2.08),
    "LElbowRoll":     (-1.55, -0.03),
    "RShoulderPitch": (-2.08,  2.08),
    "RShoulderRoll":  (-1.33,  0.31),
    "RElbowYaw":      (-2.08,  2.08),
    "RElbowRoll":     ( 0.03,  1.55),
}

# MediaPipe landmark indices.
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,   R_ELBOW     = 13, 14
L_WRIST,   R_WRIST     = 15, 16


def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value x to the closed interval [lo, hi]."""
    return max(lo, min(hi, x))


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the angle between two 2D vectors in radians."""
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(math.acos(dot))


def _compute_arm_angles_from_kp_norm(
    kp_img_norm: np.ndarray,
    mirror: bool = True
) -> Dict[str, float]:
    """Convert normalized keypoints (33×2/3) to NAO arm joint angles.

    Approximates shoulder and elbow angles using the 2D positions of
    shoulder, elbow and wrist landmarks in normalized image coordinates.

    Args:
        kp_img_norm: Array of shape (33, 2/3) with (x, y, [z]) in [0, 1].
        mirror: If True, flip x horizontally to compensate for mirroring.

    Returns:
        Dictionary mapping joint names (ARM_JOINTS) to target angles in radians,
        clamped to the predefined joint limits.
    """
    def sel(i: int) -> np.ndarray:
        p = np.array([kp_img_norm[i, 0], kp_img_norm[i, 1]], dtype=np.float32)
        if mirror:
            p[0] = 1.0 - p[0]
        return p

    Ls, Le, Lw = sel(L_SHOULDER), sel(L_ELBOW), sel(L_WRIST)
    Rs, Re, Rw = sel(R_SHOULDER), sel(R_ELBOW), sel(R_WRIST)

    shoulder_width = float(np.linalg.norm(Ls - Rs) + 1e-6)
    out = {j: 0.0 for j in ARM_JOINTS}

    # Left arm
    Lu, Lf = Le - Ls, Lw - Le
    l_roll = math.atan2(Lu[0], -Lu[1])
    l_pitch = (Ls[1] - Lw[1]) / shoulder_width
    l_pitch = _clamp(l_pitch, -1.0, 1.0) * (math.pi / 2)
    l_elbow_roll = -abs(math.pi - _angle_between(Lu, Lf))
    l_elbow_yaw = 0.0

    out["LShoulderPitch"] = _clamp(l_pitch, *LIMITS["LShoulderPitch"])
    out["LShoulderRoll"]  = _clamp(l_roll,  *LIMITS["LShoulderRoll"])
    out["LElbowYaw"]      = _clamp(l_elbow_yaw, *LIMITS["LElbowYaw"])
    out["LElbowRoll"]     = _clamp(l_elbow_roll, *LIMITS["LElbowRoll"])

    # Right arm
    Ru, Rf = Re - Rs, Rw - Re
    r_roll = math.atan2(Ru[0], -Ru[1])
    r_pitch = (Rs[1] - Rw[1]) / shoulder_width
    r_pitch = _clamp(r_pitch, -1.0, 1.0) * (math.pi / 2)
    r_elbow_roll = +abs(math.pi - _angle_between(Ru, Rf))
    r_elbow_yaw = 0.0

    out["RShoulderPitch"] = _clamp(r_pitch, *LIMITS["RShoulderPitch"])
    out["RShoulderRoll"]  = _clamp(r_roll,  *LIMITS["RShoulderRoll"])
    out["RElbowYaw"]      = _clamp(r_elbow_yaw, *LIMITS["RElbowYaw"])
    out["RElbowRoll"]     = _clamp(r_elbow_roll, *LIMITS["RElbowRoll"])

    return out


def _set_robot_pose(
    nao_ip: str,
    angles: Dict[str, float],
    duration: float = 1.5,
    stiffness: float = 1.0
) -> None:
    """Connect to NAOqi and move the robot arms to the given angles.

    The robot is first moved (if possible) to 'StandInit', then arm
    stiffness is enabled and the arm joints are interpolated to the
    requested target angles.

    Args:
        nao_ip: IP address of the NAO robot (port 9559 is assumed).
        angles: Mapping from joint name to angle (radians).
        duration: Movement duration for each joint (seconds).
        stiffness: Arm stiffness value in [0, 1].
    """
    session = qi.Session()
    session.connect(f"tcp://{nao_ip}:9559")
    motion = session.service("ALMotion")
    posture = session.service("ALRobotPosture")

    try:
        posture.goToPosture("StandInit", 0.5)
    except Exception:
        # If this fails, just continue with arm motion.
        pass

    motion.setStiffnesses(["LArm", "RArm"], stiffness)
    names = ARM_JOINTS
    targets = [float(angles[j]) for j in names]
    times = [duration] * len(names)
    try:
        motion.angleInterpolation(names, targets, times, True)
    except Exception:
        # Fallback to setAngles if interpolation is not available.
        motion.setAngles(names, targets, 0.2)


def replicate_pose_from_kp(
    kp_img_norm: List[List[float]] | np.ndarray,
    nao_ip: str,
    mirror: bool = True,
    duration: float = 1.5
) -> Dict[str, float]:
    """Replicate a pose directly from a list/array of 33 keypoints.

    Args:
        kp_img_norm: List or np.ndarray of shape (33, 2/3) with normalized
            keypoints (x, y, [z]) in image coordinates.
        nao_ip: IP address of the robot (port 9559).
        mirror: If True, mirror the pose horizontally before mapping.
        duration: Movement duration in seconds.

    Returns:
        Dictionary of joint angles (in radians) used for the movement.
    """
    kp = np.asarray(kp_img_norm, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[0] != 33 or kp.shape[1] < 2:
        raise ValueError(f"kp_img_norm must be (33, 2/3), got {kp.shape}")
    angles = _compute_arm_angles_from_kp_norm(kp, mirror=mirror)
    _set_robot_pose(nao_ip, angles, duration=duration, stiffness=1.0)
    return angles


def replicate_pose(
    pose: Pose,
    nao_ip: str,
    mirror: bool = True,
    duration: float = 1.5
) -> Dict[str, float]:
    """Replicate a pose from a Pose dataclass instance.

    This is just a thin wrapper around `replicate_pose_from_kp` that
    takes a `Pose` object instead of a raw keypoint array.

    Args:
        pose: Pose object containing `kp_img_norm` with shape (33, 2/3).
        nao_ip: IP address of the robot (port 9559).
        mirror: If True, mirror the pose horizontally before mapping.
        duration: Movement duration in seconds.

    Returns:
        Dictionary of joint angles (in radians) used for the movement.
    """
    return replicate_pose_from_kp(pose.kp_img_norm, nao_ip, mirror=mirror, duration=duration)