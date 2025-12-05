"""
Control a NAO robot to reproduce a static pose recorded with MediaPipe.

This implementation:
- uses (x,y,z) from MediaPipe, flips image Y so +Y is up,
- optionally mirrors horizontally,
- centers coordinates on the shoulder midpoint and normalizes by shoulder width,
- computes shoulder pitch/roll and elbow flex with signed conventions,
- clamps results to NAO limits (radians),
- logs computed vectors/angles for debugging.

Notes:
- Mapping sign conventions may need a single-axis flip depending on your robot posture.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import qi

logger = logging.getLogger(__name__)

# --- Debug helpers ---------------------------------------------------------

def _format_angles(angles: Dict[str, float], to_degrees: bool = True) -> str:
    parts = []
    for name in ALL_JOINTS:  # Changed from ARM_JOINTS
        val = float(angles.get(name, 0.0))
        if to_degrees:
            parts.append(f"{name}={math.degrees(val):.1f}°")
        else:
            parts.append(f"{name}={val:.3f}rad")
    return ", ".join(parts)

def _debug_log_angles(angles: Dict[str, float], prefix: str = "ANGLES") -> None:
    try:
        logger.info("%s (rad): %s", prefix, _format_angles(angles, to_degrees=False))
        logger.info("%s (deg): %s", prefix, _format_angles(angles, to_degrees=True))
    except Exception:
        pass


# --- Pose container --------------------------------------------------------

@dataclass(frozen=True)
class Pose:
    """In-memory pose with 33 normalized keypoints."""
    kp_img_norm: np.ndarray

    def __post_init__(self):
        kp = np.asarray(self.kp_img_norm, dtype=np.float32)
        if kp.ndim != 2 or kp.shape[0] != 33 or kp.shape[1] < 2:
            raise ValueError(f"kp_img_norm must be (33, 2/3), got {kp.shape}")
        object.__setattr__(self, "kp_img_norm", kp)


# --- Constants -------------------------------------------------------------

ARM_JOINTS = [
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll",
]

# Stop including leg joints in the global joint list
ALL_JOINTS = ARM_JOINTS

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

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,   R_ELBOW     = 13, 14
L_WRIST,   R_WRIST     = 15, 16

# --- removed leg landmark constants and leg-related limits ---

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1n = v1 / (np.linalg.norm(v1) + 1e-8)
    v2n = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    return float(math.acos(dot))


# --- Elbow robustness tuning ----------------------------------------------

MIN_SEG_LEN = 0.08
ELBOW_DEADZONE = math.radians(3.0)  # Reduced from 6.0 to capture smaller bends
ELBOW_GAIN = 1.4  # Increased from 1.2 to use more of NAO's range

def _signed_elbow_flex(upper: np.ndarray, fore: np.ndarray) -> float:
    lu = float(np.linalg.norm(upper))
    lf = float(np.linalg.norm(fore))

    if lu < MIN_SEG_LEN or lf < MIN_SEG_LEN:
        logger.debug("Elbow skipped: short segment")
        return 0.0

    ang = _angle_between(upper, fore)

    if ang <= ELBOW_DEADZONE:
        logger.debug("Elbow in deadzone")
        return 0.0

    # Return the angle directly without the deadzone subtraction
    # to preserve the full bend range
    adjusted = ang * ELBOW_GAIN
    return float(adjusted)


# --- Calibration ------------------------------------------------------------

PITCH_SIGN = -1.0
PITCH_SCALE = 1.8
PITCH_OFFSET_L = 0.0
PITCH_OFFSET_R = 0.0

ROLL_SIGN = 1.0
ROLL_SCALE = 1.0

_CALIB: Dict[str, float] = {
    "shoulder_width_ref": None,
    "l_pitch_neutral": 0.0,
    "r_pitch_neutral": 0.0,
}

SWAP_SIDES_OVERRIDE: Optional[bool] = True


def calibrate_tpose(kp_img_norm: np.ndarray, mirror: bool = True, assume_tpose: bool = True) -> None:
    try:
        def sel3(i: int) -> np.ndarray:
            x = float(kp_img_norm[i, 0])
            y = float(kp_img_norm[i, 1])
            z = float(kp_img_norm[i, 2]) if kp_img_norm.shape[1] > 2 else 0.0
            if mirror:
                x = 1.0 - x
            return np.array([x, -y, z], dtype=np.float32)

        Ls = sel3(L_SHOULDER)
        Rs = sel3(R_SHOULDER)

        shoulder_width = float(np.linalg.norm(Ls - Rs) + 1e-6)
        _CALIB["shoulder_width_ref"] = shoulder_width

        def compute_pitch_from_points(s, e):
            u = (e - s) / max(1e-6, shoulder_width)
            vertical = u[1]
            horiz = math.sqrt(u[0] * u[0] + u[2] * u[2])
            return math.atan2(vertical, horiz)

        Le = sel3(L_ELBOW)
        Re = sel3(R_ELBOW)

        _CALIB["l_pitch_neutral"] = compute_pitch_from_points(Ls, Le)
        _CALIB["r_pitch_neutral"] = compute_pitch_from_points(Rs, Re)

    except Exception:
        logger.exception("Calibration failed")


# --- Angle computation ------------------------------------------------------

def _compute_arm_angles_from_kp_norm(kp_img_norm: np.ndarray, mirror: bool = True) -> Dict[str, float]:
    kp = np.asarray(kp_img_norm, dtype=np.float32)

    raw_lx = float(kp[L_SHOULDER, 0])
    raw_rx = float(kp[R_SHOULDER, 0])

    lx, rx = raw_lx, raw_rx
    if mirror:
        lx = 1.0 - lx
        rx = 1.0 - rx

    if SWAP_SIDES_OVERRIDE is None:
        swap_sides = lx > rx
    else:
        swap_sides = bool(SWAP_SIDES_OVERRIDE)

    if swap_sides:
        s_idx = (R_SHOULDER, L_SHOULDER)
        e_idx = (R_ELBOW,   L_ELBOW)
        w_idx = (R_WRIST,   L_WRIST)
    else:
        s_idx = (L_SHOULDER, R_SHOULDER)
        e_idx = (L_ELBOW,    R_ELBOW)
        w_idx = (L_WRIST,    R_WRIST)

    def sel3(i: int) -> np.ndarray:
        x = float(kp[i, 0])
        y = float(kp[i, 1])
        z = float(kp[i, 2]) if kp.shape[1] > 2 else 0.0
        if mirror:
            x = 1.0 - x
        return np.array([x, -y, z], dtype=np.float32)

    Ls, Rs = sel3(s_idx[0]), sel3(s_idx[1])
    Le, Re = sel3(e_idx[0]), sel3(e_idx[1])
    Lw, Rw = sel3(w_idx[0]), sel3(w_idx[1])

    shoulder_mid = (Ls + Rs) * 0.5
    shoulder_width = float(np.linalg.norm(Ls - Rs) + 1e-6)

    def norm(p): return (p - shoulder_mid) / shoulder_width

    Ls_n, Le_n, Lw_n = norm(Ls), norm(Le), norm(Lw)
    Rs_n, Re_n, Rw_n = norm(Rs), norm(Re), norm(Rw)

    Lu = Le_n - Ls_n
    Lf = Lw_n - Le_n
    Ru = Re_n - Rs_n
    Rf = Rw_n - Re_n

    def compute_pitch(u):
        vertical = u[1]
        horiz = math.sqrt(u[0] * u[0] + u[2] * u[2])
        return math.atan2(vertical, horiz)

    def compute_roll(u):
        return math.atan2(u[0], max(1e-6, u[1]))

    raw_l_pitch = compute_pitch(Lu)
    raw_r_pitch = compute_pitch(Ru)
    raw_l_roll = compute_roll(Lu)
    raw_r_roll = compute_roll(Ru)

    raw_l_elbow = _signed_elbow_flex(Lu, Lf)
    raw_r_elbow = _signed_elbow_flex(Ru, Rf)

    l_pitch = PITCH_SIGN * ((raw_l_pitch - _CALIB["l_pitch_neutral"]) * PITCH_SCALE + PITCH_OFFSET_L)
    r_pitch = PITCH_SIGN * ((raw_r_pitch - _CALIB["r_pitch_neutral"]) * PITCH_SCALE + PITCH_OFFSET_R)

    l_roll = ROLL_SIGN * (raw_l_roll * ROLL_SCALE)
    r_roll = ROLL_SIGN * (raw_r_roll * ROLL_SCALE)

    l_elbow_roll = -(raw_l_elbow * ELBOW_GAIN)
    r_elbow_roll = +(raw_r_elbow * ELBOW_GAIN)

    arm_angles = {
        "LShoulderPitch": _clamp(l_pitch, *LIMITS["LShoulderPitch"]),
        "LShoulderRoll":  _clamp(l_roll,  *LIMITS["LShoulderRoll"]),
        "LElbowYaw":      0.0,
        "LElbowRoll":     _clamp(l_elbow_roll, *LIMITS["LElbowRoll"]),

        "RShoulderPitch": _clamp(r_pitch, *LIMITS["RShoulderPitch"]),
        "RShoulderRoll":  _clamp(r_roll,  *LIMITS["RShoulderRoll"]),
        "RElbowYaw":      0.0,
        "RElbowRoll":     _clamp(r_elbow_roll, *LIMITS["RElbowRoll"]),
    }

    # Only return arm angles (no legs)
    return arm_angles


# --- Robot control ---------------------------------------------------------

def _set_robot_pose(
    nao_ip: str,
    angles: Dict[str, float],
    duration: float = 1.5,
    stiffness: float = 1.0,
    reset_to_standinit: bool = False,
) -> None:
    session = qi.Session()
    session.connect(f"tcp://{nao_ip}:9559")

    motion = session.service("ALMotion")
    posture = session.service("ALRobotPosture")

    if reset_to_standinit:
        try:
            posture.goToPosture("StandInit", 0.5)
        except Exception:
            pass

    try:
        # Set stiffness only for the arms to avoid leg movement
        motion.setStiffnesses(["LArm", "RArm"], stiffness)
    except Exception:
        pass

    names = ARM_JOINTS  # only arm joints
    targets = [float(angles.get(j, 0.0)) for j in names]
    times = [duration] * len(names)

    _debug_log_angles({n: t for n, t in zip(names, targets)}, prefix="Sending joint targets")

    try:
        motion.angleInterpolation(names, targets, times, True)
    except Exception:
        try:
            motion.setAngles(names, targets, 0.2)
        except Exception:
            logger.exception("Failed to move NAO to target angles.")


# --- Public API ------------------------------------------------------------

def replicate_pose_from_kp(
    kp_img_norm: List[List[float]] | np.ndarray,
    nao_ip: str,
    mirror: bool = True,
    duration: float = 1.5,
    reset_to_standinit: bool = False,
) -> Dict[str, float]:
    kp = np.asarray(kp_img_norm, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[0] != 33 or kp.shape[1] < 2:
        raise ValueError(f"kp_img_norm must be (33, 2/3), got {kp.shape}")

    angles = _compute_arm_angles_from_kp_norm(kp, mirror=mirror)

    _debug_log_angles(angles, prefix="Computed pose angles")

    _set_robot_pose(
        nao_ip,
        angles,
        duration=duration,
        stiffness=1.0,
        reset_to_standinit=reset_to_standinit,
    )

    return angles


def replicate_pose(
    pose: Pose,
    nao_ip: str,
    mirror: bool = True,
    duration: float = 1.5,
    reset_to_standinit: bool = False,
) -> Dict[str, float]:
    return replicate_pose_from_kp(
        pose.kp_img_norm,
        nao_ip,
        mirror=mirror,
        duration=duration,
        reset_to_standinit=reset_to_standinit,
    )
