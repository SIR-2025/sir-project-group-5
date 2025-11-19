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
- Mapping sign conventions may need a single-axis flip depending on your robot posture; test with known poses and invert signs if needed.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import qi

# logger
logger = logging.getLogger(__name__)

# --- new: joint debug helpers ---------------------------------------------
def _format_angles(angles: Dict[str, float], to_degrees: bool = True) -> str:
    parts = []
    for name in ARM_JOINTS:
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
        # best-effort; do not raise in production path
        pass

@dataclass(frozen=True)
class Pose:
    """In-memory pose with 33 normalized keypoints."""
    kp_img_norm: np.ndarray

    def __post_init__(self):
        kp = np.asarray(self.kp_img_norm, dtype=np.float32)
        if kp.ndim != 2 or kp.shape[0] != 33 or kp.shape[1] < 2:
            raise ValueError(f"kp_img_norm must be (33, 2/3), got {kp.shape}")
        object.__setattr__(self, "kp_img_norm", kp)


ARM_JOINTS = [
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll",
]

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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1n = v1 / (np.linalg.norm(v1) + 1e-8)
    v2n = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    return float(math.acos(dot))


# --- NEW: elbow robustness tuning ----------------------------------------
# Minimum normalized segment length to consider the joint reliable (units: shoulder-width normalized)
MIN_SEG_LEN = 0.08     # increase if keypoints often too short / noisy
ELBOW_DEADZONE = math.radians(6.0)  # angles below this are treated as zero
ELBOW_GAIN = 1.2       # keep existing gain but applied after deadzone

def _signed_elbow_flex(upper: np.ndarray, fore: np.ndarray, forward_axis: np.ndarray = np.array([0.0, 0.0, 1.0])) -> float:
    """
    Robust elbow flex estimator:
    - returns 0 if segments too short or angle within deadzone
    - returns (angle - deadzone) * ELBOW_GAIN otherwise (non-negative)
    - caller applies left/right sign convention
    """
    # lengths (after normalization in calling code)
    lu = float(np.linalg.norm(upper))
    lf = float(np.linalg.norm(fore))

    # if either segment is too short, treat as straight / unreliable
    if lu < MIN_SEG_LEN or lf < MIN_SEG_LEN:
        logger.debug("Elbow skipped: short segment lu=%.3f lf=%.3f", lu, lf)
        return 0.0

    # compute raw angle between segments
    ang = _angle_between(upper, fore)  # 0..pi

    # deadzone to avoid tiny jitter causing bend
    if ang <= ELBOW_DEADZONE:
        logger.debug("Elbow within deadzone: ang=%.3f deg", math.degrees(ang))
        return 0.0

    adjusted = max(0.0, (ang - ELBOW_DEADZONE) * ELBOW_GAIN)
    logger.debug("Elbow raw=%.3f deg adjusted=%.3f deg lu=%.3f lf=%.3f",
                 math.degrees(ang), math.degrees(adjusted), lu, lf)
    return float(adjusted)


def calibrate_tpose(kp_img_norm: np.ndarray, mirror: bool = True, assume_tpose: bool = True) -> None:
    """
    Calibrate using a pose where arms are in a known reference (T-pose or relaxed down).
    Call this once (e.g., user holds T-pose) and then calls to replicate_pose will use
    the resulting neutral offsets to align human neutral -> NAO neutral.
    """
    try:
        # reuse sel3/norm logic quickly: copy of small portion used in compute fn
        def sel3(i: int) -> np.ndarray:
            x = float(kp_img_norm[i, 0])
            y = float(kp_img_norm[i, 1])
            z = float(kp_img_norm[i, 2]) if kp_img_norm.shape[1] > 2 else 0.0
            if mirror:
                x = 1.0 - x
            return np.array([x, -y, z], dtype=np.float32)

        Ls = sel3(L_SHOULDER); Rs = sel3(R_SHOULDER)
        shoulder_width = float(np.linalg.norm(Ls - Rs) + 1e-6)
        _CALIB["shoulder_width_ref"] = shoulder_width

        # compute neutral pitches from current kp using same pitch function:
        def compute_pitch_from_points(s, e):
            u = (e - s) / max(1e-6, shoulder_width)
            vertical = u[1]
            horiz = math.sqrt(u[0] * u[0] + u[2] * u[2])
            return math.atan2(vertical, horiz)

        Le = sel3(L_ELBOW); Re = sel3(R_ELBOW)
        _CALIB["l_pitch_neutral"] = compute_pitch_from_points(Ls, Le)
        _CALIB["r_pitch_neutral"] = compute_pitch_from_points(Rs, Re)

        logger.info("Calibration done: shoulder_w=%.3f l_neutral=%.3f r_neutral=%.3f",
                    _CALIB["shoulder_width_ref"], _CALIB["l_pitch_neutral"], _CALIB["r_pitch_neutral"])
    except Exception:
        logger.exception("Calibration failed")


# --- NEW: calibration / tuning parameters --------------------------------
# Use these to tune mapping. Default values are conservative; increase
# PITCH_SCALE if NAO under-raises his arms, reduce if overshoots.
PITCH_SIGN = -1.0   # flip if up/down reversed
PITCH_SCALE = 1.8   # >1 to amplify pitch (raising/lowering)
PITCH_OFFSET_L = 0.0  # rad, subtract this from left pitch after scaling
PITCH_OFFSET_R = 0.0  # rad, subtract this from right pitch after scaling

ROLL_SIGN = 1.0     # flip if roll direction is wrong
ROLL_SCALE = 1.

# Calibration storage (filled by calibrate_tpose)
_CALIB: Dict[str, float] = {
    "shoulder_width_ref": None,
    "l_pitch_neutral": 0.0,
    "r_pitch_neutral": 0.0,
}

# Add override to force left/right mapping during debugging:
# None = auto-detect, True = force "swap" (treat image L->robot R), False = force no-swap.
SWAP_SIDES_OVERRIDE: Optional[bool] = True


def _compute_arm_angles_from_kp_norm(kp_img_norm: np.ndarray, mirror: bool = True) -> Dict[str, float]:
    """
    Convert normalized keypoints (33×2/3) to NAO arm joint targets (radians).

    Auto-detects if left/right keypoints are swapped in the image and corrects mapping.
    """
    # quick validation
    kp = np.asarray(kp_img_norm, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[0] < max(L_SHOULDER, R_SHOULDER) + 1:
        raise ValueError("kp_img_norm shape unexpected")

    # read raw shoulder x positions (image coords)
    raw_lx = float(kp[L_SHOULDER, 0])
    raw_rx = float(kp[R_SHOULDER, 0])

    # Determine whether left/right appear swapped in image coordinates
    lx = raw_lx
    rx = raw_rx
    if mirror:
        lx = 1.0 - lx
        rx = 1.0 - rx

    # auto-detect or apply override
    if SWAP_SIDES_OVERRIDE is None:
        swap_sides = lx > rx  # left shoulder appears to the right of right shoulder -> swapped
    else:
        swap_sides = bool(SWAP_SIDES_OVERRIDE)

    # Log diagnostic info so you can see why mapping was chosen
    try:
        logger.info(
            "Side-map debug: raw_lx=%.3f raw_rx=%.3f mirror=%s -> adj_lx=%.3f adj_rx=%.3f swap=%s override=%s",
            raw_lx, raw_rx, mirror, lx, rx, swap_sides, SWAP_SIDES_OVERRIDE,
        )
    except Exception:
        pass

    # choose index mapping based on swap detection
    if swap_sides:
        s_idx = (R_SHOULDER, L_SHOULDER)
        e_idx = (R_ELBOW,   L_ELBOW)
        w_idx = (R_WRIST,   L_WRIST)
        mapped = "swapped (image L->robot R)"
    else:
        s_idx = (L_SHOULDER, R_SHOULDER)
        e_idx = (L_ELBOW,    R_ELBOW)
        w_idx = (L_WRIST,    R_WRIST)
        mapped = "normal (image L->robot L)"

    logger.info("Using index mapping: %s (s_idx=%s e_idx=%s w_idx=%s)", mapped, s_idx, e_idx, w_idx)

    # helper to extract 3D point from kp array and apply mirror + y-flip
    def sel3_index(i: int) -> np.ndarray:
        x = float(kp[i, 0])
        y = float(kp[i, 1])
        z = float(kp[i, 2]) if kp.shape[1] > 2 else 0.0
        if mirror:
            x = 1.0 - x
        return np.array([x, -y, z], dtype=np.float32)

    # select points using possibly-swapped indices
    Ls, Rs = sel3_index(s_idx[0]), sel3_index(s_idx[1])
    Le, Re = sel3_index(e_idx[0]), sel3_index(e_idx[1])
    Lw, Rw = sel3_index(w_idx[0]), sel3_index(w_idx[1])

    # normalize around shoulder midpoint and shoulder width
    shoulder_mid = (Ls + Rs) * 0.5
    shoulder_width = float(np.linalg.norm(Ls - Rs) + 1e-6)
    def norm_coord(p: np.ndarray) -> np.ndarray:
        return (p - shoulder_mid) / shoulder_width

    Ls_n, Le_n, Lw_n = norm_coord(Ls), norm_coord(Le), norm_coord(Lw)
    Rs_n, Re_n, Rw_n = norm_coord(Rs), norm_coord(Re), norm_coord(Rw)

    # compute vectors (shoulder->elbow, elbow->wrist)
    Lu = Le_n - Ls_n    # left upper arm
    Lf = Lw_n - Le_n    # left forearm
    Ru = Re_n - Rs_n
    Rf = Rw_n - Re_n

    # small helpers
    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-8)

    def compute_pitch(u: np.ndarray) -> float:
        vertical = u[1]
        horiz = math.sqrt(u[0] * u[0] + u[2] * u[2])
        return math.atan2(vertical, horiz)

    def compute_roll(u: np.ndarray) -> float:
        return math.atan2(u[0], max(1e-6, u[1]))

    def compute_elbow(u: np.ndarray, f: np.ndarray) -> float:
        return _signed_elbow_flex(u, f)

    # raw angles
    raw_l_pitch = compute_pitch(Lu)
    raw_r_pitch = compute_pitch(Ru)
    raw_l_roll = compute_roll(Lu)
    raw_r_roll = compute_roll(Ru)
    raw_l_elbow = compute_elbow(Lu, Lf)
    raw_r_elbow = compute_elbow(Ru, Rf)

    # apply existing scale/sign/calibration parameters
    l_neutral = _CALIB.get("l_pitch_neutral") or 0.0
    r_neutral = _CALIB.get("r_pitch_neutral") or 0.0

    l_pitch = PITCH_SIGN * ((raw_l_pitch - l_neutral) * PITCH_SCALE + PITCH_OFFSET_L)
    r_pitch = PITCH_SIGN * ((raw_r_pitch - r_neutral) * PITCH_SCALE + PITCH_OFFSET_R)

    l_roll = ROLL_SIGN * (raw_l_roll * ROLL_SCALE)
    r_roll = ROLL_SIGN * (raw_r_roll * ROLL_SCALE)

    l_elbow_roll = - (raw_l_elbow * ELBOW_GAIN)  # left convention
    r_elbow_roll = + (raw_r_elbow * ELBOW_GAIN)

    l_elbow_yaw = 0.0
    r_elbow_yaw = 0.0

    # assemble and clamp
    out: Dict[str, float] = {}
    out["LShoulderPitch"] = _clamp(l_pitch, *LIMITS["LShoulderPitch"])
    out["LShoulderRoll"]  = _clamp(l_roll,  *LIMITS["LShoulderRoll"])
    out["LElbowYaw"]      = _clamp(l_elbow_yaw, *LIMITS["LElbowYaw"])
    out["LElbowRoll"]     = _clamp(l_elbow_roll, *LIMITS["LElbowRoll"])

    out["RShoulderPitch"] = _clamp(r_pitch, *LIMITS["RShoulderPitch"])
    out["RShoulderRoll"]  = _clamp(r_roll,  *LIMITS["RShoulderRoll"])
    out["RElbowYaw"]      = _clamp(r_elbow_yaw, *LIMITS["RElbowYaw"])
    out["RElbowRoll"]     = _clamp(r_elbow_roll, *LIMITS["RElbowRoll"])

    # debug
    try:
        logger.info("swap_sides=%s mirror=%s | Lpitch=%.3f Lroll=%.3f LER=%.3f | Rpitch=%.3f Rroll=%.3f RER=%.3f",
                    swap_sides, mirror,
                    out["LShoulderPitch"], out["LShoulderRoll"], out["LElbowRoll"],
                    out["RShoulderPitch"], out["RShoulderRoll"], out["RElbowRoll"])
    except Exception:
        pass

    return out


def _set_robot_pose(nao_ip: str, angles: Dict[str, float], duration: float = 1.5, stiffness: float = 1.0) -> None:
    """Connect to NAOqi and move to a static pose (angles in radians)."""
    session = qi.Session()
    session.connect(f"tcp://{nao_ip}:9559")
    motion = session.service("ALMotion")
    posture = session.service("ALRobotPosture")

    try:
        posture.goToPosture("StandInit", 0.5)
    except Exception:
        pass

    # ensure stiffness for arms
    try:
        motion.setStiffnesses(["LArm", "RArm"], stiffness)
    except Exception:
        pass

    names = ARM_JOINTS
    targets = [float(angles.get(j, 0.0)) for j in names]
    times = [duration] * len(names)

    # --- new: log joint targets right before moving the robot --------------
    _debug_log_angles({n: t for n, t in zip(names, targets)}, prefix="Sending joint targets")

    try:
        motion.angleInterpolation(names, targets, times, True)
    except Exception:
        # fallback to immediate set (less safe)
        try:
            motion.setAngles(names, targets, 0.2)
        except Exception:
            logger.exception("Failed to move NAO to target angles.")


def replicate_pose_from_kp(
    kp_img_norm: List[List[float]] | np.ndarray,
    nao_ip: str,
    mirror: bool = True,
    duration: float = 1.5
) -> Dict[str, float]:
    """
    Replicate a pose directly from a list/array of 33 keypoints.

    Returns the angles dict used for the robot.
    """
    kp = np.asarray(kp_img_norm, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[0] != 33 or kp.shape[1] < 2:
        raise ValueError(f"kp_img_norm must be (33, 2/3), got {kp.shape}")
    angles = _compute_arm_angles_from_kp_norm(kp, mirror=mirror)

    # --- new: log computed joint angles for debugging ----------------------
    _debug_log_angles(angles, prefix="Computed pose angles")

    _set_robot_pose(nao_ip, angles, duration=duration, stiffness=1.0)
    return angles


def replicate_pose(
    pose: Pose,
    nao_ip: str,
    mirror: bool = True,
    duration: float = 1.5
) -> Dict[str, float]:
    """
    Replicate a pose from a Pose dataclass.
    """
    return replicate_pose_from_kp(pose.kp_img_norm, nao_ip, mirror=mirror, duration=duration)