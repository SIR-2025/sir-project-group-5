"""
Control a NAO robot to reproduce a static pose recorded with MediaPipe.

Focuses on X-Y plane poses (arms up/down, left/right) since depth (Z) from
MediaPipe is unreliable without stereo cameras.

NAO Shoulder Joint Behavior:
- ShoulderPitch: rotates arm forward/backward and up/down
  - 0° = arm pointing forward horizontal
  - +90° = arm pointing straight down
  - -90° = arm pointing straight up
- ShoulderRoll: rotates arm away from / toward body
  - For T-pose: need pitch ≈ +90° AND roll at max
  - Roll operates in the plane set by pitch

Joint conventions:
- ShoulderPitch: negative = up, positive = down
- ShoulderRoll: L positive = away, R negative = away
- ElbowRoll: L negative = bent, R positive = bent
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import qi

logger = logging.getLogger(__name__)


ARM_JOINTS = [
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll",
]

ALL_JOINTS = ARM_JOINTS

# NAO joint limits in radians
LIMITS: Dict[str, Tuple[float, float]] = {
    "LShoulderPitch": (-2.08,  2.08),   # -119° to +119°
    "LShoulderRoll":  (-0.31,  1.33),   # -18° to +76°
    "LElbowYaw":      (-2.08,  2.08),   # -119° to +119°
    "LElbowRoll":     (-1.55, -0.03),   # -89° to -2° (bent = negative)
    "RShoulderPitch": (-2.08,  2.08),   # -119° to +119°
    "RShoulderRoll":  (-1.33,  0.31),   # -76° to +18°
    "RElbowYaw":      (-2.08,  2.08),   # -119° to +119°
    "RElbowRoll":     ( 0.03,  1.55),   # +2° to +89° (bent = positive)
}

# MediaPipe landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,   R_ELBOW     = 13, 14
L_WRIST,   R_WRIST     = 15, 16
L_HIP,     R_HIP       = 23, 24

# Movement settings
DEFAULT_DURATION = 2.5
DEFAULT_SPEED_FRACTION = 0.15

# Tuning parameters
MIN_SEGMENT_LEN = 0.03
ELBOW_STRAIGHT_THRESHOLD = math.radians(15.0)


def _format_angles(angles: Dict[str, float], to_degrees: bool = True) -> str:
    parts = []
    for name in ALL_JOINTS:
        val = float(angles.get(name, 0.0))
        if to_degrees:
            parts.append(f"{name}={math.degrees(val):.1f}°")
        else:
            parts.append(f"{name}={val:.3f}rad")
    return ", ".join(parts)


def _debug_log_angles(angles: Dict[str, float], prefix: str = "ANGLES") -> None:
    try:
        logger.info("%s: %s", prefix, _format_angles(angles, to_degrees=True))
    except Exception:
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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_asin(x: float) -> float:
    return math.asin(max(-1.0, min(1.0, x)))


def _safe_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, x)))


def _compute_single_arm_angles(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    is_left: bool,
) -> Dict[str, float]:
    """
    Compute shoulder and elbow angles for one arm.
    
    NAO's shoulder joint system:
    - ShoulderPitch: rotation around the lateral (side-to-side) axis
      - 0° = arm horizontal, pointing FORWARD
      - +90° = arm pointing DOWN
      - -90° = arm pointing UP (overhead)
    - ShoulderRoll: rotation around the front-to-back axis (AFTER pitch)
      - Left arm: +roll = arm moves away from body (toward left)
      - Right arm: -roll = arm moves away from body (toward right)
    
    For T-pose (arms straight out to sides):
      - ShoulderPitch ≈ 0° (arm horizontal)
      - ShoulderRoll = max (Left: +76°, Right: -76°)
    
    For arms straight up:
      - ShoulderPitch = -90°
      - ShoulderRoll ≈ 0°
    
    For arms straight down:
      - ShoulderPitch = +90°
      - ShoulderRoll ≈ 0° (or small value to keep arms slightly away from body)
    """
    prefix = "L" if is_left else "R"
    
    # Compute arm vectors
    upper_arm = elbow - shoulder
    forearm = wrist - elbow
    
    upper_len = np.linalg.norm(upper_arm)
    fore_len = np.linalg.norm(forearm)
    
    # Default: arms relaxed at sides
    if upper_len < MIN_SEGMENT_LEN:
        return {
            f"{prefix}ShoulderPitch": math.radians(80.0),
            f"{prefix}ShoulderRoll": math.radians(10.0) if is_left else math.radians(-10.0),
            f"{prefix}ElbowYaw": 0.0,
            f"{prefix}ElbowRoll": 0.0,
        }
    
    # Normalize upper arm vector
    ua = upper_arm / upper_len
    
    # Our coordinate system (after transform in _compute_arm_angles_from_kp_norm):
    # X: positive = robot's left (lateral)
    # Y: positive = up (vertical)
    # Z: 0 (we ignore depth)
    
    lateral = ua[0]   # positive = arm pointing toward robot's left
    vertical = ua[1]  # positive = arm pointing up
    
    logger.debug(f"{prefix} arm direction: lateral={lateral:.3f}, vertical={vertical:.3f}")

    # === SHOULDER PITCH ===
    # Pitch controls the up/down angle of the arm in the sagittal plane
    #
    # NAO convention:
    #   pitch = -90° → arm points UP
    #   pitch = 0°   → arm points FORWARD (horizontal)
    #   pitch = +90° → arm points DOWN
    #
    # Since we're working in 2D (X-Y plane, no forward/back), we interpret:
    #   - vertical component directly maps to pitch
    #   - vertical = +1 (up): pitch = -90°
    #   - vertical = -1 (down): pitch = +90°
    #   - vertical = 0 (horizontal/sideways): pitch = 0° (arm horizontal)
    #
    # Simple mapping: pitch = -arcsin(vertical) * (π/2) / (π/2) = -arcsin(vertical)
    # But arcsin only gives -90° to +90°, which is exactly our range!

    pitch = -_safe_asin(vertical) * (math.pi / 2) / (math.pi / 2)
    # Simplifies to:
    pitch = -_safe_asin(vertical)
    
    # Verify:
    # - vertical = 1 (up): pitch = -asin(1) = -π/2 = -90° ✓
    # - vertical = 0 (horizontal): pitch = -asin(0) = 0° ✓
    # - vertical = -1 (down): pitch = -asin(-1) = π/2 = +90° ✓
    
    # === SHOULDER ROLL ===
    # Roll controls how far the arm extends sideways (away from body)
    #
    # For left arm:
    #   roll = 0° → arm close to body
    #   roll = +76° (max) → arm extended fully to the left (T-pose)
    #
    # For right arm:
    #   roll = 0° → arm close to body  
    #   roll = -76° (max) → arm extended fully to the right (T-pose)
    #
    # The lateral component tells us how much sideways extension:
    #   - lateral = +1: arm pointing fully left → left arm needs max roll, right arm needs min
    #   - lateral = -1: arm pointing fully right → right arm needs max roll, left arm needs min
    #   - lateral = 0: arm not pointing sideways → roll ≈ 0 (or small default)
    
    if is_left:
        # Left arm: positive lateral = arm pointing left = more roll
        # Map lateral [0, 1] to roll [small, max]
        # When arm points straight left (lateral ≈ 1), want max roll ≈ 76°
        # When arm points straight up/down (lateral ≈ 0), want small roll
        roll = _safe_asin(abs(lateral)) * (LIMITS["LShoulderRoll"][1] / (math.pi / 2))
        # Scale to actual limit
        roll = abs(lateral) * LIMITS["LShoulderRoll"][1]
        # Keep a minimum roll to prevent arm clipping into body
        roll = max(roll, math.radians(5.0))
    else:
        # Right arm: negative lateral = arm pointing right = more negative roll
        # Map lateral [-1, 0] to roll [min (negative), small]
        roll = -abs(lateral) * abs(LIMITS["RShoulderRoll"][0])
        # Keep a minimum roll magnitude
        roll = min(roll, math.radians(-5.0))
    
    logger.debug(f"{prefix} arm: pitch={math.degrees(pitch):.1f}°, roll={math.degrees(roll):.1f}°")
    
    # === ELBOW ANGLES ===
    elbow_roll = 0.0
    elbow_yaw = 0.0
    
    if fore_len >= MIN_SEGMENT_LEN:
        fa = forearm / fore_len
        
        # Elbow bend: angle between upper arm and forearm
        dot = float(np.dot(ua, fa))
        dot = max(-1.0, min(1.0, dot))
        bend_angle = math.acos(dot)  # 0 = straight, π = fully bent back
        
        # Only apply if significant bend
        if bend_angle > ELBOW_STRAIGHT_THRESHOLD:
            if is_left:
                elbow_roll = -bend_angle  # Left: negative = bent
            else:
                elbow_roll = bend_angle   # Right: positive = bent
        
        # === ELBOW YAW ===
        # Yaw rotates the forearm around the upper arm axis
        # This determines if a bent forearm points up, down, forward, etc.
        #
        # For the "L-shape with hands up" pose:
        # - Upper arm horizontal to the side
        # - Forearm pointing UP
        # - This requires elbow yaw to rotate the forearm into the vertical plane
        
        if bend_angle > ELBOW_STRAIGHT_THRESHOLD:
            # Project forearm onto plane perpendicular to upper arm
            fa_along_ua = np.dot(fa, ua) * ua
            fa_perp = fa - fa_along_ua
            fa_perp_len = np.linalg.norm(fa_perp)
            
            if fa_perp_len > 0.1:
                fa_perp_norm = fa_perp / fa_perp_len
                
                # Reference direction: "down" in the perpendicular plane
                # When elbow_yaw = 0, forearm bends "down" relative to the arm plane
                down = np.array([0.0, -1.0, 0.0])
                down_along_ua = np.dot(down, ua) * ua
                down_perp = down - down_along_ua
                down_perp_len = np.linalg.norm(down_perp)
                
                if down_perp_len > 0.1:
                    down_perp_norm = down_perp / down_perp_len
                    
                    # Angle from "down" to actual forearm direction
                    cos_yaw = np.dot(fa_perp_norm, down_perp_norm)
                    cos_yaw = max(-1.0, min(1.0, cos_yaw))
                    
                    # Sign from cross product
                    cross = np.cross(down_perp_norm, fa_perp_norm)
                    sign = 1.0 if np.dot(cross, ua) > 0 else -1.0
                    
                    elbow_yaw = sign * math.acos(cos_yaw)
                    
                    # Right arm has inverted yaw convention
                    if not is_left:
                        elbow_yaw = -elbow_yaw

    # Clamp all to NAO joint limits
    pitch = _clamp(pitch, *LIMITS[f"{prefix}ShoulderPitch"])
    roll = _clamp(roll, *LIMITS[f"{prefix}ShoulderRoll"])
    elbow_yaw = _clamp(elbow_yaw, *LIMITS[f"{prefix}ElbowYaw"])
    elbow_roll = _clamp(elbow_roll, *LIMITS[f"{prefix}ElbowRoll"])
    
    logger.debug(f"{prefix} arm FINAL: pitch={math.degrees(pitch):.1f}°, roll={math.degrees(roll):.1f}°, "
                 f"elbow_yaw={math.degrees(elbow_yaw):.1f}°, elbow_roll={math.degrees(elbow_roll):.1f}°")
    
    return {
        f"{prefix}ShoulderPitch": pitch,
        f"{prefix}ShoulderRoll": roll,
        f"{prefix}ElbowYaw": elbow_yaw,
        f"{prefix}ElbowRoll": elbow_roll,
    }


def _compute_arm_angles_from_kp_norm(
    kp_img_norm: np.ndarray,
    mirror: bool = True
) -> Dict[str, float]:
    """
    Compute NAO arm angles from MediaPipe keypoints.
    
    MediaPipe gives:
    - x: 0-1, left to right in image
    - y: 0-1, top to bottom in image
    - z: depth (unreliable, ignored)
    
    We transform to:
    - X: positive = robot's left
    - Y: positive = up
    - Z: ignored
    """
    kp = np.asarray(kp_img_norm, dtype=np.float32)
    
    def get_point(idx: int) -> np.ndarray:
        x = float(kp[idx, 0])  # 0-1, left to right
        y = float(kp[idx, 1])  # 0-1, top to bottom
        
        # Transform to robot frame (X-Y plane only)
        if mirror:
            robot_x = 1.0 - x  # Flip X for mirror
        else:
            robot_x = x
        
        robot_y = 1.0 - y  # Flip Y: image top = up
        robot_z = 0.0      # Ignore Z
        
        return np.array([robot_x, robot_y, robot_z], dtype=np.float32)
    
    # Get keypoints (MediaPipe labels: person's left/right)
    mp_l_shoulder = get_point(L_SHOULDER)
    mp_r_shoulder = get_point(R_SHOULDER)
    mp_l_elbow = get_point(L_ELBOW)
    mp_r_elbow = get_point(R_ELBOW)
    mp_l_wrist = get_point(L_WRIST)
    mp_r_wrist = get_point(R_WRIST)
    
    # When mirrored:
    # - Person's left arm appears on right side of image
    # - After X flip, person's left is at high X (robot's left side)
    # - Robot's left arm should mimic person's left arm (mirrored)
    # So: robot_left = mediapipe_left (after coordinate transform)
    
    if mirror:
        robot_l_shoulder = mp_l_shoulder
        robot_r_shoulder = mp_r_shoulder
        robot_l_elbow = mp_l_elbow
        robot_r_elbow = mp_r_elbow
        robot_l_wrist = mp_l_wrist
        robot_r_wrist = mp_r_wrist
    else:
        # Person facing away - swap left/right
        robot_l_shoulder = mp_r_shoulder
        robot_r_shoulder = mp_l_shoulder
        robot_l_elbow = mp_r_elbow
        robot_r_elbow = mp_l_elbow
        robot_l_wrist = mp_r_wrist
        robot_r_wrist = mp_l_wrist
    
    # Normalize by shoulder width
    shoulder_mid = (robot_l_shoulder + robot_r_shoulder) * 0.5
    shoulder_width = np.linalg.norm(robot_l_shoulder - robot_r_shoulder)
    if shoulder_width < 0.05:
        shoulder_width = 0.15
    
    def norm(p: np.ndarray) -> np.ndarray:
        return (p - shoulder_mid) / shoulder_width
    
    l_shoulder_n = norm(robot_l_shoulder)
    r_shoulder_n = norm(robot_r_shoulder)
    l_elbow_n = norm(robot_l_elbow)
    r_elbow_n = norm(robot_r_elbow)
    l_wrist_n = norm(robot_l_wrist)
    r_wrist_n = norm(robot_r_wrist)
    
    # Compute angles
    left_angles = _compute_single_arm_angles(
        l_shoulder_n, l_elbow_n, l_wrist_n, is_left=True
    )
    right_angles = _compute_single_arm_angles(
        r_shoulder_n, r_elbow_n, r_wrist_n, is_left=False
    )
    
    all_angles = {**left_angles, **right_angles}
    _debug_log_angles(all_angles, prefix="Final angles")
    
    return all_angles


def _set_robot_pose(
    nao_ip: str,
    angles: Dict[str, float],
    duration: float = DEFAULT_DURATION,
    stiffness: float = 0.7,
    reset_to_standinit: bool = False,
) -> None:
    """Send joint angles to NAO with smooth interpolation."""
    session = qi.Session()
    session.connect(f"tcp://{nao_ip}:9559")

    motion = session.service("ALMotion")
    posture = session.service("ALRobotPosture")

    if reset_to_standinit:
        try:
            posture.goToPosture("StandInit", 0.3)
        except Exception:
            pass

    try:
        motion.setStiffnesses(["LArm", "RArm"], stiffness)
    except Exception:
        pass

    names = ARM_JOINTS
    targets = [float(angles.get(j, 0.0)) for j in names]
    times = [duration] * len(names)

    _debug_log_angles({n: t for n, t in zip(names, targets)}, prefix="Sending to NAO")

    try:
        motion.angleInterpolation(names, targets, times, True)
    except Exception as e:
        logger.warning(f"angleInterpolation failed: {e}, using setAngles")
        try:
            motion.setAngles(names, targets, DEFAULT_SPEED_FRACTION)
        except Exception:
            logger.exception("Failed to set angles on NAO")


def replicate_pose_from_kp(
    kp_img_norm: List[List[float]] | np.ndarray,
    nao_ip: str,
    mirror: bool = True,
    duration: float = DEFAULT_DURATION,
    reset_to_standinit: bool = False,
) -> Dict[str, float]:
    """Make NAO replicate a pose from MediaPipe keypoints."""
    kp = np.asarray(kp_img_norm, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[0] != 33 or kp.shape[1] < 2:
        raise ValueError(f"kp_img_norm must be (33, 2/3), got {kp.shape}")

    angles = _compute_arm_angles_from_kp_norm(kp, mirror=mirror)

    _set_robot_pose(
        nao_ip,
        angles,
        duration=duration,
        stiffness=0.7,
        reset_to_standinit=reset_to_standinit,
    )

    return angles


def replicate_pose(
    pose: Pose,
    nao_ip: str,
    mirror: bool = True,
    duration: float = DEFAULT_DURATION,
    reset_to_standinit: bool = False,
) -> Dict[str, float]:
    """Make NAO replicate a Pose object."""
    return replicate_pose_from_kp(
        pose.kp_img_norm,
        nao_ip,
        mirror=mirror,
        duration=duration,
        reset_to_standinit=reset_to_standinit,
    )
