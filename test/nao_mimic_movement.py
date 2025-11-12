#!/usr/bin/env python3
"""
SIC + MediaPipe NAO mimic (NAOqi 2.8.7.x)

- Uses Social Interaction Cloud (Redis) for camera (like your demo)
- Uses direct NAOqi (qi) for motion streaming (reliable for real-time angles)
- Tracks your arms with MediaPipe Pose and mirrors them on NAO

Run on your laptop:
    python sic_nao_mediapipe_mimic.py --nao-ip 192.168.1.10 --camera 0

Keys: q quit • s toggle arm stiffness • m toggle mirror

References:
- SIC motion & posture APIs (posture/stiffness examples). See SIC tutorials. 
- NAOqi setAngles for reactive joint control (non-blocking). 

"""
import argparse
import math
import time
import queue

import cv2
import numpy as np
import mediapipe as mp
import qi

# ---- SIC imports ----
from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_camera import NaoqiCameraConf
from sic_framework.core.message_python2 import CompressedImageMessage
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest


# ------------------------- Config -------------------------
SEND_HZ = 15.0
EMA_ALPHA = 0.3
DRAW_LANDMARKS = True
INVERT_PITCH_L = True
INVERT_PITCH_R = True

LIMITS = {
    "LShoulderPitch": (-2.08, 2.08),
    "LShoulderRoll": (-0.31, 1.33),
    "LElbowYaw": (-2.08, 2.08),
    "LElbowRoll": (-1.55, -0.03),
    "RShoulderPitch": (-2.08, 2.08),
    "RShoulderRoll": (-1.33, 0.31),
    "RElbowYaw": (-2.08, 2.08),
    "RElbowRoll": (0.03, 1.55),
}

ARM_JOINTS = [
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll",
]

# ------------------------- Utils -------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

class Ema:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
    def __call__(self, x):
        if self.y is None:
            self.y = x
        else:
            self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y

# Angle helpers

def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.acos(dot)

# ------------------------- Pose mapping -------------------------
PL = mp.solutions.pose.PoseLandmark


def compute_arm_angles(landmarks, image_w, image_h, mirror=True):
    def pt(lm):
        return np.array([lm.x * image_w, lm.y * image_h], dtype=np.float32)

    def maybe_mirror(p):
        if not mirror:
            return p
        p2 = p.copy()
        p2[0] = image_w - p2[0]
        return p2

    def get(name):
        p = pt(landmarks[name])
        return maybe_mirror(p)

    out = {j: 0.0 for j in ARM_JOINTS}

    Ls = get(PL.LEFT_SHOULDER)
    Le = get(PL.LEFT_ELBOW)
    Lw = get(PL.LEFT_WRIST)
    Rs = get(PL.RIGHT_SHOULDER)
    Re = get(PL.RIGHT_ELBOW)
    Rw = get(PL.RIGHT_WRIST)

    shoulder_width = np.linalg.norm(Ls - Rs) + 1e-6

    # Left arm
    Lu = Le - Ls
    Lf = Lw - Le
    l_roll = math.atan2(Lu[0], -Lu[1])
    l_pitch = (Ls[1] - Lw[1]) / (shoulder_width)
    l_pitch = clamp(l_pitch, -1.0, 1.0) * (math.pi / 2)
    if INVERT_PITCH_L:
        l_pitch *= -1.0
    l_eraw = math.pi - angle_between(Lu, Lf)
    l_elbow_roll = -abs(l_eraw)
    l_elbow_yaw = 0.0

    out["LShoulderPitch"] = clamp(l_pitch, *LIMITS["LShoulderPitch"])
    out["LShoulderRoll"] = clamp(l_roll, *LIMITS["LShoulderRoll"])
    out["LElbowYaw"] = clamp(l_elbow_yaw, *LIMITS["LElbowYaw"])
    out["LElbowRoll"] = clamp(l_elbow_roll, *LIMITS["LElbowRoll"])

    # Right arm
    Ru = Re - Rs
    Rf = Rw - Re
    r_roll = math.atan2(Ru[0], -Ru[1])
    r_pitch = (Rs[1] - Rw[1]) / (shoulder_width)
    r_pitch = clamp(r_pitch, -1.0, 1.0) * (math.pi / 2)
    if INVERT_PITCH_R:
        r_pitch *= -1.0
    r_eraw = math.pi - angle_between(Ru, Rf)
    r_elbow_roll = +abs(r_eraw)
    r_elbow_yaw = 0.0

    out["RShoulderPitch"] = clamp(r_pitch, *LIMITS["RShoulderPitch"])
    out["RShoulderRoll"] = clamp(r_roll, *LIMITS["RShoulderRoll"])
    out["RElbowYaw"] = clamp(r_elbow_yaw, *LIMITS["RElbowYaw"])
    out["RElbowRoll"] = clamp(r_elbow_roll, *LIMITS["RElbowRoll"])

    return out

# ------------------------- NAO control (NAOqi) -------------------------
class NaoArmSender:
    def __init__(self, ip, port=9559):
        self.session = qi.Session()
        self.session.connect(f"tcp://{ip}:{port}")
        self.motion = self.session.service("ALMotion")
        self.posture = self.session.service("ALRobotPosture")
        self.stiff = False
        self.filters = {j: Ema(EMA_ALPHA) for j in ARM_JOINTS}

    def set_stiffness(self, on=True):
        try:
            # Stiffen only the arms to be safe
            names = ["LArm", "RArm"]
            self.motion.setStiffnesses(names, 1.0 if on else 0.0)
            self.stiff = on
        except Exception as e:
            print("Stiffness error:", e)

    def stand_init(self):
        try:
            self.posture.goToPosture("StandInit", 0.5)
        except Exception as e:
            print("Posture warning:", e)

    def send_angles(self, angles):
        names, vals = [], []
        for j in ARM_JOINTS:
            a = float(angles[j])
            a = self.filters[j](a)
            names.append(j)
            vals.append(a)
        try:
            self.motion.setAngles(names, vals, 0.2)
        except Exception as e:
            print("setAngles error:", e)

# ------------------------- SIC Application -------------------------
class NaoMediapipeMimic(SICApplication):
    def __init__(self, nao_ip, camera_idx=0, vflip=0, mirror=True):
        super(NaoMediapipeMimic, self).__init__()
        self.set_log_level(sic_logging.INFO)

        # SIC NAO (for camera via Redis)
        self.nao_ip = nao_ip
        self.nao = None
        self.imgs = queue.Queue()

        # Motion via NAOqi directly (low latency)
        self.ctrl = NaoArmSender(nao_ip)
        self.ctrl.set_stiffness(True)
        self.ctrl.stand_init()

    
        # Vision
        self.mirror = mirror
        self.pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawer = mp.solutions.drawing_utils

        # Rate control
        self.next_send = 0.0

        # Start SIC devices
        conf = NaoqiCameraConf(fps=30)
        self.nao = Nao(ip=self.nao_ip, top_camera_conf=conf)
        self.nao.top_camera.register_callback(self.on_image)

    


    # SIC camera callback
    def on_image(self, image_message: CompressedImageMessage):
        if self.imgs.qsize() > 1:  # drop stale frames
            try:
                self.imgs.get_nowait()
            except queue.Empty:
                pass
        self.imgs.put(image_message.image)


    def rest(self):
        """Send robot to rest safely."""
        try:
            self.motion.rest()
            self.stiff = False
            print("Robot is now in rest position.")
        except Exception as e:
            print("Rest error:", e)


    def run(self):
        try:
            while not self.shutdown_event.is_set():
                try:
                    frame = self.imgs.get(timeout=0.1)
                except queue.Empty:
                    continue

                # SIC gives RGB; OpenCV expects BGR for drawing
                vis = frame[..., ::-1].copy()
                if self.mirror:
                    vis = cv2.flip(vis, 1)

                rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                res = self.pose.process(rgb)
                h, w = vis.shape[:2]

                if res.pose_landmarks:
                    angles = compute_arm_angles(res.pose_landmarks.landmark, w, h, mirror=False)
                    t = time.time()
                    if t >= self.next_send and self.ctrl.stiff:
                        self.next_send = t + 1.0 / SEND_HZ
                        self.ctrl.send_angles(angles)

                    if DRAW_LANDMARKS:
                        self.drawer.draw_landmarks(vis, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

                cv2.putText(vis, f"Mirror:{self.mirror}  Stiff:{self.ctrl.stiff}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.imshow("SIC NAO Mimic", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.ctrl.set_stiffness(not self.ctrl.stiff)
                elif key == ord('m'):
                    self.mirror = not self.mirror
        finally:
            cv2.destroyAllWindows()
            try:
                # Request NAO to go to Rest posture via SIC
                if self.nao is not None:
                    self.nao.autonomous.request(NaoRestRequest())
                    self.logger.info("Sent NaoRestRequest – robot going to rest.")
                else:
                    # Fallback: direct NAOqi rest if SIC unavailable
                    self.ctrl.rest()
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")
            self.shutdown()




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nao-ip", required=True)
    p.add_argument("--camera", type=int, default=0, help="Unused; camera is on-robot via SIC")
    p.add_argument("--no-mirror", action="store_true")
    args = p.parse_args()

    app = NaoMediapipeMimic(nao_ip=args.nao_ip, mirror=not args.no_mirror)
    app.run()

if __name__ == "__main__":
    main()
