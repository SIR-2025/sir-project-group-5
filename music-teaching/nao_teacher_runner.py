import time
import os
import cv2
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest

from replicate_json_pose import replicate_pose
from nao_teacher import _record_poses_with_camera, _show_pose_window, Pose


def run_teacher(nao, nao_ip):
    """
    Runs the NAO teacher sequence:
        - Stand
        - Record poses from webcam
        - Replicate poses on NAO
        - Return to rest

    This function is synchronous and safe to call from another SIC app.
    """

    nao.motion.request(NaoPostureRequest("Stand", 0.5))
    time.sleep(1)

    # Eyes blue: ready to record
    nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 0, 1, 0.5))

    pose_dir = os.path.join(os.path.dirname(__file__), "poses")
    os.makedirs(pose_dir, exist_ok=True)

    poses = _record_poses_with_camera(
        logger=None,
        out_dir=pose_dir,
        duration=30.0,
        sample_interval=5.0,
        countdown=3,
        camera_index=0,
    )

    if not poses:
        nao.autonomous.request(NaoRestRequest())
        return

    # Eyes orange: playback phase
    nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0.5, 0, 0.5))

    for pose in poses:
        _show_pose_window(pose)
        nao.leds.request(NaoFadeRGBRequest("FaceLeds", 1, 0, 0, 0))
        replicate_pose(pose, nao_ip, mirror=True, duration=2.0)
        nao.leds.request(NaoFadeRGBRequest("FaceLeds", 0, 1, 0, 0))
        time.sleep(1)

    nao.autonomous.request(NaoRestRequest())
