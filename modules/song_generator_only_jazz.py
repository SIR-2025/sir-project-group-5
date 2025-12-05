"""
Jazz song generator for NAO using SunoAPI.

This module:
  - Calls the SunoAPI to generate a short jazz track.
  - While waiting for generation, makes NAO do "squats"
  - Downloads the resulting MP3, converts it to WAV, and
    plays it on NAO's speaker.

Environment:
  - Requires SUNO_API_KEY in the environment.
"""

from __future__ import annotations

import os
import time
import wave
from typing import Optional

import requests
from dotenv import load_dotenv
from pydub import AudioSegment

from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
from sic_framework.core.message_python2 import AudioRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest

load_dotenv()
SUNO_API_KEY = os.getenv("SUNO_API_KEY")


class SunoAPI:
    """Minimal SunoAPI client for music generation."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("SUNO_API_KEY is not set in the environment.")
        self.api_key = api_key
        self.base_url = "https://api.sunoapi.org/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def generate_music(self, **options) -> str:
        """Request a new music generation task and return its taskId."""
        resp = requests.post(
            f"{self.base_url}/generate", headers=self.headers, json=options
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 200:
            raise RuntimeError(f"Generation failed: {result.get('msg')}")

        return result["data"]["taskId"]

    def get_task_status(self, task_id: str) -> dict:
        """Get the status object for a given taskId."""
        resp = requests.get(
            f"{self.base_url}/generate/record-info?taskId={task_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        return resp.json()["data"]


def _download_song(audio_meta: dict, logger=None) -> str:
    """
    Download the generated MP3 from Suno and convert it to WAV.

    Args:
        audio_meta: Single track metadata dict from Suno's response.
        logger: Optional logger-like object.

    Returns:
        Path to the resulting WAV file on disk.
    """
    audio_url = audio_meta["audioUrl"]
    title = audio_meta.get("title", "nao_jazz_song")
    safe_title = "".join(c for c in title if c.isalnum() or c in ("_", "-", " "))
    mp3_path = f"{safe_title}.mp3"
    wav_path = f"{safe_title}.wav"

    if logger:
        logger.info(f"Downloading song from {audio_url} → {mp3_path}")
    resp = requests.get(audio_url)
    resp.raise_for_status()

    with open(mp3_path, "wb") as f:
        f.write(resp.content)

    if logger:
        logger.info("Converting MP3 to WAV...")
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")

    os.remove(mp3_path)
    if logger:
        logger.info(f"Song ready at {wav_path}")
    return wav_path


def _squat_once(nao, logger=None) -> None:
    """
    Make NAO perform one simple "squat":
      Stand → Rest → Stand.

    Uses blocking requests, so this takes a few seconds.
    """
    try:
        if logger:
            logger.info("Performing squat: Stand.")
        nao.motion.request(NaoPostureRequest("Stand", 0.5), block=True)

        # short pause at top
        time.sleep(0.5)

        if logger:
            logger.info("Performing squat: Rest.")
        nao.autonomous.request(NaoRestRequest())
        time.sleep(1.0)

        if logger:
            logger.info("Back to Stand.")
        nao.motion.request(NaoPostureRequest("Stand", 0.5), block=True)
        time.sleep(0.5)

    except Exception as e:
        if logger:
            logger.error(f"Error during squat: {e}")


def generate_jazz_song_with_squats(
    nao,
    logger=None,
    max_wait_time: float = 25.0,
) -> Optional[str]:
    """
    Generate a jazz song using SunoAPI while NAO does squats, then play it.

    Flow:
      1. Ask Suno for a short jazz instrumental.
      2. While the track is being generated:
         - Poll task status.
         - On each poll, let NAO perform one squat.
      3. When finished:
         - Download and convert to WAV.
         - Play the song via NAO's speaker.

    Args:
        nao: NAO device handle (must have .motion, .autonomous, .speaker).
        logger: Optional logger-like object.
        max_wait_time: Max seconds to wait for Suno before giving up.

    Returns:
        Path to the WAV file if successful, or None on failure.
    """
    if logger:
        logger.info("Starting jazz song generation with SunoAPI...")

    try:
        api = SunoAPI(api_key=SUNO_API_KEY)
    except ValueError as e:
        if logger:
            logger.error(str(e))
        return None

    try:
        if logger:
            logger.info("Requesting jazz song from Suno...")

        task_id = api.generate_music(
            prompt="Create a short, upbeat jazz instrumental.",
            customMode=True,
            style="jazz 25 seconds",
            title="nao_jazz_song",
            instrumental=True,
            model="V4_5",
            callBackUrl='https://your-server.com/music-callback'
        )
    except Exception as e:
        if logger:
            logger.error(f"Error submitting generation request: {e}")
        return None

    start_time = time.time()
    track_meta = None

    if logger:
        logger.info(f"Suno task started with id={task_id}. Waiting for completion...")

    while time.time() - start_time < max_wait_time:
        try:
            status = api.get_task_status(task_id)
        except Exception as e:
            if logger:
                logger.error(f"Error polling Suno task: {e}")
            time.sleep(5.0)
            continue

        state = status.get("status")
        if logger:
            logger.info(f"Suno task status: {state}")

        if state == "SUCCESS":
            # Expecting something like status["response"]["sunoData"][0]
            try:
                response = status["response"]
                track_meta = response["sunoData"][0]
            except Exception as e:
                if logger:
                    logger.error(f"Malformed Suno response: {e}")
            break
        elif state == "FAILED":
            if logger:
                logger.error(f"Suno task failed: {status}")
            return None

        _squat_once(nao, logger=logger)
        time.sleep(2.0)

    if track_meta is None:
        if logger:
            logger.error("Generation timeout or missing track metadata.")
        return None

    wav_path = _download_song(track_meta, logger=logger)

    try:
        if logger:
            logger.info("Playing generated jazz song on NAO...")

        nao.tts.request(
            NaoqiTextToSpeechRequest("Here is the song that I made")
        )

        with wave.open(wav_path, "rb") as wf:
            samplerate = wf.getframerate()
            data = wf.readframes(wf.getnframes())
            msg = AudioRequest(sample_rate=samplerate, waveform=bytes(data))
            nao.speaker.request(msg)
    except Exception as e:
        if logger:
            logger.error(f"Error while playing audio on NAO: {e}")
        return None

    return wav_path