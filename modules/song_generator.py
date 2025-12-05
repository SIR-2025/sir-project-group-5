# modules/song_generation.py

import os
import time
import wave
import threading

import requests
from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI

from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_motion import (
    NaoqiAnimationRequest,
    NaoPostureRequest,
)
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
from sic_framework.core.message_python2 import AudioRequest
from sic_framework.services.dialogflow_cx.dialogflow_cx import DetectIntentRequest



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUNO_API_KEY = os.getenv("SUNO_API_KEY")


class SunoAPI:
    """Simple wrapper around the Suno API."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.sunoapi.org/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def generate_music(self, **options) -> str:
        resp = requests.post(
            f"{self.base_url}/generate",
            headers=self.headers,
            json=options,
        )
        result = resp.json()
        if result.get("code") != 200:
            raise Exception(f"Generation failed: {result.get('msg')}")
        return result["data"]["taskId"]

    def wait_for_completion(self, task_id: str, max_wait_time: float = 600.0):
        start = time.time()
        while time.time() - start < max_wait_time:
            status = self.get_task_status(task_id)
            if status["status"] == "SUCCESS":
                return status["response"]
            if status["status"] == "FAILED":
                raise Exception(f"Generation failed: {status.get('errorMessage')}")
            time.sleep(30)
        raise Exception("Generation timeout")

    def get_task_status(self, task_id: str) -> dict:
        resp = requests.get(
            f"{self.base_url}/generate/record-info?taskId={task_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return resp.json()["data"]



def instrumental_gen(style: str, max_wait_time: float = 600.0) -> dict:
    """Generate an instrumental track in the given style using Suno."""
    api = SunoAPI(api_key=SUNO_API_KEY)
    task_id = api.generate_music(
        prompt=f"Create a {style} song",
        customMode=True,
        style=style,
        title=f"{style}-instrumental",
        instrumental=True,
        model="V5",
        callBackUrl="https://your-server.com/music-callback",
    )
    result = api.wait_for_completion(task_id, max_wait_time=max_wait_time)
    return result["sunoData"][0]   # first track


def download_song(audio_meta: dict) -> str:
    """Download Suno MP3 and convert to WAV. Returns path to WAV file."""
    audio_url = audio_meta["audioUrl"]
    title = audio_meta["title"]

    mp3_path = f"{title}.mp3"
    wav_path = f"{title}.wav"

    r = requests.get(audio_url)
    with open(mp3_path, "wb") as f:
        f.write(r.content)

    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")
    os.remove(mp3_path)

    return wav_path


def play_audio(nao, wav_path: str, logger=None):
    """Play a WAV file through NAO's speaker."""
    log = logger.info if logger else print
    log(f"Playing WAV on NAO: {wav_path}")

    wf = wave.open(wav_path, "rb")
    samplerate = wf.getframerate()
    wf.rewind()
    data = wf.readframes(wf.getnframes())
    msg = AudioRequest(sample_rate=samplerate, waveform=bytes(data))
    nao.speaker.request(msg)




def style_extractor(user_text: str, api_key: str = OPENAI_API_KEY) -> str:
    """Extract music style from free-form user input using OpenAI."""
    client = OpenAI(api_key=api_key)
    prompt = f"""You are a style extraction expert. From the user input, extract the
music style the user wants the song to be in. Always append '20 seconds' at the end.

Examples:
User: Create a salsa song
Output: salsa 20 seconds

User: Make a song in the style of hiphop
Output: hip hop 20 seconds

User: {user_text}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You output only the style and duration, nothing else.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()



def stretching_routine(nao, logger=None):
    log = logger.info if logger else print
    try:
        nao.motion.request(NaoPostureRequest("StandInit", 0.5), block=True)
        nao.tts.request(
            NaoqiTextToSpeechRequest(
                "Let's do some warm up movements first to get the party started!"
            )
        )
        nao.motion.request(
            NaoqiAnimationRequest("animations/Stand/Gestures/Hey_6"),
            block=True,
        )
        time.sleep(0.5)
        nao.tts.request(NaoqiTextToSpeechRequest("You! Get Moving!"))
        nao.motion.request(
            NaoqiAnimationRequest("animations/Stand/Gestures/You_1"),
            block=True,
        )
        time.sleep(0.5)
        nao.tts.request(NaoqiTextToSpeechRequest("Do some squats!"))
        nao.motion.request(
            NaoqiAnimationRequest("animations/Stand/Gestures/You_1"),
            block=True,
        )

        #squats
        for _ in range(3):
            nao.motion.request(NaoPostureRequest("Stand", 0.5))
            nao.autonomous.request(NaoRestRequest())
            time.sleep(0.5)

        nao.motion.request(NaoPostureRequest("Stand", 0.5))
        nao.tts.request(NaoqiTextToSpeechRequest("And bow to finish!"))
        nao.motion.request(
            NaoqiAnimationRequest("animations/Stand/Gestures/BowShort_1"),
            block=True,
        )

        nao.motion.request(NaoPostureRequest("StandInit", 0.5), block=True)
        nao.tts.request(
            NaoqiTextToSpeechRequest(
                "We are finished! Puh that was hard! Now lets rest until we have enough energy for the song!"
            )
        )

    except Exception as e:
        log(f"Error in stretching_routine: {e}")
        try:
            nao.autonomous.request(NaoRestRequest())
        except Exception:
            pass





def song_generation_with_exercise(
    nao,
    dialogflow_cx,
    session_id: int,
    logger=None,
    max_wait_time: float = 600.0,
):
    """
    Use an existing NAO + Dialogflow session to:
      - Ask the user for a music style.
      - Let NAO do a stretching routine.
      - Generate music with Suno.
      - Play the generated song on NAO.
    """
    log = logger.info if logger else print

    try:
        nao.tts.request(
            NaoqiTextToSpeechRequest(
                "You are right, it would be more fun with a song! "
                "I am Nao-DJ! Let me know which style you want!"
            )
        )
        log("Song generation: asking user for style via Dialogflow.")

        reply = dialogflow_cx.request(DetectIntentRequest(session_id))

        if getattr(reply, "transcript", None):
            user_text = reply.transcript
            log(f"User said (style): {user_text}")
            style = style_extractor(user_text)
            if not style:
                style = "hip hop 20 seconds"
        else:
            log("No transcript from Dialogflow; defaulting to hip hop 20 seconds.")
            style = "hip hop 20 seconds"

        log(f"Using style string for Suno: {style}")


        stretch_thread = threading.Thread(
            target=stretching_routine,
            args=(nao, logger),
            daemon=True,
        )
        stretch_thread.start()

        song_meta = instrumental_gen(style, max_wait_time=max_wait_time)
        log(f"Suno song meta received: {song_meta.get('title', 'unknown title')}")

        wav_path = download_song(song_meta)
        log(f"Downloaded and converted song to WAV: {wav_path}")

        audio = AudioSegment.from_wav(wav_path)
        cut = audio[0:25_000]
        cut.export(wav_path, format="wav")

        play_audio(nao, wav_path, logger=logger)

        nao.tts.request(
            NaoqiTextToSpeechRequest("I hope you liked My song! Lets do some dancing now! Do you know any dance that would fit my song?")
        )

    except Exception as e:
        log(f"Exception in song_generation_with_exercise: {e}")
        try:
            nao.autonomous.request(NaoRestRequest())
        except Exception:
            pass
