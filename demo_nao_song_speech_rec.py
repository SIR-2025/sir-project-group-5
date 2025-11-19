# Import basic preliminaries
from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiAnimationRequest, NaoPostureRequest
from sic_framework.core.message_python2 import AudioRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoRestRequest
from sic_framework.services.dialogflow_cx.dialogflow_cx import (
    DialogflowCX,
    DialogflowCXConf,
    DetectIntentRequest,
    QueryResult,
    RecognitionResult,
)
from openai import OpenAI
import wave
import json
from os.path import abspath, join
import numpy as np
import openai
import os
import threading
import time
from openai import OpenAI
import os
import requests
import time
import requests
from dotenv import load_dotenv
from pydub import AudioSegment
load_dotenv()

#globals
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
SUNO_API_KEY = os.getenv("SUNO_API_KEY")


#helper functions
def lyric_writer(topic,OPEN_AI_API_KEY = OPENAI_API_KEY):
    client = OpenAI(api_key=OPEN_AI_API_KEY)
    prompt = (f"""You are a macarena expert that writes  lyrics matching exactly
the rhythmic structure of the song Macarena by Los del Rio. The lyrics should contain four phrases. The content of the lyrics should be exactly about {topic}.

Task:
- Produce lyrics that can be sung to the Macarena melody and rythm.
- The pattern is four lines: 8 / 8 / 8 / 4/ 8 /8 /8 syllables. Eight syllables are equivalent to four beats.
- Each syllable corresponds to 1/8th beat.
- The lyrical content must be about the specified topic.
-The final line has to be Eeeh {topic}
-The overall lyrics should be educational.

Format:
-Output only the four lines, one per line.
- Do one syllable per 1/8th note so it aligns exactly with the Macarena rhythm.

Example: 
Topic: Artificial Intelligence 

Ai uses data to learn from examples
models find patterns in large datasets
they adjust weights to minimize errors
Eeeh artificial intelligence
"""
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "You write catchy, singable lines with exact syllable counts."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )
    return resp.choices[0].message.content.strip()

def suno_gen(lyrics,topic):
    api = SunoAPI(api_key=SUNO_API_KEY)
    
    try:
        print('Generating music...')
        music_task_id = api.generate_music(
            prompt= lyrics,
            customMode=True,
            style='Latin Dance. 25 seconds. 103 BPM',
            title=f"{topic}-macarena",
            instrumental=False,
            model='V4_5',
            callBackUrl='https://your-server.com/music-callback'
            )
        music_result = api.wait_for_completion(music_task_id)
        print(f'Music generated successfully!:{music_result}')
        final_track = music_result['sunoData'][0]
        return final_track
        
    except Exception as error:
        print(f'Error: {error}')

def download_song(audio):
    """Downloads Mp3 and turns it to wav"""
    
    audio_url = audio["audioUrl"]
    title = audio["title"]
    mp3_path = f"{title}.mp3"
    resp = requests.get(audio_url)
    open(mp3_path, "wb").write(resp.content)
    print("Song is downloaded")
    wav_path = f"{title}.wav"
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")
    print("converted to wav")
    os.remove(mp3_path)
    return wav_path

    
def instrumental_gen(style):
    api = SunoAPI(api_key=SUNO_API_KEY)
    
    try:
        print('Generating music...')
        music_task_id = api.generate_music(
            prompt= f"Create a {style} song",
            customMode=True,
            style=f'{style}',
            title=f"{style}-instrumental",
            instrumental=True,
            model='V4_5',
            callBackUrl='https://your-server.com/music-callback'
            )
        music_result = api.wait_for_completion(music_task_id)
        print(f'Music generated successfully!:{music_result}')
        final_track = music_result['sunoData'][0]
        return final_track
        
    except Exception as error:
        print(f'Error: {error}')


class SunoAPI:
    """Instantiatiates the suno api"""
    def __init__(self,api_key):
        self.api_key = api_key
        self.base_url = 'https://api.sunoapi.org/api/v1'
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def generate_music(self, **options):
        response = requests.post(f'{self.base_url}/generate', 
                               headers=self.headers, json=options)
        result = response.json()
        
        if result['code'] != 200:
            raise Exception(f"Generation failed: {result['msg']}")
        
        return result['data']['taskId']
    
    
    def wait_for_completion(self, task_id, max_wait_time=600):
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = self.get_task_status(task_id)
            
            if status['status'] == 'SUCCESS':
                return status['response']
            elif status['status'] == 'FAILED':
                raise Exception(f"Generation failed: {status.get('errorMessage')}")
            
            time.sleep(30)  
        
        raise Exception('Generation timeout')
    
    def get_task_status(self, task_id):
        response = requests.get(f'{self.base_url}/generate/record-info?taskId={task_id}',
                              headers={'Authorization': f'Bearer {self.api_key}'})
        return response.json()['data']
    

class NaoSongGeneratorDemo(SICApplication):
    """for the song generation"""
    def __init__(self):
        # Call parent constructor (handles singleton initialization)
        super(NaoSongGeneratorDemo, self).__init__()
        
        # Demo-specific initialization
        self.nao_ip = "10.0.0.245"  # TODO: Replace with your NAO's IP address
        self.dialogflow_keyfile_path = abspath(join( "conf", "google", "google-key.json"))
        self.nao = None
        self.dialogflow_cx = None
        self.session_id = np.random.randint(10000)

        self.set_log_level(sic_logging.INFO)
        
        # Log files will only be written if set_log_file is called. Must be a valid full path to a directory.
        # self.set_log_file("/Users/apple/Desktop/SAIL/SIC_Development/sic_applications/demos/nao/logs")
        
        self.setup()
    
    def on_recognition(self, message):
        """
        Callback function for Dialogflow CX recognition results.
        
        Args:
            message: The Dialogflow CX recognition result message.
        
        Returns:
            None
        """
        if message.response:
            if hasattr(message.response, 'recognition_result') and message.response.recognition_result:
                rr = message.response.recognition_result
                if hasattr(rr, 'is_final') and rr.is_final:
                    if hasattr(rr, 'transcript'):
                        self.logger.info("Transcript: {transcript}".format(transcript=rr.transcript))

    
    def style_extractor(self, input,OPEN_AI_API_KEY = OPENAI_API_KEY):
        client = OpenAI(api_key=OPEN_AI_API_KEY)
        prompt = (f"""You are a style extraction expert given the input extract the style the user wants to create the song about. Always add 20 seconds.
                  Examples: 
                  User:Create a salsa song
                  output: salsa 20 seconds 
                  User: Make a song in the style of hiphop
                  output:hip hop 20 seconds"
                  User: {input}""")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # any chat-capable model
            messages=[
                {"role": "system", "content": "You write catchy, singable lines with exact syllable counts."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    
    def setup(self):
        """Initialize and configure NAO robot and Dialogflow CX."""
        self.logger.info("Initializing NAO robot...")
        
        # Initialize NAO
        self.nao = Nao(ip=self.nao_ip, dev_test=False)
        nao_mic = self.nao.mic
        
        self.logger.info("Initializing Dialogflow CX...")
        
        # Load the key json file
        with open(self.dialogflow_keyfile_path) as f:
            keyfile_json = json.load(f)
        
        # Agent configuration
        # TODO: Replace with your agent details (use verify_dialogflow_cx_agent.py to find them)
        agent_id = "9c558c27-0369-4a6c-bbb2-55b45fe73067"  # Replace with your agent ID
        location = "europe-west4"  # Replace with your agent location if different
        
        # Create configuration for Dialogflow CX
        # Note: NAO uses 16000 Hz sample rate (not 44100 like desktop)
        dialogflow_conf = DialogflowCXConf(
            keyfile_json=keyfile_json,
            agent_id=agent_id,
            location=location,
            sample_rate_hertz=16000,  # NAO's microphone sample rate
            language="en"
        )
        
        # Initialize Dialogflow CX with NAO's microphone as input
        self.dialogflow_cx = DialogflowCX(conf=dialogflow_conf, input_source=nao_mic)
        
        self.logger.info("Initialized Dialogflow CX... registering callback function")
        # Register a callback function to handle recognition results
        self.dialogflow_cx.register_callback(callback=self.on_recognition)


    def stretching_routine(self):
        try:
            self.nao.motion.request(NaoPostureRequest("StandInit", 0.5), block=True)
            self.nao.tts.request(NaoqiTextToSpeechRequest("Let's stretch guys!"))
            self.nao.motion.request(
                NaoqiAnimationRequest("animations/Stand/Gestures/Hey_1"), 
                block=True
            )
            time.sleep(0.5)
            self.nao.tts.request(NaoqiTextToSpeechRequest("Clap your hands!"))
            self.nao.motion.request(
                NaoqiAnimationRequest("animations/Stand/Gestures/Clap_1"),
                block=True
            )
            time.sleep(0.5)
            self.nao.tts.request(NaoqiTextToSpeechRequest("Act as if you were explaining!"))
            self.nao.motion.request(
                NaoqiAnimationRequest("animations/Stand/Gestures/Explain_1"),
                block=True
            )
            time.sleep(0.5)

            self.nao.tts.request(NaoqiTextToSpeechRequest("Let your body talk!"))
            self.nao.motion.request(
                NaoqiAnimationRequest("animations/Stand/BodyTalk/ScratchHead_1"),
                block=True
            )
            time.sleep(0.5)
            self.nao.tts.request(NaoqiTextToSpeechRequest("And bow to finish!"))
            self.nao.motion.request(
                NaoqiAnimationRequest("animations/Stand/Gestures/BowShort_1"),
                block=True
            )

            self.nao.motion.request(NaoPostureRequest("StandInit", 0.5), block=True)
            self.nao.tts.request(NaoqiTextToSpeechRequest("We are finished!Puh that was hard! Now lets rest until we have enough energy for the song!"))

        except Exception as e:
            self.logger.error(f"Error in stretching_routine: {e}")
            self.nao.autonomous.request(NaoRestRequest())



    def play_audio(self):
        try:
            self.logger.info("Passing audio to nao")
            self.wavefile.rewind()
            data = self.wavefile.readframes(self.wavefile.getnframes())
            msg = AudioRequest(sample_rate=self.samplerate, waveform=bytes(data))
            self.nao.speaker.request(msg)
        except Exception as e:
            self.logger.error(f"ERROR!")
    
    
    
    def run(self):
        """Main loop"""
        try:
            self.nao.tts.request(NaoqiTextToSpeechRequest("Hello, I am Nao, lets make a song!!! Let me know which style you want!"))
            self.logger.info(" -- Ready -- ")
            
            while not self.shutdown_event.is_set():
                self.logger.info(" ----- Your turn to talk!")
                reply = self.dialogflow_cx.request(DetectIntentRequest(self.session_id))
                if reply.transcript:
                    self.logger.info("User said: {text}".format(text=reply.transcript))
                    self.style = self.style_extractor(reply.transcript)
                    if self.style ==None:
                        self.style = "hip-hop"
                    
                elif not reply.transcript:
                    self.logger.info("User said nothing")
                    self.style = "hip-hop"
                
                stretch_thread = threading.Thread(target = self.stretching_routine,daemon=True)
                stretch_thread.start()
                self.song = instrumental_gen(self.style)
                self.downloaded = download_song(self.song)
                self.wavefile = wave.open(self.downloaded, "rb")
                self.samplerate = self.wavefile.getframerate()
                self.logger.info("Passing audio to nao")
                self.wavefile.rewind()
                data = self.wavefile.readframes(self.wavefile.getnframes())
                msg = AudioRequest(sample_rate=self.samplerate, waveform=bytes(data))
                self.nao.speaker.request(msg)
                    
        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
            self.nao.autonomous.request(NaoRestRequest())
        except Exception as e:
            self.logger.error("Exception: {}".format(e))
            self.nao.autonomous.request(NaoRestRequest())
        finally:
            self.nao.autonomous.request(NaoRestRequest())
            self.shutdown()


if __name__ == "__main__":
    demo = NaoSongGeneratorDemo()
    demo.run()