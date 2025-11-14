# nao_gpt_voice_chat.py

# ---------- Imports ----------
from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoWakeUpRequest, NaoRestRequest
from sic_framework.services.dialogflow.dialogflow import RecognitionResult   # for recognition message format

from openai import OpenAI
from dotenv import load_dotenv
import json, os, numpy as np
from time import sleep

# ---------- Class ----------
class NaoGPTVoiceChat(SICApplication):
    """
    NAO listens via its mic (like in Dialogflow demo),
    sends recognized speech to GPT-4o-mini, and speaks the reply.
    """

    def __init__(self):
        super(NaoGPTVoiceChat, self).__init__()
        self.set_log_level(sic_logging.INFO)

        # Load environment and OpenAI client
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing in environment (.env file)")
        self.client = OpenAI(api_key=api_key)

        # Robot setup
        self.nao_ip = "10.0.0.181"   
        self.nao = None
        self.session_id = np.random.randint(10000)

        # Conversation context
        self.history = [
            {"role": "system",
             "content": "You are NAO, a friendly and helpful humanoid robot who answers people briefly and kindly."}
        ]

        self.setup()

    # ---------- Setup ----------
    def setup(self):
        """Initialize NAO and wake it up."""
        self.logger.info("Initializing NAO...")
        self.nao = Nao(ip=self.nao_ip)
        self.nao.autonomous.request(NaoWakeUpRequest())

        # Get NAO microphone as input source
        self.nao_mic = self.nao.mic

        # You can stream mic audio manually or use a lightweight STT layer
        # For now we use Dialogflow's recognition format for callbacks
        # but we’ll plug in our own STT pipeline soon (using Whisper or similar).

        self.logger.info("Setup complete. NAO ready to listen.")

    # ---------- Recognition Handler ----------
    def on_recognition(self, message):
        """
        This callback receives recognition results from the microphone.
        In your setup, you'll either connect Dialogflow's recognition result
        or another STT service that publishes 'RecognitionResult' messages.
        """
        result: RecognitionResult = message.response.recognition_result
        if result and result.is_final:
            user_text = result.transcript.strip()
            self.logger.info(f"🗣️ User said: {user_text}")
            self.handle_query(user_text)

    # ---------- LLM Request ----------
    def handle_query(self, user_text: str):
        """Send user query to GPT and speak back the reply."""
        self.history.append({"role": "user", "content": user_text})

        try:
            # Query GPT-4o-mini
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.history,
                temperature=0.7,
            )
            answer = response.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": answer})

            # Log + speak
            self.logger.info(f"🤖 NAO: {answer}")
            self.nao.tts.request(NaoqiTextToSpeechRequest(answer, animated=True))

        except Exception as e:
            self.logger.error(f"OpenAI error: {e}")
            self.nao.tts.request(NaoqiTextToSpeechRequest("Sorry, I had an error thinking."))

    # ---------- Main Run Loop ----------
    def run(self):
        """Main loop — speak greeting and process continuously."""
        try:
            self.logger.info("Starting NAO GPT Voice Chat...")
            self.nao.tts.request(NaoqiTextToSpeechRequest("Hello, I am NAO. Talk to me!"))

            # ---- Connect recognition callback ----
            # If you have a Dialogflow-like recognition stream running:
            # self.dialogflow.register_callback(self.on_recognition)
            # For now, simulate via text (testing only):
            while not self.shutdown_event.is_set():
                user_text = input("\nYou (type for now, replace with speech recognition later): ")
                if user_text.lower() in ["exit", "quit"]:
                    break
                self.handle_query(user_text)
                sleep(1)

        except Exception as e:
            self.logger.error(f"Error: {e}")
        finally:
            self.nao.autonomous.request(NaoRestRequest())
            self.shutdown()
            self.logger.info("Application shut down.")

# ---------- Entry Point ----------
if __name__ == "__main__":
    app = NaoGPTVoiceChat()
    app.run()
