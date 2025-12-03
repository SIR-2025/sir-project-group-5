from openai import OpenAI
import os
import requests
import time
import requests
from dotenv import load_dotenv
from pydub import AudioSegment
load_dotenv()


OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
SUNO_API_KEY = os.getenv("SUNO_API_KEY")


# style extractor prompt
def style_extractor(user_input, OPEN_AI_API_KEY=OPENAI_API_KEY):
    client = OpenAI(api_key=OPEN_AI_API_KEY)

    system_message = (
        "You are an expert at identifying musical styles from user requests. "
        "Your task is to extract **only** the musical style mentioned in the user's text. "
        "Output only the style name, in lowercase, with no extra words. "
        "If no style is mentioned, output 'classical'."
    )

    examples = [
        ("I want to dance to a song in style of jazz.", "jazz"),
        ("Please make a classical piece that feels like nature.", "classical"),
        ("Write a pop song about summer.", "pop"),
        ("Can you make a song about robots?", "classical")
    ]

    messages = [{"role": "system", "content": system_message}]

    # add examples in desired format
    for user, assistant in examples:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

    # Now add the real user request
    messages.append({"role": "user", "content": user_input})

    # Send the request
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )

    return resp.choices[0].message.content.strip()


class SunoAPI:
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
            
            time.sleep(30)  # Wait 30 seconds
        
        raise Exception('Generation timeout')
    
    def get_task_status(self, task_id):
        response = requests.get(f'{self.base_url}/generate/record-info?taskId={task_id}',
                              headers={'Authorization': f'Bearer {self.api_key}'})
        return response.json()['data']
    
    

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
        
        # Wait for completion
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
        
        # Wait for completion
        music_result = api.wait_for_completion(music_task_id)
        print(f'Music generated successfully!:{music_result}')
        final_track = music_result['sunoData'][0]
        return final_track
        
    except Exception as error:
        print(f'Error: {error}')
    
    
def main():
    print("Starting")
    user_request = input("Describe the type of song you want: ")
    style = style_extractor(user_request)
    print(f"Extracted style: {style}")
    audio = instrumental_gen(f"{style} - catchy dance, The maximal duration should be 20 seconds")
    print("Instrumental generated")
    wav_path = download_song(audio)
    print(f"Saved to: {wav_path}")



if __name__ == "__main__":
    main()