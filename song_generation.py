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
        model="gpt-4o-mini",  # any chat-capable model
        messages=[
            {"role": "system", "content": "You write catchy, singable lines with exact syllable counts."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
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


def main():
    print("Starting")
    topic  = input("What is the topic?")
    lyrics = lyric_writer(topic)
    print(f"The generated lyrics are {lyrics}")
    print("lyrics are generated")
    audio = suno_gen(lyrics,topic)
    print("Song is generated")
    audio_url = audio["audioUrl"]
    title = audio["title"]
    title = audio["title"]
    r = requests.get(audio_url)
    mp3_path = f"{title}.mp3"
    open(mp3_path, "wb").write(r.content)
    print("Song is downloaded")
    wav_path = f"{title}.wav"
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")
    print("converted to wav")
    os.remove(mp3_path)


if __name__ == "__main__":
    main()