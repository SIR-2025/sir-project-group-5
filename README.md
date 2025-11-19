# SIR Project - Group 5  
### NAO Interactive Dance Teacher

This project implements a fully interactive NAO robot behavior where NAO:

1. **Greets a user and generates a custom song** (via Dialogflow CX + TTS).  
2. **Watches the user dance** using its camera.  
3. **Records multiple static dance poses** using MediaPipe Pose.  
4. **Mirrors and maps human poses to NAO joints**, replaying the dance back.  
5. **Teaches the learned dance** to the next person that interacts with it.

The full pipeline is multimodal (vision + speech) and runs live on the NAO robot using SIC Framework components.

---

## Entry Point

# For all Python dependencies, install requirements.txt: 
pip install -r requirements.txt


Run:
```bash
python main.py
```

`main.py` launches the interactive application (`NaoTeachMode`) which handles:

- NAO initialization (camera, mic, TTS, motion, LEDs)  
- Dialogflow CX communication  
- Pose learning & playback  
- Live OpenCV GUI with skeleton overlays  
- Intent-based behavior switching (e.g., “start teaching”)  

---

## Modules

### `modules/pose_teacher.py`
Headless MediaPipe-based pose recording system:
- Captures poses every few seconds  
- Outputs JSON & in-memory pose objects  
- Supports GUI callbacks for live feedback  

### `modules/replicate_json_pose.py`
Converts normalized MediaPipe keypoints to NAO joint angles  
and executes the corresponding NAO motion:
- Shoulder & elbow inference  
- Joint limit clamping  
- Mirroring for camera alignment  

### `runners/teacher_runner.py`
Simple wrapper used by the main application to drive recording and playback.

### `main.py`
Full application logic:
- Dialogflow loop  
- Intent routing  
- Camera UI and overlays  
- Teaching workflow (record → playback → teach)  

---

## Dialogflow CX

Use voice commands such as:

- **“Start teaching”**  
- **“Stop teaching”**  
- (Extendable to custom commands for music, conversation, etc.)

NAO replies using TTS and executes appropriate behaviors.

---

## Directory Structure

```
main.py
modules/
pose_teacher.py
replicate_json_pose.py
runners/
teacher_runner.py
conf/
google/google-key.json
poses/
```
---