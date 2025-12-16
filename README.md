# SIR Project – Group 5  
## NAO Interactive Dance Teacher

This project implements a fully interactive multimodal behavior for the NAO robot, where NAO can generate music, observe human movement, learn dance poses, and teach the learned choreography to new users.

The system integrates speech, vision, motion, and audio generation into a single real-time pipeline using the SIC Framework, Dialogflow CX, MediaPipe Pose, and external music generation services.

---

## Project Overview

During an interaction, NAO is able to:

1. **Greet the user and hold a spoken conversation** using Dialogflow CX.
2. **Generate a custom song** based on the user’s requested style.
3. **Play the generated music** through NAO’s speakers.
4. **Observe the user dancing** via NAO’s camera.
5. **Record multiple static dance poses** using MediaPipe Pose.
6. **Replay the recorded dance** by mapping human poses to NAO’s joints.
7. **Teach the learned dance** to a new user, pose by pose, validating imitation in real time.

The system is designed to be **interactive, robust, and modular**, with all UI and visualization handled exclusively in `main.py`.

---

## Entry Point

### Installation

Install all required Python dependencies:

```bash
pip install -r requirements.txt
```

Make sure you also provide:

- A valid Dialogflow CX service account key
- Required API keys (OpenAI / Suno) via .env

## Run the Application

```bash
python main.py
```

This launches the main interactive application (NaoTeachMode).

---

### Main Application (main.py)

main.py is the central orchestrator of the entire system. It is responsible for:

- Initializing NAO (camera, microphone, motion, LEDs, TTS)
- Running the Dialogflow CX intent loop
- Managing interaction states (idle, teaching, learning, song generation)
- Rendering the OpenCV GUI:
- Live camera feed
- Live skeleton overlay
- “Ghost pose” overlays during learning
- Pose thumbnails
- Routing callbacks between headless modules and the UI
- Starting and stopping background threads safely

All drawing and visualization logic lives only in main.py.

---

### Core Modules

#### modules/pose_teacher.py

Headless pose learning logic based on MediaPipe Pose.

Responsibilities:

- Run MediaPipe Pose on camera frames
- Extract and normalize keypoints
- Compute pose similarity metrics

This module:

- Opens no windows
- Performs no drawing
- Contains no UI logic

---

#### modules/pose_learner.py

Headless pose teaching logic based on MediaPipe Pose.

Responsibilities:

- Decide when a pose is successfully imitated
- Report pose state back to main.py via callbacks

---

#### modules/replicate_json_pose.py

Pose-to-robot motion mapping module.

Responsibilities:

- Convert normalized MediaPipe keypoints into NAO joint angles
- Infer shoulder and elbow configurations
- Clamp values to NAO joint limits
- Apply mirroring for camera alignment
- Send motion commands via SIC (no direct qi dependency)

---

#### modules/song_generator.py

Song generation pipeline.

Responsibilities:

- Extract musical style from free-form user speech
- Generate instrumental or lyrical music tracks
- Download, convert, and trim audio (≤ 40 seconds)
- Prepare audio for NAO playback

---

### Runners

Every module has its own runner in order to be executed inside of a script or as a standalone application

---

### Directory Structure
```
main.py
modules/
│   pose_teacher.py
│   pose_learner.py
│   replicate_json_pose.py
│   song_generator.py
runners/
│   teacher_runner.py
│   learner_runner.py
│   song_runner.py
conf/
│   google/
│       google-key.json
poses/
music/
requirements.txt
README.md
```

### Result

The final system allows NAO to act as an interactive dance teacher that:
- Listens
- Watches
- Learns
- Teaches
- And adapts to each new user

All in a single, cohesive multimodal pipeline.