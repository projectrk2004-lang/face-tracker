# face-tracker# Intelligent Face Tracker with Auto-Registration and Visitor Counting
Video: <PASTE YOUR LOOM / YOUTUBE LINK HERE>
## Features
- YOLOv8 face detection + ByteTrack tracking
- InsightFace embeddings, automatic registration, re-identification
- Entry/exit logging to events.log, image folders and SQLite
- Unique visitor count from the database
- RTSP support with auto reconnect
## Setup
1. Install Python 3.11
2. `python -m venv venv` and activate it
3. `pip install -r requirements.txt`
4. Put the sample video at `data/sample.mp4`
5. Put `yolov8n-face.pt` in `models/` (download link: <https://drive.google.com/file/d/1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb/view>)
6. Run: `python main.py` (press q to quit)
## Sample config.json
({
"video_source": "data/sample.mp4",
"frame_skip": 3,
"detection_confidence": 0.5,
"similarity_threshold": 0.45,
"exit_timeout_frames": 30,
"min_face_size": 40,
"tracking_log_every": 10,
"use_gpu": false,
"show_window": true,
"reconnect_attempts": 5,
"reconnect_delay_sec": 2,
"db_path": "db/faces.db",
"log_file": "logs/events.log",
"yolo_model": "models/yolov8n-face.pt"
}
)
## Assumptions
- One camera, mostly frontal faces
- similarity_threshold=0.45 chosen by testing on the sample video
- A face is "exited" after exit_timeout_frames frames without detection
- Faces remain registered in the DB across runs
## AI Planning document
See docs/planning.md
## Architecture diagram
![Architecture](docs/architecture.png)
## Compute estimate
(paste the table from docs/planning.md)
## Sample output
- logs/events.log
- logs/entries/ and logs/exits/
- sample_output/faces.csv and sample_output/events.csv
- Final unique visitor count on the sample video: <https://www.loom.com/share/38d893ce23674ebfa650bc951b261f83>
This project is a part of a hackathon run by https://katomaran.com