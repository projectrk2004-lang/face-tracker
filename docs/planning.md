# Planning Document - Intelligent Face Tracker
## 1. Problem statement
Count unique visitors from a video/RTSP stream by detecting, tracking and
recognizing faces. Log every entry and exit with an image and timestamp.
## 2. Features
1. YOLOv8 face detection (frame_skip configurable in config.json)
2. ByteTrack tracking (temporary track ids)
3. InsightFace (buffalo_l) 512-d embeddings
4. Auto-registration of new faces with a unique face_id
5. Re-identification using cosine similarity (no double counting)
6. Entry / exit event logging (events.log + images + SQLite)
7. Unique visitor count from the database
8. RTSP support with automatic reconnect
9. Crash-safe logging (commit after every write, flush on shutdown)
## 3. Tech choices
| Module | Choice | Why |
|---|---|---|
| Detection | YOLOv8n-face | fast, runs on CPU |
| Tracking | ByteTrack (ultralytics) | simple and robust |
| Recognition | InsightFace buffalo_l | state-of-the-art accuracy |
| Database | SQLite | no server, one file |
## 4. Compute estimate
(paste the table from Part E)
## 5. Risks and mitigation
- Side/blurred faces -> min_face_size filter, retry embedding
- Similar-looking people -> tune similarity_threshold
- Track id switches -> track relink logic