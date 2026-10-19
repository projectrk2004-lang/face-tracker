
import time
import cv2

from src.utils import load_config
from src.detector import FaceDetector

# Load project configuration
cfg = load_config()

# Initialize face detector
det = FaceDetector(
    cfg["yolo_model"],
    cfg["detection_confidence"],
    cfg["min_face_size"]
)

# Open the configured video
cap = cv2.VideoCapture(cfg["video_source"])

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video source: {cfg['video_source']}"
    )

times = []

try:
    for i in range(60):
        ok, frame = cap.read()

        if not ok:
            break

        start_time = time.perf_counter()
        det.detect_and_track(frame)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        times.append(elapsed_ms)

    if times:
        print(
            "Average detection time per frame:",
            round(sum(times) / len(times), 1),
            "ms"
        )
        print("Frames processed:", len(times))
        print(
            "Approximate detector FPS:",
            round(1000 / (sum(times) / len(times)), 2)
        )
    else:
        print("No frames were processed. Check your video source.")

finally:
    cap.release()
