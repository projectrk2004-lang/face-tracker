
import cv2
from src.utils import load_config
from src.detector import FaceDetector


def main():
    cfg = load_config()

    det = FaceDetector(
        cfg["yolo_model"],
        cfg["detection_confidence"],
        cfg["min_face_size"]
    )

    cap = cv2.VideoCapture(cfg["video_source"])

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    try:
        for i in range(1, 101):
            ok, frame = cap.read()

            if not ok:
                print("Video ended or frame could not be read.")
                break

            if i % 10 == 0:
                detections = det.detect_and_track(frame)
                print("Frame", i, "->", detections)

    finally:
        cap.release()


if __name__ == "__main__":
    main()
