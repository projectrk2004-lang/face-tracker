
import time
import cv2

from src.database import Database
from src.detector import FaceDetector
from src.logger import save_face_image, setup_logger
from src.recognizer import FaceRecognizer
from src.tracker import VisitorTracker
from src.utils import load_config


def draw(frame, detections, tracker, count):
    """Draw boxes, face IDs and the unique visitor count."""

    for tid, box, conf in detections:
        x1, y1, x2, y2 = map(int, box)

        face_id = tracker.state.get(tid, {}).get("face_id", "?")

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"ID {face_id}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Unique visitors: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )


def main():
    cfg = load_config()

    logger = setup_logger(cfg["log_file"])
    db = Database(cfg["db_path"])

    cap = None
    tracker = None
    frame_no = 0

    try:
        detector = FaceDetector(
            cfg["yolo_model"],
            cfg["detection_confidence"],
            cfg["min_face_size"]
        )

        recognizer = FaceRecognizer(
            cfg["similarity_threshold"],
            cfg["use_gpu"]
        )

        tracker = VisitorTracker(
            cfg,
            db,
            recognizer,
            logger,
            save_face_image
        )

        source = cfg["video_source"]

        is_stream = (
            isinstance(source, str)
            and source.lower().startswith(
                ("rtsp://", "http://", "https://")
            )
        )

        # Convert webcam index strings such as "0" to integer 0.
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video source: {source}"
            )

        frame_no = 0
        retries = 0
        last_dets = []

        logger.info(
            f"SYSTEM_START | source={source} | "
            f"frame_skip={cfg['frame_skip']}"
        )

        while True:
            ok, frame = cap.read()

            if not ok:
                if not is_stream:
                    break

                retries += 1

                if retries > cfg["reconnect_attempts"]:
                    logger.error("STREAM_LOST | giving up")
                    break

                logger.warning(
                    f"STREAM_RECONNECT | attempt={retries}"
                )

                cap.release()
                time.sleep(cfg["reconnect_delay_sec"])
                cap = cv2.VideoCapture(source)
                continue

            retries = 0
            frame_no += 1

            if frame_no % cfg["frame_skip"] == 0:
                last_dets = detector.detect_and_track(frame)
                tracker.update(frame, last_dets, frame_no)

            if cfg["show_window"]:
                draw(
                    frame,
                    last_dets,
                    tracker,
                    db.get_unique_count()
                )

                cv2.imshow("Face Tracker", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("INTERRUPTED | shutdown requested")

    finally:
        try:
            if tracker is not None:
                tracker.flush(frame_no)

            logger.info(
                f"SYSTEM_STOP | unique_visitors="
                f"{db.get_unique_count()}"
            )
        finally:
            if cap is not None:
                cap.release()

            cv2.destroyAllWindows()
            db.close()


if __name__ == "__main__":
    main()
