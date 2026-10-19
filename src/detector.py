
from ultralytics import YOLO


class FaceDetector:
    """YOLO face detection + ByteTrack tracking in one call."""

    def __init__(self, model_path, conf, min_size):
        self.model = YOLO(model_path)
        self.conf = conf
        self.min_size = min_size

    def detect_and_track(self, frame):
        """Return a list of (track_id, [x1, y1, x2, y2], confidence)."""

        result = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        detections = []

        if result.boxes is None or result.boxes.id is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, tid, confidence in zip(boxes, ids, confs):
            w = box[2] - box[0]
            h = box[3] - box[1]

            if w < self.min_size or h < self.min_size:
                continue  # Too small for reliable face recognition

            detections.append(
                (int(tid), box.tolist(), float(confidence))
            )

        return detections
