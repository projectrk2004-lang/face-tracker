
import json
import cv2


def load_config(path="config.json"):
    """Read config.json and return it as a Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def crop_face(frame, box, pad=0.25, min_side=160):
    """
    Crop a face from the frame with some extra padding.
    Very small crops are enlarged so InsightFace can read them.
    Returns None if the crop is empty.
    """
    x1, y1, x2, y2 = [int(v) for v in box[:4]]

    w, h = x2 - x1, y2 - y1

    px, py = int(w * pad), int(h * pad)

    H, W = frame.shape[:2]

    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(W, x2 + px), min(H, y2 + py)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    short_side = min(crop.shape[:2])

    if short_side < min_side:
        scale = min_side / short_side
        crop = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    return crop
