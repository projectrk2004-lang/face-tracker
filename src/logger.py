
import logging
import os
from datetime import datetime
import cv2


def setup_logger(log_file):
    """Create a logger that writes to a file and prints on screen."""

    folder = os.path.dirname(log_file)

    if folder:
        os.makedirs(folder, exist_ok=True)

    logger = logging.getLogger("tracker")

    if logger.handlers:  # Already set up
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(message)s")

    file_handler = logging.FileHandler(
        log_file, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console)

    return logger


def save_face_image(crop, event_type, face_id):
    """
    Save a cropped face under logs/<event_type>s/YYYY-MM-DD/.
    event_type must be 'entry' or 'exit'.
    Returns the saved file path.
    """

    if event_type not in ("entry", "exit"):
        raise ValueError("event_type must be 'entry' or 'exit'")

    if crop is None or crop.size == 0:
        raise ValueError("Cannot save an empty face crop")

    now = datetime.now()

    folder = os.path.join(
        "logs",
        event_type + "s",
        now.strftime("%Y-%m-%d")
    )

    os.makedirs(folder, exist_ok=True)

    name = (
        f"face_{face_id}_{event_type}_"
        f"{now.strftime('%H%M%S_%f')}.jpg"
    )

    path = os.path.join(folder, name)

    success = cv2.imwrite(path, crop)

    if not success:
        raise OSError(f"Failed to save face image: {path}")

    return path
