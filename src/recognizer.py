
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    """Creates face embeddings using InsightFace and matches them."""

    def __init__(self, threshold, use_gpu=False):
        if use_gpu:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]
        else:
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=providers
        )

        self.app.prepare(
            ctx_id=0 if use_gpu else -1,
            det_size=(640, 640)
        )

        self.threshold = threshold

    def get_embedding(self, crop):
        """Return a normalized 512-number embedding, or None."""

        if crop is None or crop.size == 0:
            return None

        faces = self.app.get(crop)

        if not faces:
            return None

        # If multiple faces are found, select the largest one.
        best = max(
            faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )
        )

        return best.normed_embedding.astype(np.float32)

    def match(self, embedding, known_faces):
        """
        known_faces = {face_id: embedding}
        Returns (face_id, score) for a match,
        or (None, best_score) for an unknown face.
        """

        if embedding is None:
            return None, -1.0

        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
            raise ValueError("Embedding must be a finite 1D array")

        if not known_faces:
            return None, -1.0

        best_id = None
        best_score = -1.0

        for face_id, saved in known_faces.items():
            saved = np.asarray(saved, dtype=np.float32)

            if saved.shape != embedding.shape:
                continue

            if not np.all(np.isfinite(saved)):
                continue

            score = float(np.dot(embedding, saved))

            if score > best_score:
                best_id = face_id
                best_score = score

        if best_id is not None and best_score >= self.threshold:
            return best_id, best_score

        return None, best_score
