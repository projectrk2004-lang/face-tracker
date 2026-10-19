
import cv2
from src.utils import load_config
from src.recognizer import FaceRecognizer


def main():
    cfg = load_config()

    rec = FaceRecognizer(
        cfg["similarity_threshold"],
        cfg["use_gpu"]
    )

    img = cv2.imread("data/test_face.jpg")

    if img is None:
        print("Error: Image not found or could not be loaded.")
        return

    emb = rec.get_embedding(img)

    if emb is None:
        print("No face detected in the image.")
    else:
        print("Embedding shape:", emb.shape)
        print("Embedding dtype:", emb.dtype)


if __name__ == "__main__":
    main()
