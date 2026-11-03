from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "ASD_VGG16_LSTM_model.keras"
)

_vgg = None
_lstm = None
_prediction_lock = Lock()


class VideoProcessingError(ValueError):
    """Raised when an uploaded video cannot be read."""
    pass


def predict_video(video_path):
    """Return experimental class scores for one video."""
    global _vgg, _lstm

    # Read five evenly spaced frames, matching training.
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    try:
        if not cap.isOpened():
            raise VideoProcessingError(
                "Could not open the uploaded video."
            )

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            raise VideoProcessingError(
                "Video has no usable frame count."
            )

        indices = np.linspace(0, total - 1, 5, dtype=int)

        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = cap.read()

            if success:
                frames.append(cv2.resize(frame, (224, 224)))
    finally:
        cap.release()

    if not frames:
        raise VideoProcessingError(
            "Could not read any sampled video frames."
        )

    images = np.stack(frames).astype(np.float32)

    # Preserve the color handling and preprocessing used in training.
    images = preprocess_input(images)

    # Load models once and process one prediction at a time.
    with _prediction_lock:
        if _lstm is None:
            _lstm = load_model(MODEL_PATH, compile=False)

        if _vgg is None:
            _vgg = VGG16(weights="imagenet", include_top=True)

        features = _vgg.predict(
            images, batch_size=5, verbose=0
        )

        # Pad missing sampled frames with zeros, matching training.
        sequence = np.zeros((5, 1000), dtype=np.float32)
        sequence[:len(features)] = features

        model_input = np.expand_dims(sequence, axis=0)
        scores = _lstm.predict(model_input, verbose=0)[0]

    if scores.shape != (2,) or not np.all(np.isfinite(scores)):
        raise RuntimeError("Model returned an invalid output.")

    # Training labels: 0 = ASD, 1 = TD.
    label = ["ASD", "TD"][int(np.argmax(scores))]

    return {
        "predicted_class": label,
        "model_scores": {
            "ASD": float(scores[0]),
            "TD": float(scores[1]),
        },
        "frames_extracted": len(frames),
        "experimental": True,
        "notice": (
            "Academic experiment only. Not a clinically validated "
            "diagnostic or screening tool. Scores are not "
            "clinical probabilities."
        ),
    }