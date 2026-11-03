from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

VIDEO_PATH = r"C:\Users\gayat\Downloads\Subj_81_part_8(ASD).mp4"
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "ASD_VGG16_LSTM_model.keras"
)

# Sample five evenly spaced frames, matching proper_train.py.
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []

try:
    if not cap.isOpened():
        raise ValueError("Could not open the video.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError("Video has no usable frame count.")

    indices = np.linspace(0, total - 1, 5, dtype=int)

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()

        if success:
            frames.append(cv2.resize(frame, (224, 224)))
finally:
    cap.release()

if not frames:
    raise ValueError("Could not read any sampled frames.")

print("Frames extracted:", len(frames))

# Preserve the same color handling and preprocessing as training.
images = np.stack(frames).astype(np.float32)
images = preprocess_input(images)

print("Loading VGG16...")
vgg = VGG16(weights="imagenet", include_top=True)
features = vgg.predict(images, batch_size=5, verbose=0)

# Pad missing frames with zeros, matching training.
sequence = np.zeros((5, 1000), dtype=np.float32)
sequence[:len(features)] = features
model_input = np.expand_dims(sequence, axis=0)

print("Model input shape:", model_input.shape)

print("Loading LSTM model...")
model = load_model(MODEL_PATH, compile=False)
scores = model.predict(model_input, verbose=0)[0]

if scores.shape != (2,) or not np.all(np.isfinite(scores)):
    raise ValueError("Model returned an invalid output.")

# Training labels: 0 = ASD, 1 = TD.
label = ["ASD", "TD"][int(np.argmax(scores))]

print("Experimental predicted class:", label)
print("ASD model score:", float(scores[0]))
print("TD model score:", float(scores[1]))
print("Academic experiment only; not a medical diagnosis.")