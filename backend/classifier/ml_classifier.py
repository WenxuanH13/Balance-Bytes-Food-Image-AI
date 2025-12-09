# backend/classifier/ml_classifier.py

import os
import io
import json
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "food_classifier_model_v0.keras")
MODEL_PATH = os.path.abspath(MODEL_PATH)

CLASS_INDICES_PATH = os.path.join(BASE_DIR, "..", "model", "food_class_indices.json")
CLASS_INDICES_PATH = os.path.abspath(CLASS_INDICES_PATH)

print("Loading classifier model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading class indices from:", CLASS_INDICES_PATH)
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)  

index_to_class = {idx: name for name, idx in class_indices.items()}

IMG_SIZE = (224, 224)


def preprocess_image_bytes(image_bytes: bytes):
    """Convert raw image bytes -> preprocessed batch for the classifier."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def classify_food(image_bytes: bytes):
    arr = preprocess_image_bytes(image_bytes)
    preds = model.predict(arr)[0] 

    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = index_to_class.get(idx, f"class_{idx}")

    return {
        "label": label,
        "confidence": confidence,
        "index": idx,
    }
