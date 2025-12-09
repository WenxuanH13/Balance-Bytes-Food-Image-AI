# backend/classifier/ml_detector.py

import os
import io
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "food_detector_model_v6.keras")
MODEL_PATH = os.path.abspath(MODEL_PATH)

print("Loading detector model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)


def preprocess_image_bytes(image_bytes: bytes):
    """Convert raw image bytes -> preprocessed batch for the detector."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def detect_food(image_bytes: bytes):
    arr = preprocess_image_bytes(image_bytes)
    prob_food = float(model.predict(arr)[0][0]) 
    prob_non_food = 1.0 - prob_food
    is_food = prob_food >= 0.5

    return {
        "is_food": is_food,
        "prob_food": prob_food,
        "prob_non_food": prob_non_food,
    }
