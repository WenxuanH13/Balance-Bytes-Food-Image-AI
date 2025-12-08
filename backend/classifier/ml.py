import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Path to model file (adjust if needed)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "food_classifier.h5")
MODEL_PATH = os.path.abspath(MODEL_PATH)

print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(file):
    arr = preprocess_image(file)
    pred = float(model.predict(arr)[0][0])
    label = "food" if pred > 0.5 else "non-food"
    return label, pred