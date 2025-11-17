import tensorflow as tf
import numpy as np
from PIL import Image

# Load MobileNetV2 model (classification)
MODEL_PATH = "models/mobilenetv2_fruit_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (must match your training dataset)
CLASS_NAMES = [
    "apple", "banana", "cherry", "chickoo", "grapes",
    "kiwi", "mango", "orange", "strawberry"
]

IMG_SIZE = 224

def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype("float32")
    img = img / 255.0                     # normalize
    return np.expand_dims(img, axis=0)

def predict_fruit(image):
    img = preprocess_image(image)
    preds = model.predict(img)[0]
    index = np.argmax(preds)
    confidence = float(preds[index])
    fruit = CLASS_NAMES[index]
    return {"fruit": fruit, "confidence": confidence}
