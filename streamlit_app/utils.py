import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image, target_size):
    image = image.convert("RGB")
    image = image.resize(target_size)

    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict(model, image_array, class_names):
    preds = model.predict(image_array)
    confidence = float(np.max(preds))
    class_idx = int(np.argmax(preds))
    return class_names[class_idx], confidence

import json

def load_class_names(path):
    with open(path, "r") as f:
        return json.load(f)
    