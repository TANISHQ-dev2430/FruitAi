from PIL import Image
import numpy as np

def load_image(image_file):
    img = Image.open(image_file).convert("RGB")
    return img

def pil_to_array(img):
    return np.array(img)

def resize_image(img, size=(224, 224)):
    return img.resize(size)
