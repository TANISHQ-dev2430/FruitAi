import numpy as np
from PIL import Image
import cv2

def hsv_ripeness_score(img):

    img = np.array(Image.open(img).convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # mango/banana ripe color range (yellow/orange)
    lower = np.array([15, 80, 80])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    ripe_pixels = np.sum(mask > 0)
    total_pixels = img.shape[0] * img.shape[1]

    score = ripe_pixels / total_pixels
    return float(score)

def estimate_days_left(ripeness_score):
    score = ripeness_score
    if score < 0.4:
        return 4
    elif score < 0.7:
        return 2
    else:
        return 0.5

def estimate_ripeness(image, fruit):
    hsv_score = hsv_ripeness_score(image)       # 0 to 1
    ripeness_percentage = round(hsv_score * 100, 1)
    days_left = estimate_days_left(hsv_score)

    label = "ripe" if ripeness_percentage >= 60 else "unripe"

    return {
        "label": label,
        "ripeness_score": ripeness_percentage,
        "days_left": days_left,
    }
