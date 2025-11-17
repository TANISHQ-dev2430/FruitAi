from ultralytics import YOLO
import numpy as np
from PIL import Image

MODEL_PATH = "models/best-yolotrainedmodel.pt"

# Load YOLO model
yolo_model = YOLO(MODEL_PATH)

def detect_disease(image):
    img = Image.open(image).convert("RGB")
    results = yolo_model(img)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = r.names[cls_id]
            detections.append({"label": label, "confidence": conf})

    if len(detections) == 0:
        return {"disease": "No disease detected", "confidence": None}

    # return top detection
    top = max(detections, key=lambda x: x["confidence"])
    return {
        "disease": top["label"],
        "confidence": top["confidence"]
    }
