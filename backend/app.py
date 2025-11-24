# backend/app.py
import io
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import numpy as np
import uvicorn
import base64
import logging

# ML libs loaded lazily to avoid import errors on startup on machines without GPU libs
TF_AVAILABLE = True
TORCH_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except Exception:
    TF_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50
except Exception:
    TORCH_AVAILABLE = False

# CONFIG
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESNET50_PATH = MODELS_DIR / "best_resnet50_final.pth"      # ResNet50 model weights
MOBILENET_PATH = MODELS_DIR / "mobilenetv2_fruit_model.h5" # put your mobilenet .h5 here
MOBILENET_CLASSES_PATH = MODELS_DIR / "mobilenet_classes.json"  # class names mapping
DISEASE_CLASSES_PATH = MODELS_DIR / "disease_classes.json"  # disease class names

# FastAPI app
app = FastAPI(title="FruitAI Backend")

# Allow requests from frontend dev server (adjust origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],  # set explicit origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

# Simple response models
class Detection(BaseModel):
    label: str
    confidence: float
    bbox: Optional[List[float]] = None  # [x1,y1,x2,y2] absolute pixel coords

class FruitResult(BaseModel):
    fruit: str
    fruit_confidence: Optional[float]
    

class RipenessResult(BaseModel):
    ripeness_score: float
    estimated_days_left: Optional[float]
    clip_ripe_prob: Optional[float]
    hsv_ripe_prob: Optional[float]
    fused_ripe_prob: Optional[float]
    label: Optional[str]

# GLOBAL MODELS (loaded on startup)
resnet50_model = None
mobilenet_model = None
mobilenet_classes = None
disease_classes = None
yolo_model = None
YOLO_PATH = MODELS_DIR / "best-yolotrainedmodel.pt"

@app.on_event("startup")
def load_models():
    global resnet50_model, mobilenet_model, mobilenet_classes, disease_classes
    global yolo_model
    # Load ResNet50
    if TORCH_AVAILABLE and RESNET50_PATH.exists():
        try:
            resnet50_model = resnet50(weights=None)
            # Load the state dict
            state_dict = torch.load(str(RESNET50_PATH), map_location="cpu")
            # The model has 4 output classes, so we need to adjust the final layer
            # Get number of classes from state_dict
            num_classes = state_dict['fc.weight'].shape[0]
            # Modify the model's fc layer to match
            resnet50_model.fc = torch.nn.Linear(resnet50_model.fc.in_features, num_classes)
            resnet50_model.load_state_dict(state_dict)
            resnet50_model.eval()
            logger.info("Loaded ResNet50 model from %s with %d classes", RESNET50_PATH, num_classes)
        except Exception as e:
            logger.exception("Failed to load ResNet50 model: %s", e)
            resnet50_model = None
    else:
        logger.warning("PyTorch ResNet50 not available or model not found at %s", RESNET50_PATH)

    # Load YOLO (prefer ultralytics package if installed, otherwise torch.hub)
    if TORCH_AVAILABLE and YOLO_PATH.exists():
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                # Try ultralytics package first (more robust if installed)
                from ultralytics import YOLO as UltralyticsYOLO
                yolo_model = UltralyticsYOLO(str(YOLO_PATH))
                try:
                    yolo_model.to(device)
                except Exception:
                    pass
                logger.info("Loaded YOLO (ultralytics) model from %s, device=%s", YOLO_PATH, device)
            except Exception:
                # Fallback to torch.hub yolov5
                yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(YOLO_PATH))
                try:
                    yolo_model.to(device)
                except Exception:
                    pass
                yolo_model.eval()
                logger.info("Loaded YOLO (torch.hub yolov5) model from %s, device=%s", YOLO_PATH, device)
        except Exception as e:
            logger.exception("Failed to load YOLO model: %s", e)
            yolo_model = None
    else:
        logger.warning("YOLO model not found or PyTorch missing at %s", YOLO_PATH)

    # Load MobileNet (TF)
    if TF_AVAILABLE and MOBILENET_PATH.exists():
        try:
            mobilenet_model = tf.keras.models.load_model(str(MOBILENET_PATH))
            # optional: warmup
            logger.info("Loaded MobileNet model from %s", MOBILENET_PATH)
        except Exception as e:
            logger.exception("Failed to load MobileNet model: %s", e)
            mobilenet_model = None
    else:
        logger.warning("TensorFlow MobileNet not loaded or model missing at %s", MOBILENET_PATH)

    # Load MobileNet class names
    if MOBILENET_CLASSES_PATH.exists():
        try:
            import json
            with open(MOBILENET_CLASSES_PATH, 'r') as f:
                mobilenet_classes = json.load(f)
            logger.info("Loaded MobileNet classes from %s", MOBILENET_CLASSES_PATH)
        except Exception as e:
            logger.exception("Failed to load MobileNet classes: %s", e)
            mobilenet_classes = None
    else:
        logger.warning("MobileNet classes file not found at %s", MOBILENET_CLASSES_PATH)

    # Load Disease class names
    if DISEASE_CLASSES_PATH.exists():
        try:
            import json
            with open(DISEASE_CLASSES_PATH, 'r') as f:
                disease_classes = json.load(f)
            logger.info("Loaded disease classes from %s", DISEASE_CLASSES_PATH)
        except Exception as e:
            logger.exception("Failed to load disease classes: %s", e)
            disease_classes = None
    else:
        logger.warning("Disease classes file not found at %s", DISEASE_CLASSES_PATH)

# Helper: read image bytes -> PIL RGB
def read_imagefile(file_bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

# Helper: bytes -> base64 (if needed)
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Endpoint: classify fruit (simple wrapper)
@app.post("/api/predict/fruit", response_model=FruitResult)
async def predict_fruit(file: UploadFile = File(...)):
    contents = await file.read()
    img = read_imagefile(contents)

    if mobilenet_model is None:
        # If TF model not present, return fallback
        return {"fruit": "unknown", "fruit_confidence": None}
    # prepare for model: resize to 224 and preprocess
    arr = np.array(img.resize((224,224))).astype("float32")
    arr = preprocess_input(arr)  # MobileNetV2 preprocessing
    arr = np.expand_dims(arr, 0)
    preds = mobilenet_model.predict(arr)[0]
    
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    
    # Use loaded class names if available
    if mobilenet_classes and len(mobilenet_classes) > top_idx:
        label = mobilenet_classes[top_idx]
        # Clean up label: remove " fruit" suffix and convert to lowercase
        label = label.replace(" fruit", "").replace("fruit ", "").strip().lower()
    else:
        # fallback
        label = f"class_{top_idx}"

    # User-requested workaround: map frequent incorrect 'chickoo' predictions to 'apple'
    if isinstance(label, str) and label.lower() == "chickoo":
        label = "apple"

    return {"fruit": label, "fruit_confidence": confidence}

# Endpoint: ripeness estimation (calls internal fusion logic)
@app.post("/api/predict/ripeness", response_model=RipenessResult)
async def predict_ripeness(file: UploadFile = File(...), fruit: str = ""):
    # HSV-based ripeness estimation using color analysis
    contents = await file.read()
    img = read_imagefile(contents)
    np_img = np.array(img)[:,:,::-1].copy()  # BGR for cv2-like ops

    # Convert to HSV (OpenCV: H=0-180, S=0-255, V=0-255)
    import cv2
    hsv = cv2.cvtColor(np_img, cv2.COLOR_BGR2HSV).astype(float)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    fruit_l = (fruit or "").lower()
    
    if "banana" in fruit_l:
        # Banana ripeness: based on color progression and darkness (speckles)
        # Unripe: Green (H 40-90)
        # Ripe: Yellow (H 15-40)
        # Very Ripe: Brown/Black speckles (low V in yellow range)
        
        # Get the dominant color regions
        green_mask = ((h >= 40) & (h <= 90))  # Unripe/green
        yellow_mask = ((h >= 15) & (h <= 40))  # Ripe yellow
        
        # For ripeness: check if yellow dominates and has some darkening (speckles)
        # Brown spots are low V values in yellow hue range
        yellow_bright = ((h >= 15) & (h <= 40) & (v >= 100)).mean()  # Bright yellow
        yellow_dark = ((h >= 15) & (h <= 40) & (v >= 50) & (v < 100)).mean()  # Darker yellow (ripening)
        brown_spots = ((h >= 15) & (h <= 40) & (v < 50)).mean()  # Brown/black speckles (very ripe)
        green = green_mask.mean()
        
        # Ripeness calculation:
        # 0-30%: Mostly green
        # 30-60%: Transition to yellow
        # 60-85%: Yellow with some dark speckles
        # 85-100%: Mostly brown/dark (very ripe)
        if green > 0.4:
            ripe_prob = 0.2  # Unripe
        else:
            # Favor yellow+darkening for ripeness
            ripe_prob = float(np.clip(yellow_bright * 0.4 + yellow_dark * 0.4 + brown_spots * 0.8, 0, 1))
            # Boost ripeness if there are brown spots (very ripe indicator)
            if brown_spots > 0.05:
                ripe_prob = float(np.clip(ripe_prob + 0.3, 0, 1))
        
    elif "mango" in fruit_l:
        # Mango ripeness: transitions from green to yellow-orange to red-orange
        # Unripe: Green (H 40-90)
        # Ripe: Yellow-Orange (H 15-35)
        # Very Ripe: Red-Orange (H 0-15)
        
        green = ((h >= 40) & (h <= 90) & (s >= 40)).mean()
        yellow_orange = ((h >= 20) & (h <= 35) & (s >= 50) & (v >= 100)).mean()
        red_orange = ((h >= 0) & (h <= 20) & (s >= 60) & (v >= 100)).mean()
        dark_spots = ((h >= 0) & (h <= 35) & (v < 80)).mean()
        
        if green > 0.3:
            ripe_prob = 0.3  # Unripe
        else:
            # Combine yellow-orange and red-orange indicators
            ripe_prob = float(np.clip(yellow_orange * 0.6 + red_orange * 0.8 + dark_spots * 0.3, 0, 1))
        
    elif "apple" in fruit_l:
        # Apple ripeness: based on red saturation and brightness
        # Unripe: Lighter red/green
        # Ripe: Deep red with high saturation
        # Very Ripe: Dark red with brown tones
        
        red_mask = ((h <= 15) | (h >= 160))
        red_bright = (red_mask & (s >= 60) & (v >= 120)).mean()
        red_deep = (red_mask & (s >= 50) & (v >= 80) & (v < 120)).mean()
        red_dark = (red_mask & (s >= 40) & (v < 80)).mean()
        
        green_unripe = ((h >= 40) & (h <= 90)).mean()
        
        if green_unripe > 0.2:
            ripe_prob = 0.3
        else:
            # Prefer deep red and dark red over bright red
            ripe_prob = float(np.clip(red_bright * 0.4 + red_deep * 0.7 + red_dark * 0.5, 0, 1))
        
    else:
        # Default: analyze overall color saturation and brightness
        # Ripe fruits tend to have higher saturation and slightly darker values
        saturated = (s >= 50).mean()
        bright = (v >= 100).mean()
        mid_dark = ((v >= 80) & (v < 120)).mean()
        ripe_prob = float(np.clip(saturated * 0.4 + bright * 0.3 + mid_dark * 0.3, 0.2, 1))

    ripeness_score = round(ripe_prob * 100, 1)
    estimated_days_left = round(max(0.0, (1.0 - ripe_prob) * 7.0), 1)

    return {
        "ripeness_score": ripeness_score,
        "estimated_days_left": estimated_days_left,
        "clip_ripe_prob": None,
        "hsv_ripe_prob": round(ripe_prob, 3),
        "fused_ripe_prob": round(ripe_prob, 3),
        "label": "ripe" if ripe_prob >= 0.5 else "unripe"
    }

# Endpoint: disease detection (ResNet50 classification)
@app.post("/api/predict/disease")
async def predict_disease(file: UploadFile = File(...), fruit: str = ""):
    contents = await file.read()
    img = read_imagefile(contents)
    if resnet50_model is None:
        raise HTTPException(status_code=503, detail="ResNet50 model not loaded on server")

    # Prepare image for ResNet50 inference
    try:
        # Resize to 224x224
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Convert to tensor (float32) and add batch dimension
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        
        # Run inference
        with torch.no_grad():
            outputs = resnet50_model(img_tensor)
        
        # Get predictions
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)
        
        # Log all probabilities for debugging
        logger.info("Disease detection probabilities: %s", [f"{p:.4f}" for p in probs.tolist()])
        logger.info("Predicted class: %d with confidence: %.4f", int(pred_idx), float(confidence))
        
        # Set confidence threshold - if below threshold, mark as Healthy
        CONFIDENCE_THRESHOLD = 0.5
        
        # Get disease label from disease_classes if available
        disease_label = "Unknown"
        final_confidence = float(confidence)
        
        if final_confidence >= CONFIDENCE_THRESHOLD:
            # High confidence - use the predicted disease
            if disease_classes and len(disease_classes) > int(pred_idx):
                disease_label = disease_classes[int(pred_idx)]
            else:
                disease_label = f"disease_class_{int(pred_idx)}"
        else:
            # Low confidence - mark as Healthy
            disease_label = "Healthy"
            final_confidence = 1.0  # Set confidence to 1.0 for healthy
        
        detections = [{
            "label": disease_label,
            "confidence": final_confidence
        }]
        
    except Exception as e:
        logger.exception("ResNet50 inference failed: %s", e)
        raise HTTPException(status_code=500, detail="ResNet50 inference failed")

    return {"detections": detections}


# Endpoint: YOLO detection (for live webcam frames or single image)
@app.post("/api/predict/yolo")
async def predict_yolo(file: UploadFile = File(...), conf: float = Form(0.25)):
    contents = await file.read()
    img = read_imagefile(contents)
    if yolo_model is None:
        # Return empty detections (allow frontend Live mode to run without failing)
        logger.warning("YOLO model not loaded; returning empty detections to client")
        return {"detections": [], "model_loaded": False}

    try:
        # Run YOLO inference (support ultralytics YOLO and torch.hub yolov5 result formats)
        img_np = np.array(img)
        results = None
        try:
            # ultralytics YOLO object is callable and returns a Results object/list
            results = yolo_model(img_np, imgsz=640, conf=conf)
        except TypeError:
            # some models (torch.hub yolov5) accept just the image
            results = yolo_model(img_np)

        # Normalize extraction
        detections = []
        # If results provides pandas accessor (both yolov5 and ultralytics often do)
        try:
            df = results.pandas().xyxy[0]
            for _, row in df.iterrows():
                if float(row['confidence']) < float(conf):
                    continue
                detections.append({
                    "label": str(row.get('name', 'unknown')),
                    "confidence": float(row['confidence']),
                    "bbox": [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
                })
        except Exception:
            # Fallback: try to parse ultralytics-v8 Results -> results[0].boxes
            res0 = results[0] if isinstance(results, (list, tuple)) else results
            boxes = getattr(res0, 'boxes', None)
            names = getattr(res0, 'names', {}) or {}
            if boxes is not None:
                # boxes.xyxy, boxes.conf, boxes.cls are tensors
                try:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy()
                except Exception:
                    # Older structures: boxes.data as tensor Nx6 (xyxy+conf+cls)
                    try:
                        data = boxes.data.cpu().numpy()
                        xyxy = data[:, :4]
                        confs = data[:, 4]
                        cls = data[:, 5].astype(int)
                    except Exception:
                        xyxy = np.array([])
                        confs = np.array([])
                        cls = np.array([])

                for i in range(len(confs)):
                    if confs[i] < conf:
                        continue
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    label = names.get(int(cls[i]), str(int(cls[i]))) if len(names) else str(int(cls[i]))
                    detections.append({
                        "label": label,
                        "confidence": float(confs[i]),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
            else:
                # Last resort: no detections
                detections = []
    except Exception as e:
        logger.exception("YOLO inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"YOLO inference failed: {e}")

    return {"detections": detections}

# Simple health-check
@app.get("/api/ping")
async def ping():
    return {"status": "ok", "resnet50_loaded": bool(resnet50_model), "mobilenet_loaded": bool(mobilenet_model)}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
