# FruitAi

FruitClassifier is a research and demo repository that provides image-based fruit analysis: fruit type classification, ripeness estimation, and disease detection. The project is split into a Python FastAPI backend that serves model inference endpoints and a React/Vite frontend that provides an interactive UI with webcam support.

This README documents the repository layout, the included models, how to run the services locally, and the available API endpoints.

Contents
- Overview
- Models included
- Project layout
- Local setup and run instructions
- API reference and examples
- Notes and troubleshooting

Overview
--------
The system implements three core capabilities:
- Fruit classification: a MobileNetV2-based model for identifying fruit types (apple, banana, mango).
- Ripeness estimation: a heuristic and learned hybrid approach using color (HSV) analysis and learned models to produce a ripeness score and estimate days to ripen.
- Disease detection: an object-detection model (YOLO family) that detects disease instances on fruit with bounding boxes, labels and confidence scores.

The backend exposes REST endpoints for these tasks and the frontend consumes them to present an interactive UI including a live webcam mode.

Models included
---------------
Model files are expected under `backend/models/`:
- `best_resnet50_final.pth` — ResNet50-based disease classifier (used for disease classification endpoints).
- `mobilenetv2_fruit_model.h5` — MobileNetV2 fruit classifier for fruit type detection.
- `best-yolotrainedmodel.pt` — YOLO trained model for disease detection (used for live webcam detection).
- `mobilenet_classes.json`, `disease_classes.json` — optional mapping files for class indices to human-readable labels.

Project layout
--------------
Top-level layout (relevant files/folders):

- `backend/`
	- `app.py` — FastAPI application and model loading/inference endpoints.
	- `models/` — place model weights and class JSONs here.
	- `requirements.txt` — backend Python dependencies (if present).

- `frontend/fruitai/`
	- React + Vite frontend application.
	- `src/components/` — UI components including webcam capture.
	- `src/pages/` — UI pages such as Home and Results.
	- `src/services/api.js` — API helper functions used by the frontend.

Local setup
-----------
Prerequisites
- Python 3.8+ with a virtual environment.
- Node.js 16+ / npm or pnpm for frontend.
- Optional: CUDA-enabled GPU and appropriate PyTorch build for hardware acceleration.

Backend
1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install backend dependencies (from `backend/`):

```powershell
cd backend
pip install -r requirements.txt
# If you plan to use ultralytics YOLOv8, install it (example):
pip install ultralytics
```

3. Place the model files into `backend/models/`.

4. Start the backend server:

```powershell
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Frontend
1. Install dependencies and start dev server (from repository root):

```powershell
cd frontend\fruitai
npm install
npm run dev
```

2. Open the URL shown by Vite (usually `http://localhost:5173`) and navigate to the Upload/Live page. The Live webcam mode uses the YOLO endpoint for real-time detections.

API reference
-------------
All endpoints are under the `/api` prefix on the backend. Example base URL: `http://localhost:8000/api`.

- `POST /api/predict/fruit` — classify fruit type
	- Form field: `file` (image file)
	- Response: `{ fruit: <label>, fruit_confidence: <float> }

- `POST /api/predict/ripeness` — estimate ripeness
	- Form fields: `file` (image), optional `fruit` (string)
	- Response: ripeness structure with `ripeness_score`, `estimated_days_left`, and other fields

- `POST /api/predict/disease` — classify disease on a single uploaded image
	- Form fields: `file` (image), optional `fruit` (string)
	- Response: `{ detections: [ { label, confidence }, ... ] }

- `POST /api/predict/yolo` — run YOLO detection (used by Live webcam)
	- Form fields: `file` (image), optional `conf` (confidence threshold float)
	- Response: `{ detections: [ { label, confidence, bbox: [x1,y1,x2,y2] }, ... ], model_loaded: <bool> }

- `GET /api/ping` — basic health check that reports which models are loaded

Curl example (YOLO):

```bash
curl -F "file=@/path/to/image.jpg" -F "conf=0.25" http://localhost:8000/api/predict/yolo
```

Notes and troubleshooting
-------------------------
- If the YOLO endpoint returns empty detections, check server logs to verify the YOLO model loaded successfully. The backend attempts to load YOLO using the `ultralytics` package (if installed) or falls back to `torch.hub` for YOLOv5.
- If using GPU acceleration, ensure PyTorch is installed with the matching CUDA version. The backend will automatically use CUDA if available.
- Cross-origin requests: the backend enables CORS for local development. In production, restrict `allow_origins` in `backend/app.py`.

Contributing and next steps
---------------------------
- Improvements you might consider:
	- Add end-to-end tests and evaluation scripts (mAP for detection, classification accuracy for crops).
	- Implement a fused pipeline (YOLO detection → crop → ResNet classification) for finer-grained disease/fine-class labeling.
	- Provide model download or installation scripts to place model files in `backend/models/` automatically.

If you'd like, I can add documentation pages for API contracts, example clients, or CI steps to validate model loading at startup.


