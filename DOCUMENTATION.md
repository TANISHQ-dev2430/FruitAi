# FruitClassifier - Complete Project Documentation

## Table of Contents
1. Project Overview
2. Architecture
3. Directory Structure and Components
4. Technologies Used
5. Detailed Component Breakdown
6. Machine Learning Models
7. API Endpoints Reference
8. Setup and Installation
9. Running the Application
10. Data Flow
11. User Guide
12. Troubleshooting
13. Future Enhancements

---

## 1. Project Overview

FruitClassifier is a comprehensive web-based application that leverages deep learning and computer vision to analyze fruit images. The system provides three main capabilities:

1. **Fruit Classification**: Identifies the type of fruit (Apple, Banana, Mango)
2. **Ripeness Estimation**: Calculates how ripe a fruit is and estimates days until it reaches optimal ripeness
3. **Disease Detection**: Detects diseases on fruits using real-time YOLO object detection

The application is built with a modern architecture separating the backend (API server) and frontend (user interface), allowing for scalability and independent deployment.

### Key Features
- Real-time webcam stream analysis with YOLO detection
- Single image upload for comprehensive fruit analysis
- Ripeness estimation with HSV color analysis
- Disease classification with confidence scores
- Analysis history tracking
- Responsive user interface with modern UI components

---

## 2. Architecture

The application follows a **Client-Server Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND (React/Vite)                   │
│         Running on localhost:5173 (Development)              │
│  - User Interface                                             │
│  - Webcam Integration                                         │
│  - Real-time Detection Display                                │
│  - Results Visualization                                      │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST API
                     │ (JSON over HTTP)
┌────────────────────▼────────────────────────────────────────┐
│                 BACKEND (FastAPI)                            │
│         Running on localhost:8000                            │
│  - API Endpoints                                              │
│  - Model Inference                                            │
│  - Request Processing                                        │
│  - Data Persistence                                          │
└─────────────────────────────────────────────────────────────┘
```

### Communication Flow
1. User uploads image or enables live webcam
2. Frontend captures the image/frame
3. Frontend sends HTTP POST request to backend API
4. Backend processes the image using ML models
5. Backend returns JSON response with results
6. Frontend displays results to user

---

## 3. Directory Structure and Components

### Root Level
```
fruit_ai_app/
├── README.md                 # Quick start guide
├── DOCUMENTATION.md          # This file - comprehensive documentation
├── backend/                  # FastAPI backend server
├── frontend/                 # React/Vite frontend application
└── .gitignore               # Git ignore rules
```

### Backend Structure
```
backend/
├── app.py                    # Main FastAPI application (Primary file)
├── requirements.txt          # Python dependencies
├── models/                   # Machine learning model weights
│   ├── best_resnet50_final.pth              # ResNet50 disease classifier
│   ├── best-yolotrainedmodel.pt             # YOLO disease detection model
│   ├── mobilenetv2_fruit_model.h5           # MobileNetV2 fruit classifier
│   ├── disease_classes.json                 # Disease class names mapping
│   └── mobilenet_classes.json               # Fruit class names mapping
├── data/                     # Static data
│   └── nutrition.json        # Fruit nutrition information
├── saved_results/            # Historical analysis results (timestamped JSON files)
│   ├── 20251121T140442Z_mango.json
│   ├── 20251121T140733Z_mango.json
│   └── ...                   # More timestamped results
├── utils/                    # Utility functions (if present)
└── __pycache__/              # Python bytecode cache
```

### Frontend Structure
```
frontend/fruitai/
├── package.json              # Node.js dependencies and scripts
├── vite.config.js            # Vite build configuration
├── index.html                # HTML entry point
├── src/
│   ├── main.jsx              # React app entry point
│   ├── App.jsx               # Root React component
│   ├── App.css               # Main stylesheet
│   ├── index.css             # Global styles
│   ├── components/           # Reusable UI components
│   │   ├── WebCamCapture.jsx        # Webcam capture & live YOLO detection
│   │   ├── FileUpload.jsx           # File upload interface
│   │   ├── DiseaseBoxesCanvas.jsx   # Disease detection visualization
│   │   ├── Gauge.jsx                # Circular ripeness gauge
│   │   └── Loader.jsx               # Loading spinner
│   ├── pages/                # Page components
│   │   ├── Home.jsx          # Main analysis page (upload/webcam)
│   │   ├── Results.jsx       # Results display page
│   │   └── History.jsx       # Analysis history page
│   ├── services/             # API communication
│   │   └── api.js            # API helper functions
│   └── assets/               # Static assets (images, etc.)
├── public/                   # Public static files
└── node_modules/             # Installed npm packages
```

---

## 4. Technologies Used

### Backend Stack
- **FastAPI**: Modern Python web framework for building APIs
- **Uvicorn**: ASGI server to run FastAPI
- **PyTorch**: Deep learning framework for model inference
- **TensorFlow/Keras**: Used for MobileNetV2 model
- **NumPy**: Numerical computing library
- **OpenCV (cv2)**: Computer vision library for image processing
- **Pillow (PIL)**: Image processing library
- **Ultralytics**: YOLOv5/YOLOv8 library for object detection

### Frontend Stack
- **React 19**: Modern JavaScript library for building UI
- **Vite**: Next-generation frontend build tool
- **React Router**: Client-side routing
- **Axios**: HTTP client for API calls
- **React Webcam**: Webcam integration
- **React Circular Progressbar**: Visual ripeness gauge
- **Chart.js**: Data visualization library
- **ESLint**: Code quality linting

### Deployment & Development
- **Node.js/npm**: JavaScript runtime and package manager
- **Python 3.8+**: Python runtime with virtual environments
- **CORS**: Cross-Origin Resource Sharing for API access

---

## 5. Detailed Component Breakdown

### 5.1 Backend Components

#### Main Application File: `backend/app.py`

This is the core of the backend system. Key responsibilities:

##### Startup Phase
```python
@app.on_event("startup")
def load_models():
```
- Loads all ML models on server startup
- Initializes ResNet50, MobileNetV2, and YOLO models
- Loads class mapping JSON files
- Sets up device (CPU/GPU) for inference
- Logs loading status for debugging

##### Model Objects
```
resnet50_model      # ResNet50 for disease classification
mobilenet_model     # MobileNetV2 for fruit classification
yolo_model          # YOLO for real-time object detection
mobilenet_classes   # JSON mapping fruit class indices to names
disease_classes     # JSON mapping disease class indices to names
```

#### API Endpoints Provided

**1. Fruit Classification Endpoint**
```
POST /api/predict/fruit
- Input: Image file (multipart form)
- Output: {"fruit": "apple", "fruit_confidence": 0.95}
- Purpose: Identifies fruit type
```

**2. Ripeness Estimation Endpoint**
```
POST /api/predict/ripeness
- Input: Image file + optional fruit name
- Output: {
    "ripeness_score": 75.5,
    "estimated_days_left": 2.0,
    "hsv_ripe_prob": 0.755,
    "fused_ripe_prob": 0.755,
    "label": "ripe"
  }
- Purpose: Estimates fruit ripeness
```

**3. Disease Detection Endpoint**
```
POST /api/predict/disease
- Input: Image file + optional fruit name
- Output: {"detections": [{"label": "Healthy", "confidence": 1.0}]}
- Purpose: Classifies fruit diseases from full image
```

**4. YOLO Live Detection Endpoint** (New)
```
POST /api/predict/yolo
- Input: Image file + confidence threshold (form data)
- Output: {
    "detections": [
      {
        "label": "disease_name",
        "confidence": 0.92,
        "bbox": [x1, y1, x2, y2]
      }
    ],
    "model_loaded": true
  }
- Purpose: Real-time detection with bounding boxes for live webcam
```

**5. Health Check Endpoint**
```
GET /api/ping
- Output: {
    "status": "ok",
    "resnet50_loaded": true,
    "mobilenet_loaded": true
  }
- Purpose: Verify backend is running and models are loaded
```

#### Model Loading Strategy

**ResNet50 (Disease Classifier)**
- Loads from: `backend/models/best_resnet50_final.pth`
- Framework: PyTorch
- Purpose: Classifies disease on fruit
- Input size: 224x224 RGB images
- Output: Class probabilities for diseases

**MobileNetV2 (Fruit Classifier)**
- Loads from: `backend/models/mobilenetv2_fruit_model.h5`
- Framework: TensorFlow/Keras
- Purpose: Identifies fruit type (Apple, Banana, Mango)
- Input size: 224x224 RGB images
- Output: Class probabilities for fruits

**YOLO (Object Detection)**
- Loads from: `backend/models/best-yolotrainedmodel.pt`
- Framework: PyTorch + Ultralytics
- Purpose: Detects disease instances with bounding boxes
- Input size: 640x640 (automatically resized)
- Output: Bounding boxes with class labels and confidence

#### Image Processing Pipeline

1. **Receive**: Image uploaded as multipart form data
2. **Read**: Convert to PIL Image in RGB format
3. **Resize**: Resize to model input size (224x224 or 640x640)
4. **Normalize**: Apply model-specific preprocessing
5. **Infer**: Run through deep learning model
6. **Post-Process**: Extract predictions and confidence scores
7. **Return**: Format as JSON response

#### Ripeness Estimation Algorithm (HSV-Based)

For each fruit type, color ranges are analyzed:

**Banana**
- Green (H: 40-90) = Unripe
- Yellow (H: 15-40) = Ripe
- Brown spots (low V values) = Very Ripe
- Formula considers green coverage, yellow dominance, and spot detection

**Mango**
- Green (H: 40-90, high S) = Unripe
- Yellow-Orange (H: 20-35) = Ripe
- Red-Orange (H: 0-20) = Very Ripe
- Formula blends color transitions

**Apple**
- Light Red/Green = Unripe
- Deep Red (high S, high V) = Ripe
- Dark Red (high S, low V) = Very Ripe

Result: Ripeness score 0-100, estimated days to ripen (0-7)

### 5.2 Frontend Components

#### Root Component: `src/App.jsx`

Manages:
- Overall application state
- Page navigation (Home, Results, History)
- Result sharing between pages
- Header with logo and navigation

```
Structure:
┌─ App.jsx
   ├─ Header (Logo, Navigation, History Button)
   ├─ Home Page (Upload/Webcam Interface)
   ├─ Results Page (Analysis Display)
   └─ History Page (Past Analyses)
```

#### Page Components

**Home.jsx** - Main Analysis Interface
- State: `mode` (upload/webcam/live), `file`, `imagePreview`, `loading`
- Features:
  - Toggle between Upload and Webcam modes
  - Live YOLO detection mode
  - File upload with drag-and-drop
  - Webcam capture functionality
- Flow:
  1. User selects input mode (upload/webcam/live)
  2. Selects or captures image
  3. Clicks analyze button
  4. Calls backend API endpoints sequentially
  5. Receives results and passes to Results page

**Results.jsx** - Results Display
- Displays:
  - Identified fruit type with confidence
  - Ripeness gauge (circular visual)
  - Ripeness percentage and days to ripen
  - Disease detection results
  - Disease canvas visualization
- Updates when result prop changes
- Shows empty state if no analysis yet

**History.jsx** - Analysis History
- Displays previous analyses
- Shows timestamps for each analysis
- Allows review of past results

#### Component Library

**WebCamCapture.jsx** - Webcam Interface
- Features:
  - Real-time webcam video stream
  - Canvas overlay for YOLO detections
  - Capture & Analyze button (standard mode)
  - Start/Stop Live buttons (live mode)
  - Floating detections panel showing:
    - Model load status
    - Detected objects with confidence
- Live Mode Process:
  1. User clicks "Start Live"
  2. Component captures frames every 600ms
  3. Sends frames to `/api/predict/yolo`
  4. Receives detections with bounding boxes
  5. Draws boxes and labels on canvas overlay
  6. Updates floating panel with detections
  7. User clicks "Stop Live" to end

**FileUpload.jsx** - File Input
- Features:
  - Drag-and-drop area
  - Click-to-browse file selection
  - File type validation (image only)
  - File name display
  - Analyze button

**DiseaseBoxesCanvas.jsx** - Disease Visualization
- Displays uploaded image
- Overlays disease predictions
- Shows labels and confidence scores
- Handles image resizing to fit canvas

**Gauge.jsx** - Ripeness Meter
- Circular progress bar
- Shows ripeness percentage (0-100)
- Color-coded (red to green gradient typically)

**Loader.jsx** - Loading Indicator
- Animated spinner component
- Shows during API calls

#### API Service: `src/services/api.js`

Helper functions for backend communication:

```javascript
predictFruit(file)           // Calls /api/predict/fruit
predictRipeness(file, fruit) // Calls /api/predict/ripeness
predictDisease(file, fruit)  // Calls /api/predict/disease
predictYOLO(file, conf)      // Calls /api/predict/yolo
```

Each function:
1. Creates FormData object
2. Appends file and parameters
3. Makes axios POST request
4. Returns parsed JSON response

---

## 6. Machine Learning Models

### Model Overview Table

| Model | Task | Framework | Input Size | Classes | Location |
|-------|------|-----------|-----------|---------|----------|
| MobileNetV2 | Fruit Classification | TensorFlow/Keras | 224x224 | 3 (Apple, Banana, Mango) | `best_resnet50_final.pth` |
| ResNet50 | Disease Classification | PyTorch | 224x224 | 4 (Disease types) | `mobilenetv2_fruit_model.h5` |
| YOLO | Object Detection | PyTorch (Ultralytics) | 640x640 | Variable | `best-yolotrainedmodel.pt` |

### Model Details

**MobileNetV2 (Fruit Classifier)**
- Pre-trained on ImageNet, fine-tuned on fruit dataset
- Lightweight architecture (suitable for edge devices)
- Fast inference (~50-100ms)
- Classes: Apple, Banana, Mango
- Outputs probability distribution over classes

**ResNet50 (Disease Classifier)**
- Residual network with 50 layers
- Fine-tuned for disease classification
- Classes: Disease type 0, 1, 2, 3... + Healthy
- Confidence threshold: 0.5 (below threshold = Healthy)
- Outputs: Class index and confidence

**YOLO (Object Detection)**
- Real-time object detection model
- Detects disease instances with bounding boxes
- Each detection includes:
  - Bounding box coordinates (x1, y1, x2, y2)
  - Class label (disease name)
  - Confidence score (0-1)
- Can process multiple objects in single image
- Used for live webcam streaming

### Class Mappings

**mobilenet_classes.json**
```json
[
  "apple fruit",
  "banana fruit",
  "mango fruit"
]
```

**disease_classes.json**
```json
[
  "disease_type_1",
  "disease_type_2",
  "disease_type_3",
  "disease_type_4"
]
```

---

## 7. API Endpoints Reference

### Base URL
Development: `http://localhost:8000/api`

### Endpoints

#### 1. POST /api/predict/fruit
**Purpose**: Classify fruit type from image

**Request**
```bash
curl -F "file=@image.jpg" http://localhost:8000/api/predict/fruit
```

**Response**
```json
{
  "fruit": "apple",
  "fruit_confidence": 0.9543
}
```

**Error Response** (if MobileNet not loaded)
```json
{
  "fruit": "unknown",
  "fruit_confidence": null
}
```

---

#### 2. POST /api/predict/ripeness
**Purpose**: Estimate fruit ripeness

**Request**
```bash
curl -F "file=@image.jpg" -F "fruit=banana" http://localhost:8000/api/predict/ripeness
```

**Response**
```json
{
  "ripeness_score": 72.5,
  "estimated_days_left": 3.0,
  "clip_ripe_prob": null,
  "hsv_ripe_prob": 0.725,
  "fused_ripe_prob": 0.725,
  "label": "ripe"
}
```

---

#### 3. POST /api/predict/disease
**Purpose**: Detect disease from single fruit image

**Request**
```bash
curl -F "file=@image.jpg" -F "fruit=apple" http://localhost:8000/api/predict/disease
```

**Response**
```json
{
  "detections": [
    {
      "label": "Healthy",
      "confidence": 1.0
    }
  ]
}
```

**With Disease**
```json
{
  "detections": [
    {
      "label": "Early Blight",
      "confidence": 0.89
    }
  ]
}
```

---

#### 4. POST /api/predict/yolo
**Purpose**: Real-time YOLO detection with bounding boxes

**Request**
```bash
curl -F "file=@frame.jpg" -F "conf=0.25" http://localhost:8000/api/predict/yolo
```

**Response**
```json
{
  "detections": [
    {
      "label": "Disease_A",
      "confidence": 0.92,
      "bbox": [120.5, 150.2, 340.8, 420.1]
    },
    {
      "label": "Disease_B",
      "confidence": 0.78,
      "bbox": [350.0, 180.0, 500.0, 400.0]
    }
  ],
  "model_loaded": true
}
```

**Bounding Box Format**: [x1, y1, x2, y2]
- x1, y1: Top-left corner coordinates
- x2, y2: Bottom-right corner coordinates
- Coordinates are in pixel values relative to input image

**When Model Not Loaded**
```json
{
  "detections": [],
  "model_loaded": false
}
```

---

#### 5. GET /api/ping
**Purpose**: Health check and model status

**Request**
```bash
curl http://localhost:8000/api/ping
```

**Response**
```json
{
  "status": "ok",
  "resnet50_loaded": true,
  "mobilenet_loaded": true
}
```

---

## 8. Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- pip (Python package manager)
- npm (Node package manager)
- Git (optional, for cloning)

### Backend Setup

**Step 1: Navigate to backend directory**
```powershell
cd backend
```

**Step 2: Create Python virtual environment**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

**Step 3: Install Python dependencies**
```powershell
pip install -r requirements.txt
```

Additional dependencies (if not in requirements.txt):
```powershell
pip install ultralytics opencv-python
```

**Step 4: Verify model files exist**
Check that these files are in `backend/models/`:
- `best_resnet50_final.pth`
- `mobilenetv2_fruit_model.h5`
- `best-yolotrainedmodel.pt`
- `disease_classes.json`
- `mobilenet_classes.json`

### Frontend Setup

**Step 1: Navigate to frontend directory**
```powershell
cd frontend\fruitai
```

**Step 2: Install Node dependencies**
```powershell
npm install
```

**Step 3: Verify setup**
```powershell
npm list
```

---

## 9. Running the Application

### Starting Backend Server

From `backend/` directory with virtual environment activated:

**Development mode (with auto-reload)**
```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Production mode**
```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Expected Output**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Loaded ResNet50 model from backend/models/best_resnet50_final.pth with 4 classes
INFO:     Loaded MobileNet model from backend/models/mobilenetv2_fruit_model.h5
INFO:     Loaded YOLO (ultralytics) model from backend/models/best-yolotrainedmodel.pt, device=cuda
```

### Starting Frontend Development Server

From `frontend/fruitai/` directory:

```powershell
npm run dev
```

**Expected Output**
```
VITE v7.2.4  ready in 254 ms

➜  Local:   http://localhost:5173/
➜  press h to show help
```

### Accessing the Application

Open your browser and navigate to:
```
http://localhost:5173
```

---

## 10. Data Flow

### Complete Flow: Upload Image Analysis

```
1. USER INTERACTION
   └─ User opens http://localhost:5173
   └─ Navigates to Home page
   └─ Switches to Upload mode
   └─ Selects image file

2. FRONTEND
   └─ FileUpload.jsx triggers file selection
   └─ Displays selected file name
   └─ User clicks "Analyze Image"
   └─ Home.jsx calls handleAnalyze()

3. API CALLS (Sequential)
   
   Call 1: Fruit Classification
   ├─ POST /api/predict/fruit
   ├─ File: image.jpg
   ├─ Response: {"fruit": "apple", "fruit_confidence": 0.95}
   └─ Frontend stores: fruit = "apple"
   
   Call 2: Ripeness Estimation
   ├─ POST /api/predict/ripeness
   ├─ File: image.jpg, Param: fruit="apple"
   ├─ Response: {"ripeness_score": 75, ...}
   └─ Frontend stores: ripeness data
   
   Call 3: Disease Detection
   ├─ POST /api/predict/disease
   ├─ File: image.jpg, Param: fruit="apple"
   ├─ Response: {"detections": [{"label": "Healthy", "confidence": 1.0}]}
   └─ Frontend stores: diseases array

4. RESULT AGGREGATION
   └─ Home.jsx combines all results into single object:
      {
        fruit: "apple",
        fruit_confidence: 0.95,
        ripeness: {...},
        diseases: [...],
        imagePreview: "data:image/jpeg;base64,..."
      }

5. RESULT DISPLAY
   └─ Frontend calls setResult() with combined data
   └─ Results.jsx receives updated prop
   └─ Renders:
      ├─ Fruit name and confidence
      ├─ Gauge with ripeness score
      ├─ Disease detection results
      └─ Image preview with overlays

6. HISTORY UPDATE
   └─ Result added to history with timestamp
   └─ Can be accessed via History page
```

### Complete Flow: Live Webcam Detection

```
1. USER INTERACTION
   └─ User opens http://localhost:5173
   └─ Navigates to Home page
   └─ Clicks Live (play icon)
   └─ Clicks "Start Live"

2. FRONTEND
   └─ WebCamCapture.jsx activates live mode
   └─ Starts interval timer (every 600ms)

3. FRAME CAPTURE LOOP (Continuous)
   
   For each frame (every ~600ms):
   ├─ Capture screenshot from webcam video element
   ├─ Convert to blob format
   ├─ Create File object
   └─ POST /api/predict/yolo
   
4. BACKEND PROCESSING
   ├─ Receive image frame
   ├─ Run YOLO inference
   ├─ Extract detections with bounding boxes
   └─ Return JSON: {detections: [...], model_loaded: true}

5. FRONTEND RENDERING
   ├─ Receive YOLO response
   ├─ Update detections state
   ├─ Canvas overlay draws:
   │  ├─ Bounding boxes (green lines)
   │  ├─ Labels with confidence
   │  └─ Semi-transparent rectangles for readability
   └─ Floating panel updates:
      ├─ Model status ("Model OK" or "Model missing")
      ├─ List of detected objects
      └─ Confidence percentages

6. USER STOPS
   └─ User clicks "Stop Live"
   └─ Interval timer cleared
   └─ Webcam video continues (can capture & analyze)
```

---

## 11. User Guide

### First-Time Use

1. **Start the Application**
   - Run backend server
   - Run frontend dev server
   - Open browser to http://localhost:5173

2. **Upload Image Analysis**
   - In Home page, ensure you're in "Upload" mode (folder icon)
   - Click upload area or drag image
   - Click "Analyze Image"
   - Wait for processing (5-15 seconds depending on hardware)
   - View results in Results panel

3. **Webcam Snapshot**
   - Click webcam icon to switch to Webcam mode
   - Click "Capture & Analyze"
   - Wait for processing
   - View results in Results panel

4. **Live Webcam Detection**
   - Click play icon to switch to Live mode
   - Grant camera permissions when prompted
   - Click "Start Live"
   - Watch real-time detection with bounding boxes
   - Click "Stop Live" to end streaming

### Interpreting Results

**Fruit Classification**
- Shows identified fruit name
- Confidence percentage (0-100%)
- Higher confidence = more reliable

**Ripeness Estimation**
- Ripeness Score: 0-100% (0 = unripe, 100 = very ripe)
- Estimated Days to Ripen: Approximate days until optimal ripeness
- Label: "unripe", "ripe", or "very ripe"
- Gauge visual: Color indicates ripeness level

**Disease Detection**
- Disease Name: Type of disease detected (or "Healthy" if none)
- Confidence: How confident the model is about the detection
- Confidence Threshold: Below 50% confidence marked as "Healthy"
- Bounding Boxes: Red boxes on image showing disease location

### Viewing History

- Click "History" button in top-right
- Shows all previous analyses with timestamps
- Can review past results
- Click "Back to Analysis" to return to main interface

---

## 12. Troubleshooting

### Issue: Backend fails to start

**Symptom**: Error when running uvicorn

**Solutions**:
1. Verify Python 3.8+ installed: `python --version`
2. Check virtual environment activated
3. Reinstall dependencies: `pip install -r requirements.txt`
4. Check port 8000 is available: `netstat -ano | findstr :8000`
5. Try different port: `uvicorn app:app --port 8001`

### Issue: Models not loading

**Symptom**: Backend logs show "Failed to load ... model"

**Solutions**:
1. Verify model files in `backend/models/`:
   - `best_resnet50_final.pth`
   - `mobilenetv2_fruit_model.h5`
   - `best-yolotrainedmodel.pt`
2. Check file sizes are reasonable (not corrupted):
   - ResNet50: ~100MB
   - MobileNetV2: ~50-100MB
   - YOLO: ~100-200MB
3. Verify model file paths correct
4. Check disk space available
5. Try redownloading models if corrupted

### Issue: Frontend won't connect to backend

**Symptom**: "Request failed" errors in browser console

**Solutions**:
1. Verify backend running: `curl http://localhost:8000/api/ping`
2. Check CORS configuration in backend/app.py
3. Verify frontend API URL in `src/services/api.js`:
   ```javascript
   const API_BASE = 'http://localhost:8000/api'
   ```
4. Check firewall settings
5. Ensure ports match (backend: 8000, frontend: 5173)

### Issue: Live YOLO detection shows no boxes

**Symptom**: Live mode runs but no bounding boxes appear

**Solutions**:
1. Check backend logs: "Loaded YOLO ..." message
2. Ensure `best-yolotrainedmodel.pt` exists and is valid
3. Check floating detections panel shows "Model OK" or "Model missing"
4. If "Model missing", YOLO didn't load - check model file
5. Try stopping and restarting Live mode
6. Ensure good lighting for detection to work

### Issue: Slow inference

**Symptom**: Analysis takes more than 20 seconds

**Solutions**:
1. Enable GPU if available (check backend logs for device=cuda)
2. If on CPU, this is normal - may take 10-30 seconds
3. Reduce image resolution before uploading
4. Check system resources (RAM, CPU) not maxed out
5. YOLO live mode may be slower - expected with continuous frames

### Issue: Webcam permission denied

**Symptom**: "Permission denied" when accessing webcam

**Solutions**:
1. Reload page and allow camera permission
2. Check browser permissions:
   - Chrome: Settings → Privacy → Camera
   - Firefox: Preferences → Privacy → Permissions
3. Try different browser
4. Ensure https on production (browsers require https for camera)
5. Restart browser

### Issue: Image upload fails

**Symptom**: File upload gives error

**Solutions**:
1. Ensure file is an image (JPG, PNG, GIF)
2. Check file size under 10MB
3. Try different image format
4. Check browser console for detailed error
5. Verify backend is running

---

## 13. Future Enhancements

### Potential Improvements

1. **Model Enhancements**
   - Train disease classifier on more disease types
   - Implement multi-disease detection on single fruit
   - Add ripeness regression model for better accuracy
   - Create ensemble model combining all classifiers

2. **Features**
   - Batch image upload and analysis
   - Export analysis results as PDF
   - Compare ripeness trends over time
   - Notification system for optimal harvest time
   - Mobile app version
   - Cloud deployment

3. **Performance**
   - Model quantization for faster inference
   - Caching layer for repeated images
   - Batch inference optimization
   - WebSocket for real-time updates instead of polling

4. **UI/UX**
   - Dark/light theme toggle
   - Multi-language support
   - Detailed disease information pop-ups
   - 3D visualization of ripeness stages
   - Video upload support

5. **Analytics**
   - Dashboard showing aggregate statistics
   - Disease prevalence reports
   - Ripeness distribution charts
   - Detection accuracy metrics

6. **Integration**
   - API documentation (Swagger/OpenAPI)
   - Authentication system
   - Database for result persistence
   - Integration with agricultural management systems
   - IoT sensor data integration

---

## Glossary

- **API**: Application Programming Interface - methods for frontend to communicate with backend
- **Bounding Box**: Rectangle coordinates defining location of detected object
- **Confidence Score**: Probability that detection/classification is correct (0-1 scale)
- **CORS**: Cross-Origin Resource Sharing - allows frontend to access backend API
- **FastAPI**: Modern Python web framework for building REST APIs
- **YOLO**: "You Only Look Once" - real-time object detection algorithm
- **ResNet**: Residual Neural Network - deep learning architecture
- **MobileNetV2**: Lightweight deep learning model for mobile/edge devices
- **HSV**: Hue, Saturation, Value - color space for analyzing fruit color
- **Vite**: Modern frontend build tool with fast development server
- **React**: JavaScript library for building user interfaces
- **Multipart Form**: HTTP method for sending files and data

---

## Support and Contact

For issues or questions:
1. Check troubleshooting section
2. Review backend logs: `uvicorn` console output
3. Review frontend logs: Browser Developer Tools (F12 → Console)
4. Verify all setup steps completed correctly

---

## License

Please refer to project license file if present.

---

**Document Version**: 1.0
**Last Updated**: November 25, 2025
**Application Version**: 1.0

