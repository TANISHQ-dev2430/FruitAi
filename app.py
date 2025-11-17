# app.py — Updated: improved auto-detection (MobileNet index-mapping + multi-crop + CLIP fusion)
# UI/layout retained exactly as before; only detection internals modified.

import streamlit as st
import warnings, os, time, json
warnings.filterwarnings("ignore", message=".*use_column_width.*")
from PIL import Image, ImageDraw, ImageFont
import numpy as np, cv2, torch
import plotly.graph_objects as go
import altair as alt
import pandas as pd

# ----------------- CONFIG -----------------
YOLO_PATH = "models/best-yolotrainedmodel.pt"   # your YOLO model
MOBILENET_PATH = "models/mobilenetv2_fruit_model.h5"  # optional MobileNet model (Keras .h5)
MOBILENET_CLASSES_JSON = "models/mobilenet_classes.json"  # optional mapping list of class names
FALLBACK_FRUITS = ["banana", "mango", "apple"]

# Cure tips mapping (customize)
CURE_TIPS = {
    "anthracnose": "Remove infected tissue; apply copper fungicide.",
    "powdery_mildew": "Improve airflow; apply sulfur treatment.",
    "healthy": "No disease detected."
}

# ----------------- MODEL LOADERS (cached) -----------------
@st.cache_resource
def load_yolo(path=YOLO_PATH):
    if not os.path.exists(path):
        return None
    try:
        from ultralytics import YOLO
        return YOLO(path)
    except Exception as e:
        st.error("YOLO load error: " + str(e))
        return None

@st.cache_resource
def load_mobilenet(path=MOBILENET_PATH):
    if not os.path.exists(path):
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        # attempt to read classes mapping if present
        cls_map = None
        if os.path.exists(MOBILENET_CLASSES_JSON):
            try:
                with open(MOBILENET_CLASSES_JSON, "r", encoding="utf-8") as f:
                    cls_map = json.load(f)
            except Exception:
                cls_map = None
        return {"model": model, "classes": cls_map}
    except Exception as e:
        st.warning("MobileNet not loaded: " + str(e))
        return None

@st.cache_resource
def load_clip():
    try:
        from transformers import CLIPProcessor, CLIPModel
    except Exception as e:
        st.error("Install 'transformers' to use CLIP ripeness fallback.")
        raise e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return proc, model, device

# ----------------- UTILITIES -----------------
def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def prepare_frame_for_yolo(frame_bgr):
    return np.ascontiguousarray(frame_bgr, dtype=np.uint8)

def measure_text(draw, text, font):
    bbox = draw.textbbox((0,0), text, font=font)
    return bbox[2]-bbox[0], bbox[3]-bbox[1]

def draw_boxes_pil(pil_img, detections, box_color=(255,0,0)):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        label = f"{d['label']} {d['confidence']:.2f}"
        w,h = measure_text(draw, label, font)
        draw.rectangle([x1,y1,x2,y2], outline=box_color, width=3)
        draw.rectangle([x1, y1-h-6, x1+w+6, y1], fill=box_color)
        draw.text((x1+3, y1-h-4), label, fill=(255,255,255), font=font)
    return img

# ---------------- YOLO / RIPENESS ----------------
def detect_diseases_yolo(yolo_model, frame_bgr, conf=0.35):
    if yolo_model is None:
        return []
    try:
        arr = prepare_frame_for_yolo(frame_bgr)
        res = yolo_model(arr, conf=conf)
    except Exception as e:
        st.warning("YOLO runtime error: " + str(e))
        return []
    dets = []
    names = res[0].names
    for box in res[0].boxes:
        cls = int(box.cls)
        confv = float(box.conf)
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        lbl = names[cls] if isinstance(names, (list,dict)) and cls in names else str(cls)
        dets.append({"label": lbl, "confidence": round(confv,3), "bbox":[x1,y1,x2,y2]})
    return dets

# CLIP and HSV helpers
def clip_similarity_batch(proc, model, device, image_pil, prompts):
    inputs = proc(text=prompts, images=image_pil, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
        img_emb = out.image_embeds
        txt_emb = out.text_embeds
        img_emb = img_emb / img_emb.norm(p=2,dim=-1,keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(p=2,dim=-1,keepdim=True)
        sims = (img_emb @ txt_emb.T).cpu().numpy()[0]
    return float(sims.mean())

def hsv_ripeness_from_bgr(bgr_img, fruit):
    img = cv2.resize(bgr_img, (256,256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h,s,v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    fruit = fruit.lower()
    if fruit == "banana":
        yellow = ((h>=15)&(h<=35)&(s>70)&(v>60)).mean()
        green  = ((h>=40)&(h<=85)&(s>40)&(v>40)).mean()
        return float(np.clip(0.8*yellow + 0.2*(1-green),0,1))
    if fruit == "mango":
        orange = ((h>=10)&(h<=25)&(s>80)&(v>60)).mean()
        yellow = ((h>=18)&(h<=40)&(s>55)&(v>60)).mean()
        green  = ((h>=35)&(h<=95)&(s>35)&(v>40)).mean()
        hsv_prob = float(np.clip(0.55*orange + 0.35*yellow + 0.1*(1-green),0,1))
        golden_mask = ((h>=20)&(h<=40)&(s>45)&(v>72))
        pct_golden = golden_mask.mean()
        return float(np.clip(hsv_prob + 0.4*pct_golden,0,1))
    if fruit == "apple":
        red = (((h<=10)|(h>=160))&(s>60)&(v>50)).mean()
        return float(np.clip(red,0,1))
    return 0.5

def fused_ripeness(pil_img, bgr_img, fruit, clip_proc, clip_model, clip_device):
    prompts_map = {
        "banana": (["yellow ripe banana","ripe banana"], ["green unripe banana"]),
        "mango": (["ripe mango yellow/orange","sweet ripe mango"], ["green unripe mango"]),
        "apple": (["red ripe apple","ripe apple"], ["green unripe apple"])
    }
    ripe_prompts, unripe_prompts = prompts_map.get(fruit, prompts_map["banana"])
    try:
        clip_r = clip_similarity_batch(clip_proc, clip_model, clip_device, pil_img, ripe_prompts)
        clip_u = clip_similarity_batch(clip_proc, clip_model, clip_device, pil_img, unripe_prompts)
    except Exception:
        clip_r, clip_u = 0.5, 0.5
    sims = np.array([clip_r, clip_u], dtype=np.float32)
    ex = np.exp(sims - sims.max()); probs = ex/ex.sum()
    clip_ripe_prob = float(probs[0])
    hsv_prob = hsv_ripeness_from_bgr(bgr_img, fruit)
    alpha = 0.55 if fruit == "mango" else 0.75
    fused = float(np.clip(alpha * clip_ripe_prob + (1 - alpha) * hsv_prob, 0, 1))
    ripeness_score = round(fused * 100, 1)
    estimated_days = round(max(0.0, (1.0 - fused) * (3.0 if fruit == "mango" else 7.0)), 1)
    ok_margin = 0.06 if fruit == "mango" else 0.12
    ok = bool(abs(fused - 0.5) >= ok_margin)
    return {
        "clip_ripe_prob": round(clip_ripe_prob, 3),
        "hsv_ripe_prob": round(hsv_prob, 3),
        "fused_ripe_prob": round(fused, 3),
        "ripeness_score": ripeness_score,
        "estimated_days_left": estimated_days,
        "ok": ok,
        "label": ("ripe" if fused >= 0.5 else "unripe")
    }

# ------------------ NEW: Improved auto detection logic ------------------
def multi_crop_images(pil_img):
    """Return list of PIL crops/resizes for more robust predictions:
       - whole resized to 224
       - center crop 0.8 -> resized
       - center crop 0.6 -> resized
    """
    imgs = []
    w,h = pil_img.size
    def resize(p):
        return p.resize((224,224))
    imgs.append(resize(pil_img))
    # center 80%
    c1 = pil_img.crop((int(0.1*w), int(0.1*h), int(0.9*w), int(0.9*h)))
    imgs.append(resize(c1))
    # center 60%
    c2 = pil_img.crop((int(0.2*w), int(0.2*h), int(0.8*w), int(0.8*h)))
    imgs.append(resize(c2))
    return imgs

def mobilenet_predict_ensemble(mobilenet_resource, pil_img):
    """
    Given mobilenet_resource = {'model':tf.keras.Model, 'classes': optional list}
    Returns: dict of aggregated probabilities over target fruits (banana,mango,apple)
             and best fruit/confidence.
    """
    try:
        import tensorflow as tf
        model = mobilenet_resource['model']
        cls_map = mobilenet_resource.get('classes', None)
    except Exception:
        return None

    # Prepare crops
    crops = multi_crop_images(pil_img)
    preds_accum = None

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    for crop in crops:
        arr = np.array(crop).astype("float32")
        arr = preprocess_input(arr)
        batch = np.expand_dims(arr, 0)
        try:
            out = model.predict(batch)[0]  # softmax vector
        except Exception:
            # if model expects different input, abort
            return None
        if preds_accum is None:
            preds_accum = out
        else:
            preds_accum += out
    preds_accum /= len(crops)  # average

    # If class mapping present, map indices to fruits by substring match
    if cls_map:
        # build list-of-indices for each target fruit
        fruit_indices = {f: [] for f in FALLBACK_FRUITS}
        for idx, name in enumerate(cls_map):
            low = name.lower()
            for f in FALLBACK_FRUITS:
                if f in low:
                    fruit_indices[f].append(idx)
        # sum probs for each fruit
        fruit_probs = {}
        for f in FALLBACK_FRUITS:
            if fruit_indices[f]:
                fruit_probs[f] = float(np.sum([preds_accum[i] for i in fruit_indices[f]]))
            else:
                fruit_probs[f] = 0.0
        # normalize
        s = sum(fruit_probs.values()) or 1.0
        for k in fruit_probs:
            fruit_probs[k] = fruit_probs[k] / s
        # choose best
        best_f = max(fruit_probs, key=fruit_probs.get)
        return {"probs": fruit_probs, "best": best_f, "confidence": fruit_probs[best_f]}
    else:
        # If no mapping, we fallback to a heuristic: try to map top-k indices to fruit strings via CLIP later
        # But at least return top3 indices & their probs
        top_idx = int(np.argmax(preds_accum))
        top_conf = float(preds_accum[top_idx])
        return {"probs_vec": preds_accum.tolist(), "best_index": top_idx, "confidence": top_conf}

def auto_detect_fruit(pil_img, mobilenet_resource, clip_proc, clip_model, clip_device):
    """
    Robust auto detect:
     1) If mobilenet_resource exists and has classes mapping, sum probs per fruit -> accept if high enough.
     2) Else try mobilenet ensemble (multi-crop) and if confident (>=0.55) accept mapped result if mapping exists.
     3) Otherwise use CLIP fallback (multi-prompt)
     4) If both available and both give signals, fuse them (weighted) for higher reliability.
    Returns: (label, confidence)
    """
    mobilenet_result = None
    if mobilenet_resource is not None:
        mobilenet_result = mobilenet_predict_ensemble(mobilenet_resource, pil_img)

    clip_scores = {}
    # CLIP prompts used for classification (multi prompts)
    prompts_per_fruit = {
        "banana": ["banana", "ripe banana", "green banana", "banana close up"],
        "mango": ["mango", "ripe mango", "green mango", "mango close up"],
        "apple": ["apple", "ripe apple", "green apple", "apple close up"]
    }

    # Run CLIP similarity only if needed or to fuse
    try_clip = True
    # If mobilenet_result has strong probs mapped to fruits, we may skip heavy CLIP step
    if mobilenet_result and "probs" in mobilenet_result:
        if mobilenet_result["confidence"] >= 0.65:
            # confident mapping present; return directly
            return mobilenet_result["best"], float(mobilenet_result["confidence"])
        # else we'll compute clip_scores to fuse

    # compute CLIP scores (average similarity per fruit using prompts)
    try:
        for f,prompts in prompts_per_fruit.items():
            s = clip_similarity_batch(clip_proc, clip_model, clip_device, pil_img, prompts)
            clip_scores[f] = s
    except Exception:
        clip_scores = {f: 0.0 for f in FALLBACK_FRUITS}

    # Normalize CLIP similarities to probabilities
    sims = np.array([clip_scores[f] for f in FALLBACK_FRUITS], dtype=np.float32)
    ex = np.exp(sims - sims.max()); clip_probs = (ex / ex.sum()).tolist()
    clip_prob_map = {f: clip_probs[i] for i,f in enumerate(FALLBACK_FRUITS)}

    # If Mobilenet provided mapped probs, fuse weighted; otherwise use CLIP result
    if mobilenet_result and "probs" in mobilenet_result:
        # fusion weights: give more weight to mobilenet when it has mapping, else to CLIP
        w_m = 0.6
        w_c = 0.4
        fused = {}
        for f in FALLBACK_FRUITS:
            fused[f] = w_m * mobilenet_result["probs"].get(f, 0.0) + w_c * clip_prob_map.get(f, 0.0)
        # normalize
        s = sum(fused.values()) or 1.0
        for k in fused: fused[k] = fused[k]/s
        best = max(fused, key=fused.get)
        return best, float(fused[best])
    elif mobilenet_result and "probs_vec" in mobilenet_result:
        # no classes mapping, attempt to infer mapping by using CLIP to map top indices -> fruit
        # Approach: For top-3 indices, synthesize a label by comparing class-name candidates via CLIP.
        # But without class strings we can't map. So fall back to CLIP only.
        best_clip = max(clip_prob_map, key=clip_prob_map.get)
        return best_clip, float(clip_prob_map[best_clip])
    else:
        # mobilenet not available -> use CLIP
        best_clip = max(clip_prob_map, key=clip_prob_map.get)
        return best_clip, float(clip_prob_map[best_clip])

# -------------------- APP LAYOUT & UI (unchanged) --------------------
st.set_page_config(layout="wide", page_title="Fruit AI — Dashboard")
st.markdown("""
<style>
.card {
  background-color: #0f1720;
  border-radius: 12px;
  padding: 18px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.35);
  color: #e6eef8;
}
.small-muted { color:#9aa5b1; font-size:0.95rem; }
.badge { display:inline-block; padding:6px 10px; border-radius:999px; background:#2b6b4a; color:white; margin-right:6px; margin-bottom:6px; }
.kv { color:#9aa5b1; font-size:0.95rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='margin-bottom:0.1rem'>🍓 Fruit AI — Ripeness & Disease Inspector</h1>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Upload an image / use webcam • results and analytics update live</div>", unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Mode", ["Upload / Snapshot", "Real-time Webcam"])
    fruit_choice = st.selectbox("Fruit selection", ["auto", "banana", "mango", "apple"])
    conf = st.slider("YOLO confidence", 0.05, 0.9, 0.35, 0.05)
    show_boxes = st.checkbox("Show bounding boxes", True)
    st.markdown("---")
    st.header("Diagnostics")
    st.write("YOLO model:", YOLO_PATH, "exists?", os.path.exists(YOLO_PATH))
    st.write("MobileNet:", MOBILENET_PATH, "exists?", os.path.exists(MOBILENET_PATH))
    st.caption("Place models in the models/ folder and restart app if changed.")

with st.spinner("Loading models (this may take a moment)..."):
    yolo_model = load_yolo(YOLO_PATH)
    mobilenet_res = load_mobilenet(MOBILENET_PATH)  # returns None or {'model':..., 'classes':[...] }
    clip_proc, clip_model, clip_device = load_clip()

if "history" not in st.session_state: st.session_state.history = []
if "last_result" not in st.session_state: st.session_state.last_result = None

col_left, col_right = st.columns([1.4, 0.9], gap="large")

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Input")
    if mode == "Upload / Snapshot":
        uploaded = st.file_uploader("Upload image (.jpg/.png)", type=["jpg","jpeg","png"])
        if st.button("Analyze image"):
            if not uploaded:
                st.warning("Upload an image first.")
            else:
                pil = Image.open(uploaded).convert("RGB")
                bgr = pil_to_bgr(pil)

                # detect fruit (use new auto_detect_fruit)
                if fruit_choice == "auto":
                    detected_fruit, fruit_conf = auto_detect_fruit(pil, mobilenet_res, clip_proc, clip_model, clip_device)
                else:
                    detected_fruit, fruit_conf = fruit_choice, None

                rip = fused_ripeness(pil, bgr, detected_fruit, clip_proc, clip_model, clip_device)
                dets = detect_diseases_yolo(yolo_model, bgr, conf=conf)
                annotated = draw_boxes_pil(pil, dets) if (show_boxes and dets) else pil
                res = {"fruit":detected_fruit, "fruit_confidence":fruit_conf, "ripeness":rip, "diseases":dets}
                st.session_state.last_result = res
                st.session_state.history.append({"time": time.time(), "score": rip["ripeness_score"], "fruit": detected_fruit, "diseases":[d["label"] for d in dets]})
                st.image(annotated, use_container_width=True)
    else:
        st.markdown("### Real-time Webcam")
        if "realtime_running" not in st.session_state: st.session_state.realtime_running = False
        if st.button("Start Webcam"):
            st.session_state.realtime_running = True
        if st.button("Stop Webcam"):
            st.session_state.realtime_running = False

        frame_slot = st.empty()
        if st.session_state.realtime_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam.")
                st.session_state.realtime_running = False
            else:
                fps_t = time.time()
                try:
                    while st.session_state.realtime_running:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        h,w = frame.shape[:2]
                        if w > 900:
                            frame = cv2.resize(frame, (900, int(h*900/w)))
                        arr = prepare_frame_for_yolo(frame)
                        dets = detect_diseases_yolo(yolo_model, arr, conf=conf)
                        pil_frame = bgr_to_pil(frame)
                        display_img = draw_boxes_pil(pil_frame, dets) if (show_boxes and dets) else pil_frame
                        frame_slot.image(display_img, use_container_width=True)
                        fps = 1.0 / max(1e-6, (time.time()-fps_t))
                        fps_t = time.time()
                        st.sidebar.write(f"Realtime — detections: {len(dets)} — FPS: {fps:.1f}")
                        time.sleep(0.03)
                except Exception as e:
                    st.error("Realtime error: " + str(e))
                finally:
                    cap.release()

    if st.button("Capture & analyze current webcam frame"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            st.error("Could not capture frame.")
        else:
            pil = bgr_to_pil(frame)
            bgr = pil_to_bgr(pil)
            if fruit_choice == "auto":
                detected_fruit, fruit_conf = auto_detect_fruit(pil, mobilenet_res, clip_proc, clip_model, clip_device)
            else:
                detected_fruit, fruit_conf = fruit_choice, None
            rip = fused_ripeness(pil, bgr, detected_fruit, clip_proc, clip_model, clip_device)
            dets = detect_diseases_yolo(yolo_model, bgr, conf=conf)
            annotated = draw_boxes_pil(pil, dets) if (show_boxes and dets) else pil
            res = {"fruit":detected_fruit, "fruit_confidence":fruit_conf, "ripeness":rip, "diseases":dets}
            st.session_state.last_result = res
            st.session_state.history.append({"time": time.time(), "score": rip["ripeness_score"], "fruit": detected_fruit, "diseases":[d["label"] for d in dets]})
            st.image(annotated, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Analysis & Details")
    last = st.session_state.last_result
    if last is None:
        st.info("No analysis yet. Upload or capture to see details here.")
    else:
        fruit = last["fruit"].title()
        fconf = last.get("fruit_confidence", None)
        st.markdown(f"### {fruit}  {'— {:.2f}'.format(fconf) if fconf else ''}")
        rip = last["ripeness"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rip["ripeness_score"],
            number={'suffix':' %', 'font':{'size':36}},
            gauge={'axis': {'range':[0,100]},
                   'bar': {'color':"rgba(255,150,50,0.9)"},
                   'steps': [{'range':[0,40], 'color':'#ff6b6b'},
                             {'range':[40,70], 'color':'#ffb703'},
                             {'range':[70,100], 'color':'#90be6d'}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                   }
        ))
        fig.update_layout(height=260, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Ripeness details"):
            st.write(f"- Label: **{rip['label'].upper()}**")
            st.write(f"- Score: **{rip['ripeness_score']}%**")
            st.write(f"- CLIP ripe prob: **{rip['clip_ripe_prob']}**")
            st.write(f"- HSV ripe prob: **{rip['hsv_ripe_prob']}**")
            st.write(f"- Estimated days to ripen: **{rip['estimated_days_left']}**")
            if not rip["ok"]:
                st.warning("Ripeness estimate close to the threshold — try additional angles for better accuracy.")

        st.markdown("**Disease detections**")
        dets = last["diseases"]
        if dets:
            for d in dets:
                lab = d["label"]
                confv = d["confidence"]
                st.markdown(f"<div class='badge'>{lab} • {confv:.2f}</div>", unsafe_allow_html=True)
                tip = CURE_TIPS.get(lab.lower(), "No cure tip available.")
                st.caption(tip)
        else:
            st.markdown("<div style='padding:12px; border-radius:8px; background:#173d27; color:#dff7e0'>No disease detected.</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Full result (JSON)**")
        st.json(last)
        st.download_button("Download last result JSON", json.dumps(last, indent=2), file_name="fruit_ai_result.json")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
st.markdown("## Session Analytics")
hist = st.session_state.history
if hist:
    df = pd.DataFrame(hist)
    df['ts'] = pd.to_datetime(df['time'], unit='s')
    line = alt.Chart(df).mark_line(point=True).encode(x='ts:T', y='score:Q', color='fruit', tooltip=['ts','score','fruit']).interactive()
    st.altair_chart(line, use_container_width=True)
    all_d = [lab for rec in hist for lab in rec['diseases']]
    if all_d:
        df2 = pd.DataFrame({"disease":all_d})
        df2 = df2.groupby("disease").size().reset_index(name='count')
        bar = alt.Chart(df2).mark_bar().encode(x='disease:N', y='count:Q', color='disease', tooltip=['disease','count'])
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("No disease detections recorded yet.")
else:
    st.info("No session samples yet — analyze images to populate analytics.")

st.caption("Tip: For best ripeness results, provide a clear, close-up image of the fruit surface (good lighting).")
