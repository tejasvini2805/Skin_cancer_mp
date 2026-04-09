from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import io
import base64
import logging
from datetime import datetime
from typing import List, Dict, Optional
import json
import os
import random
import numpy as np
import requests
import re
import time
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. EXACT CONFIGURATION FROM YOUR CSV
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "edgefusion_v3_hybrid_champion.pth"

# Extracted directly from df['label'].unique()
CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
NUM_CLASSES = len(CLASS_NAMES)

# Extracted directly from pd.get_dummies() columns
SEX_CATEGORIES = ['female', 'male', 'unknown']
LOC_CATEGORIES = [
    'anterior torso', 'head/neck', 'lateral torso', 'lower extremity', 
    'oral/genital', 'palms/soles', 'posterior torso', 'unknown', 'upper extremity'
]
NUM_META_FEATURES = 1 + len(SEX_CATEGORIES) + len(LOC_CATEGORIES) # 13 features total

# ==============================================================================
# 2. EXACT ARCHITECTURE FROM TRAINING SCRIPT
# ==============================================================================
class EdgeFusionV3Net(nn.Module):
    def __init__(self, num_classes, num_meta_features):
        super(EdgeFusionV3Net, self).__init__()

        # 1. Vision Backbone
        self.vision = timm.create_model('efficientformerv2_s1', pretrained=False, num_classes=0)

        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 224, 224)
            vision_out_features = self.vision(dummy_img).shape[1]

        # 2. Metadata MLP
        self.meta_mlp = nn.Sequential(
            nn.Linear(num_meta_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # 3. SE Gated Fusion
        combined_dim = vision_out_features + 32
        self.se_gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 4),
            nn.ReLU(),
            nn.Linear(combined_dim // 4, combined_dim),
            nn.Sigmoid()
        )

        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, meta):
        v_feat = self.vision(img)
        m_feat = self.meta_mlp(meta)
        fused_raw = torch.cat((v_feat, m_feat), dim=1)
        attention_weights = self.se_gate(fused_raw)
        fused_gated = fused_raw * attention_weights
        return self.classifier(fused_gated)

# ==============================================================================
# 3. GLOBAL PREDICTION HISTORY
# ==============================================================================
prediction_history: List[Dict] = []
verification_codes_phone: Dict[str, str] = {}
MEDICAL_KB: Dict = {}

# Persistent history file
HISTORY_PATH = os.path.join(os.path.dirname(__file__), 'data', 'history.json')

def load_history_from_disk():
    global prediction_history
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r', encoding='utf-8') as hf:
                data = json.load(hf)
                if isinstance(data, list):
                    prediction_history = data
                    logger.info(f"Loaded {len(prediction_history)} history items from disk")
                else:
                    prediction_history = []
        else:
            prediction_history = []
    except Exception as e:
        logger.exception('Failed to load history from disk')
        prediction_history = []

def save_history_to_disk():
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, 'w', encoding='utf-8') as hf:
            json.dump(prediction_history, hf, ensure_ascii=False, indent=2)
        logger.debug(f"Saved {len(prediction_history)} history items to disk")
    except Exception as e:
        logger.exception('Failed to save history to disk')

try:
    KB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'medical_kb.json')
    if os.path.exists(KB_PATH):
        with open(KB_PATH, 'r', encoding='utf-8') as f:
            MEDICAL_KB = json.load(f)
            logger.info(f"Loaded medical KB with {len(MEDICAL_KB)} entries")
    else:
        logger.warning('medical_kb.json not found; continuing without KB')
except Exception as e:
    logger.exception('Failed to load medical KB')
    MEDICAL_KB = {}

# emaill send configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Fallback: try to read .env file manually so local dev flags like DUMMY_SMS work
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    try:
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    # don't overwrite existing environment variables
                    if k not in os.environ:
                        os.environ[k] = v
            logger.info(f"Loaded .env fallback from {env_path}")
        else:
            logger.warning("python-dotenv not installed and .env not found; skipping .env loading")
    except Exception:
        logger.exception('Failed to parse .env fallback; continuing without it')
SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
SMTP_USE_SSL = os.getenv('SMTP_USE_SSL', '0')  # set to '1' to use SMTP_SSL (port 465)
SMTP_DEBUG = os.getenv('SMTP_DEBUG', '0')
REQUIRE_SMTP = os.getenv('REQUIRE_SMTP', '0')  # set to '1' to error when SMTP not configured
EMAIL_FROM = os.getenv('EMAIL_FROM', SMTP_USER)

# ==============================================================================
# 4. FASTAPI INITIALIZATION
# ==============================================================================
app = FastAPI(
    title="OncoDetect EdgeFusion API",
    description="AI-Powered Skin Lesion Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 OncoDetect API Starting Up...")
    # Load persisted history so predictions survive restarts
    try:
        load_history_from_disk()
    except Exception:
        logger.exception('Failed loading history during startup')

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 OncoDetect API Shutting Down...")

print(f"Loading EdgeFusionV3Net on {DEVICE}...")
# Attempt to prefer an ONNX runtime model if available to avoid heavy PyTorch
# binary installs during CI/local Docker builds. If `model.onnx` exists and
# can be loaded by onnxruntime, inference will use that. Otherwise fall back
# to the original PyTorch model and `.pth` state dict.
USE_ONNX = False
onnx_session = None
onnx_img_input = None
onnx_meta_input = None
onnx_path = os.path.join(os.path.dirname(__file__), 'model.onnx')
try:
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            input_names = [inp.name for inp in onnx_session.get_inputs()]
            # Common exported signature places image then metadata; record names if present
            onnx_img_input = input_names[0] if len(input_names) > 0 else None
            onnx_meta_input = input_names[1] if len(input_names) > 1 else None
            USE_ONNX = True
            logger.info(f"✓ ONNX model loaded via onnxruntime (will run on CPU). Inputs: {input_names}")
            print("ONNX model loaded — using ONNX Runtime for inference")
        except Exception:
            logger.exception('Failed to load ONNX model; will attempt Torch path')
            USE_ONNX = False
except Exception:
    logger.exception('ONNX detection failed; continuing')

model = None
if not USE_ONNX:
    # Load the PyTorch model as before
    try:
        model = EdgeFusionV3Net(NUM_CLASSES, NUM_META_FEATURES)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        logger.info(f"✓ Torch model loaded successfully on {DEVICE}!")
        print("Torch model loaded successfully!")
    except Exception:
        logger.exception('Failed to load Torch model; if you intended to use ONNX, ensure model.onnx is present')
        raise

# Matches your train/val transforms exactly
vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==============================================================================
# 5. DATA MODELS & TRANSLATION LOGIC
# ==============================================================================
class PatientMetadata(BaseModel):
    age: str
    gender: str
    location: str

class AssessmentRequest(BaseModel):
    image: str 
    metadata: PatientMetadata

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    raw_class: str
    timestamp: str
    id: str

class HistoryItemResponse(BaseModel):
    id: str
    prediction: str
    confidence: float
    age: str
    gender: str
    location: str
    timestamp: str

class PhoneRequest(BaseModel):
    phone_number: str


class VerifyPhoneRequest(BaseModel):
    phone_number: str
    code: str


def send_email_via_smtp(to_email: str, subject: str, body: str):
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        logger.warning("SMTP is not configured; email not sent")
        if REQUIRE_SMTP == '1':
            # In production enforce SMTP presence
            raise RuntimeError("SMTP not configured but REQUIRED by server settings")
        return None

    try:
        from email.message import EmailMessage
        import smtplib

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = EMAIL_FROM
        msg['To'] = to_email
        msg.set_content(body)

        # Choose SSL or STARTTLS based on env var
        if SMTP_USE_SSL == '1':
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                if int(SMTP_DEBUG):
                    server.set_debuglevel(1)
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                if int(SMTP_DEBUG):
                    server.set_debuglevel(1)
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)

        return True
    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
        return False


# ------------------ SMS / OTP SENDING ---------------------------------
# Provider configuration (supports Twilio via env vars)
SMS_PROVIDER = os.getenv('SMS_PROVIDER', 'twilio')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER')
DUMMY_SMS = os.getenv('DUMMY_SMS', '0')


def send_sms_via_twilio(to_number: str, message: str) -> bool:
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_FROM_NUMBER:
        logger.warning('Twilio not fully configured')
        return False

    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {
        'From': TWILIO_FROM_NUMBER,
        'To': to_number,
        'Body': message
    }
    try:
        resp = requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=10)
        if resp.status_code >= 200 and resp.status_code < 300:
            return True
        logger.error(f"Twilio send failed: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        logger.exception(f"Twilio request error: {e}")
        return False


def send_sms_via_provider(to_number: str, message: str) -> bool:
    if DUMMY_SMS == '1':
        logger.info(f"DUMMY_SMS active — would send to {to_number}: {message}")
        return True

    if SMS_PROVIDER == 'twilio':
        return send_sms_via_twilio(to_number, message)

    logger.error(f"Unsupported SMS_PROVIDER: {SMS_PROVIDER}")
    return False


def process_metadata(meta: PatientMetadata) -> torch.Tensor:
    # 1. Scale Age (Exactly like df['age'] / 100.0)
    try:
        age_scaled = float(meta.age) / 100.0
    except ValueError:
        age_scaled = 0.5 

    # 2. One-hot encode Sex
    sex_raw = meta.gender.lower()
    sex_vector = [1.0 if sex_raw == cat else 0.0 for cat in SEX_CATEGORIES]

    # 3. Translate Mobile App string -> CSV dataset string
    app_loc = meta.location
    loc_mapping = {
        "Head & Neck": "head/neck",
        "Upper Extremity": "upper extremity",
        "Lower Extremity": "lower extremity",
        "Anterior Trunk": "anterior torso",
        "Back": "posterior torso",
        "Acral (Soles/Palms)": "palms/soles",
        "Unknown / Other": "unknown"
    }

    # Apply translation, default to "unknown" if it doesn't map perfectly
    dataset_loc = loc_mapping.get(app_loc, "unknown")

    # 4. One-hot encode Localization
    loc_vector = [1.0 if dataset_loc == cat else 0.0 for cat in LOC_CATEGORIES]

    # 5. Concatenate into exactly 13 features [Age + 3 Sex + 9 Loc]
    feature_list = [age_scaled] + sex_vector + loc_vector

    return torch.tensor([feature_list], dtype=torch.float32)

# ==============================================================================
# 6. PREDICTION ENDPOINT
# ==============================================================================
# Note: email verification endpoints removed; phone/SMS OTP is primary.


@app.post("/send_sms_code")
async def send_sms_code(request: PhoneRequest):
    if not request.phone_number:
        raise HTTPException(status_code=400, detail="phone_number required")

    phone = request.phone_number.strip()
    # simple sanitize: require leading + or digits
    if not re.match(r"^\+?[0-9]{7,15}$", phone):
        raise HTTPException(status_code=400, detail="invalid_phone_number_format")

    code = f"{random.randint(0, 999999):06d}"
    verification_codes_phone[phone] = code

    message = f"Your OncoDetect verification code is: {code}"
    sent = False
    try:
        if DUMMY_SMS == '1':
            logger.info(f"DUMMY_SMS active — code for {phone}: {code}")
            sent = True
        else:
            sent = send_sms_via_provider(phone, message)
    except Exception:
        logger.exception('Error sending SMS')
        sent = False

    if not sent:
        return JSONResponse(status_code=502, content={"status": "error", "detail": "sms_send_failed"})

    resp = {"status": "code_sent"}
    if DUMMY_SMS == '1':
        resp['code'] = code
    return resp


@app.post("/verify_phone_code")
async def verify_phone_code(request: VerifyPhoneRequest):
    if not request.phone_number or not request.code:
        raise HTTPException(status_code=400, detail="phone_number and code required")

    phone = request.phone_number.strip()
    stored = verification_codes_phone.get(phone)
    if not stored:
        raise HTTPException(status_code=404, detail="no_code_for_phone")

    if request.code.strip() != stored:
        raise HTTPException(status_code=400, detail="invalid_code")

    verification_codes_phone.pop(phone, None)
    return {"status": "verified"}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "model": "EdgeFusionV3Net",
        "device": DEVICE,
        "predictions_processed": len(prediction_history)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_lesion(request: Request, file: UploadFile = File(None), age: Optional[str] = Form(None), gender: Optional[str] = Form(None), bodyLocationLabel: Optional[str] = Form(None)):
    """
    Predict skin lesion type from either:
      - JSON body matching AssessmentRequest (image as data URL / base64 + metadata)
      - multipart/form-data with an uploaded file (`file`) and form fields `age`, `gender`, `bodyLocationLabel`

    This endpoint accepts both formats for flexibility in mobile/browser uploads.
    """
    try:
        # Case 1: multipart/form-data upload
        if file is not None:
            content = await file.read()
            import base64 as _b64
            mime = file.content_type or 'image/png'
            image_dataurl = f"data:{mime};base64,{_b64.b64encode(content).decode('ascii')}"
            meta = PatientMetadata(age=(age or '0'), gender=(gender or 'unknown'), location=(bodyLocationLabel or 'unknown'))
            prediction_result = run_inference_from_base64(image_dataurl, meta)
        else:
            # Case 2: JSON body (existing path)
            try:
                body = await request.json()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")
            ar = AssessmentRequest(**body)
            prediction_result = run_inference_from_base64(ar.image, ar.metadata)

        prediction_id = prediction_result['id']
        predicted_friendly = prediction_result['prediction']
        confidence_percent = prediction_result['confidence']
        predicted_class_name = prediction_result['raw_class']
        timestamp = prediction_result['timestamp']

        # Store in history (and persist)
        history_item = prediction_result['history_item']
        prediction_history.append(history_item)
        try:
            save_history_to_disk()
        except Exception:
            logger.exception('Failed to persist history after predict')
        logger.info(f"[{prediction_id}] ✓ Prediction: {predicted_friendly} ({confidence_percent}%)")

        # Include heatmap data URL and full history_item (with confidence_breakdown)
        resp = {
            "prediction": predicted_friendly,
            "confidence": confidence_percent,
            "raw_class": predicted_class_name,
            "timestamp": timestamp,
            "id": prediction_id,
            "heatmap": prediction_result.get('heatmap'),
            "history_item": history_item,
            "confidence_breakdown": history_item.get('confidence_breakdown')
        }

        return JSONResponse(status_code=200, content=resp)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception('Error during inference')
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/history", response_model=List[HistoryItemResponse])
async def get_history(limit: int = 50):
    """Get prediction history"""
    return [
        HistoryItemResponse(**{k: v for k, v in item.items() if k != 'confidence_breakdown'})
        for item in sorted(prediction_history, key=lambda x: x['timestamp'], reverse=True)[:limit]
    ]

@app.delete("/history/{prediction_id}")
async def delete_prediction(prediction_id: str):
    """Delete a specific prediction from history"""
    global prediction_history
    prediction_history = [p for p in prediction_history if p['id'] != prediction_id]
    try:
        save_history_to_disk()
    except Exception:
        logger.exception('Failed to persist history after delete')
    return {"status": "deleted", "id": prediction_id}

@app.get("/stats")
async def get_stats():
    """Get statistics about predictions"""
    if not prediction_history:
        return {
            "total_predictions": 0,
            "top_prediction": None,
            "avg_confidence": 0
        }

    predictions = [p['prediction'] for p in prediction_history]
    confidence_scores = [p['confidence'] for p in prediction_history]

    from collections import Counter
    most_common = Counter(predictions).most_common(1)

    return {
        "total_predictions": len(prediction_history),
        "top_prediction": most_common[0][0] if most_common else None,
        "avg_confidence": round(sum(confidence_scores) / len(confidence_scores), 2),
        "prediction_breakdown": dict(Counter(predictions))
    }


# ------------------ Admin / Dev Helpers ---------------------------------
@app.post("/admin/clear_history")
async def admin_clear_history(confirm: bool = False):
    """Clear server-side prediction history (dev helper).

    Call with `?confirm=true` to actually clear. This will truncate the
    in-memory history and overwrite the persistent history file.
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="confirm=true query parameter required")

    global prediction_history
    prediction_history = []
    try:
        save_history_to_disk()
    except Exception:
        logger.exception('Failed to persist cleared history')

    return {"status": "cleared", "total_predictions": 0}


def run_inference_from_base64(image_base64: str, metadata: PatientMetadata) -> Dict:
    """Helper to run inference from data URL / base64 string and PatientMetadata."""
    import uuid
    prediction_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()

    logger.info(f"[{prediction_id}] Processing prediction request...")
    image_data_str = image_base64.split(',')[-1] if ',' in image_base64 else image_base64
    image_bytes = base64.b64decode(image_data_str)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = vision_transform(img).unsqueeze(0).to(DEVICE)

    meta_tensor = process_metadata(metadata).to(DEVICE)

    # If an ONNX session is available, run inference via onnxruntime (CPU)
    if USE_ONNX and onnx_session is not None:
        try:
            # Convert tensors to numpy in NCHW and float32
            img_np = img_tensor.cpu().numpy()
            meta_np = meta_tensor.cpu().numpy()
            feed = {}
            if onnx_img_input:
                feed[onnx_img_input] = img_np
            if onnx_meta_input:
                feed[onnx_meta_input] = meta_np
            outputs = onnx_session.run(None, feed)
            # Assume first output is logits [1, num_classes]
            logits = outputs[0]
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = probs / probs.sum(axis=1, keepdims=True)
            probabilities = probs[0]
            predicted_idx = int(np.argmax(probabilities))
            confidence_score = float(probabilities[predicted_idx])
        except Exception:
            logger.exception('ONNX inference failed; falling back to Torch (if available)')
            # fallback to torch path if available
            if model is None:
                raise RuntimeError('No valid model available for inference')
            with torch.no_grad():
                outputs = model(img_tensor, meta_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence_score, predicted_idx = torch.max(probabilities, 0)
                predicted_idx = int(predicted_idx)
                confidence_score = float(confidence_score)
    else:
        # Standard Torch inference
        with torch.no_grad():
            outputs = model(img_tensor, meta_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence_score, predicted_idx = torch.max(probabilities, 0)

    predicted_class_name = CLASS_NAMES[predicted_idx.item()]
    confidence_percent = round(confidence_score.item() * 100, 2)

    friendly_names = {
        "AKIEC": "Actinic Keratosis",
        "BCC": "Basal Cell Carcinoma",
        "BKL": "Benign Keratosis",
        "DF": "Dermatofibroma",
        "MEL": "Melanoma",
        "NV": "Nevus",
        "SCC": "Squamous Cell Carcinoma",
        "VASC": "Vascular Lesion"
    }

    predicted_friendly = friendly_names.get(predicted_class_name, predicted_class_name)

    history_item = {
        "id": prediction_id,
        "prediction": predicted_friendly,
        "confidence": confidence_percent,
        "raw_class": predicted_class_name,
        "age": metadata.age,
        "gender": metadata.gender,
        "location": metadata.location,
        "timestamp": timestamp,
        "confidence_breakdown": {
            CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
            for i in range(NUM_CLASSES)
        }
    }
    # Attempt to compute Grad-CAM heatmap for the vision backbone
    try:
        heatmap = compute_gradcam(model, img_tensor, meta_tensor, predicted_idx.item())
        # encode heatmap (grayscale) as PNG data URL
        from io import BytesIO
        from PIL import Image as PILImage
        hm_img = PILImage.fromarray((heatmap * 255).astype('uint8'), mode='L')
        buf = BytesIO()
        hm_img.save(buf, format='PNG')
        buf.seek(0)
        import base64 as _b64
        heatmap_dataurl = 'data:image/png;base64,' + _b64.b64encode(buf.read()).decode('ascii')
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        heatmap_dataurl = None

    return {
        'id': prediction_id,
        'prediction': predicted_friendly,
        'confidence': confidence_percent,
        'raw_class': predicted_class_name,
        'timestamp': timestamp,
        'history_item': history_item,
        'heatmap': heatmap_dataurl
    }


def compute_gradcam(full_model, img_tensor: torch.Tensor, meta_tensor: torch.Tensor, target_idx: int):
    """Compute Grad-CAM by capturing the last 4-D activation in the vision
    backbone (more robust than relying on nn.Conv2d class names).

    Returns a 2D numpy array normalized 0..1 matching the input image size.
    """
    vision = full_model.vision

    activations = []
    handles = []

    def forward_hook(module, inp, out):
        try:
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                activations.append((module, out))
        except Exception:
            pass

    # attach hooks to all modules (cheap for single forward)
    for m in vision.modules():
        try:
            handles.append(m.register_forward_hook(forward_hook))
        except Exception:
            continue

    # forward pass
    full_model.zero_grad()
    out = full_model(img_tensor, meta_tensor)

    # remove hooks
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    if not activations:
        raise RuntimeError('No 4-D activations captured from vision backbone')

    # take the last captured activation
    module, activation = activations[-1]

    grads = {}

    def backward_hook(grad):
        grads['value'] = grad

    # register tensor hook to capture grad of that activation
    activation.register_hook(backward_hook)

    # scalar score for target class
    score = out[0, target_idx]

    # backward
    score.backward(retain_graph=False)

    if 'value' not in grads:
        raise RuntimeError('Gradients not captured for Grad-CAM')

    grad = grads['value']  # [1,C,H,W]

    # global-average-pool gradients to get weights
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1,C,1,1]
    cam = torch.sum(weights * activation, dim=1)[0]  # [H,W]
    cam = torch.relu(cam)

    cam_np = cam.detach().cpu().numpy()
    if cam_np.max() > 0:
        cam_np = cam_np - cam_np.min()
        cam_np = cam_np / (cam_np.max() + 1e-8)
    else:
        cam_np = cam_np * 0.0

    # upsample to input image size
    _, _, H, W = img_tensor.shape
    import numpy as _np
    from PIL import Image as PILImage
    cam_img = PILImage.fromarray((_np.uint8(cam_np * 255)))
    cam_img = cam_img.resize((W, H), resample=PILImage.BILINEAR)
    cam_resized = _np.array(cam_img).astype('float32') / 255.0
    return cam_resized


def compute_image_quality_from_pil(pil_img: Image.Image) -> Dict:
    """Compute simple image-quality heuristics: blur, lighting, and subject scale.

    Returns a dict with numeric scores (0..100) and short suggestions.
    """
    try:
        arr = np.array(pil_img.convert('L')).astype(np.float32)
        # compute gradient magnitude as a proxy for sharpness
        gx = np.diff(arr, axis=1)
        gy = np.diff(arr, axis=0)
        gx = np.pad(gx, ((0,0),(0,1)), mode='constant')
        gy = np.pad(gy, ((0,1),(0,0)), mode='constant')
        grad = np.sqrt(gx**2 + gy**2)
        blur_score = float(np.var(grad))

        # Normalize blur score into 0..100 (heuristic)
        blur_norm = max(0.0, min(100.0, (blur_score / 200.0) * 100.0))

        mean_brightness = float(np.mean(arr))
        std_brightness = float(np.std(arr))

        lighting_issue = None
        if mean_brightness < 60:
            lighting_issue = 'underexposed'
        elif mean_brightness > 195:
            lighting_issue = 'overexposed'
        elif std_brightness < 20:
            lighting_issue = 'low_contrast'

        # Estimate subject area: fraction of pixels that differ significantly from median
        median = np.median(arr)
        subject_mask = np.abs(arr - median) > 15
        subject_fraction = float(np.mean(subject_mask))

        # Map subject fraction to a scale score (ideal ~0.05..0.5)
        if subject_fraction < 0.02:
            scale_issue = 'too_far'
        elif subject_fraction > 0.6:
            scale_issue = 'too_close'
        else:
            scale_issue = None

        return {
            'blur_score': round(100.0 - blur_norm, 2),
            'brightness_mean': round(mean_brightness, 2),
            'brightness_std': round(std_brightness, 2),
            'lighting_issue': lighting_issue,
            'subject_fraction': round(subject_fraction, 4),
            'scale_issue': scale_issue,
            'suggestions': []
        }
    except Exception as e:
        logger.exception('compute_image_quality failed')
        return {'error': str(e)}


@app.post('/image_quality')
async def image_quality(payload: Dict):
    """Assess an image (base64 or data URL) for blur, lighting, and scale issues.

    Expects JSON body: { "image": "data:image/...;base64,..." }
    """
    try:
        img_b64 = payload.get('image')
        if not img_b64:
            raise HTTPException(status_code=400, detail='image required')

        img_data = img_b64.split(',')[-1] if ',' in img_b64 else img_b64
        img_bytes = base64.b64decode(img_data)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        result = compute_image_quality_from_pil(pil_img)

        # Build human-friendly suggestions
        suggestions = []
        if result.get('blur_score') is not None and result['blur_score'] < 40:
            suggestions.append('Image appears blurry — try steadying the camera or tap to focus.')
        if result.get('lighting_issue') == 'underexposed':
            suggestions.append('Image is dark — increase lighting or move to a brighter area.')
        if result.get('lighting_issue') == 'overexposed':
            suggestions.append('Image is too bright — reduce direct light or move to diffused lighting.')
        if result.get('scale_issue') == 'too_far':
            suggestions.append('Move closer to the lesion — it appears small in the frame.')
        if result.get('scale_issue') == 'too_close':
            suggestions.append('Move slightly further away so the lesion is fully visible.')

        result['suggestions'] = suggestions
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception('image_quality failed')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/v1/predict')
async def predict_multipart(file: UploadFile = File(...), age: Optional[str] = Form(None), gender: Optional[str] = Form(None), bodyLocationLabel: Optional[str] = Form(None)):
    """Accept multipart/form-data with file upload and metadata to support the UI."""
    try:
        content = await file.read()
        # Convert bytes to base64 data URL
        import base64 as _b64
        data_url = f"data:{file.content_type};base64,{_b64.b64encode(content).decode('utf-8')}"

        # Build a minimal PatientMetadata object
        metadata = PatientMetadata(age=age or '0', gender=(gender or 'unknown'), location=(bodyLocationLabel or 'unknown'))

        result = run_inference_from_base64(data_url, metadata)
        # Persist into server-side history so stats and history endpoints reflect this prediction
        try:
            history_item = result.get('history_item')
            if history_item:
                prediction_history.append(history_item)
                save_history_to_disk()
                logger.info(f"[{history_item.get('id')}] Stored history item from multipart predict")
        except Exception as e:
            logger.warning(f"Failed to append/persist history item: {e}")

        # Return the full result (including heatmap data URL) so the UI can display it
        return result
    except Exception as e:
        logger.error(f"Error in multipart predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/smtp_health")
async def smtp_health():
    """Check SMTP connectivity using configured settings."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return JSONResponse(status_code=503, content={"status": "smtp_not_configured"})

    try:
        import smtplib

        if SMTP_USE_SSL == '1':
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=10)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
            server.ehlo()
            server.starttls()
            server.ehlo()

        if int(SMTP_DEBUG):
            server.set_debuglevel(1)

        server.login(SMTP_USER, SMTP_PASSWORD)
        server.quit()
        return {"status": "smtp_ok"}
    except Exception as e:
        logger.error(f"SMTP health check failed: {e}")
        return JSONResponse(status_code=502, content={"status": "smtp_error", "detail": str(e)})


@app.post('/notifications/send')
async def notifications_send(payload: Dict, background_tasks: BackgroundTasks):
    """Send a one-off notification email. Uses background task to avoid blocking.

    Body: { "email": "user@example.com", "subject": "...", "body": "..." }
    """
    email = payload.get('email')
    subject = payload.get('subject', 'OncoDetect Notification')
    body = payload.get('body', '')
    if not email:
        raise HTTPException(status_code=400, detail='email required')

    try:
        if DUMMY_VERIFICATION == '1':
            logger.info(f"(dev) notification to {email}: {subject} / {body}")
            return {"status": "sent", "dev": True}

        background_tasks.add_task(send_email_via_smtp, email, subject, body)
        return {"status": "scheduled"}
    except Exception as e:
        logger.exception('notification send failed')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/guidance/preventive')
async def preventive_guidance(payload: Dict):
    """Provide preventive guidance (sunscreen, UV alerts, behavioral nudges).

    Accepts: { metadata: { age, gender, location }, uv_index: optional number, skin_type: optional }
    """
    try:
        metadata = payload.get('metadata', {}) or {}
        uv_index = payload.get('uv_index')
        skin_type = (payload.get('skin_type') or 'unknown').lower()

        # Basic SPF recommendation rules (heuristic)
        spf = 30
        if uv_index is not None:
            try:
                uv = float(uv_index)
            except Exception:
                uv = None
        else:
            uv = None

        if uv is not None:
            if uv >= 8:
                spf = 50
            elif uv >= 3:
                spf = 30
            else:
                spf = 15
        else:
            # fall back to skin type heuristics
            if skin_type in ('i','ii','fair','very fair'):
                spf = 50
            else:
                spf = 30

        nudges = [
            f"Apply broad-spectrum sunscreen SPF {spf}+ and reapply every 2 hours when outdoors.",
            "Seek shade between 10:00am and 4:00pm when UV is highest.",
            "Wear protective clothing, hat, and UV-blocking sunglasses.",
            "Photograph lesions monthly and compare for changes."
        ]

        alerts = []
        if uv is not None and uv >= 8:
            alerts.append('UV index is very high today — minimize sun exposure.')
        elif uv is not None and uv >= 6:
            alerts.append('UV index is high — use SPF50 and limit midday sun.')

        # Add simple age-based guidance
        age = metadata.get('age')
        try:
            age_n = int(age)
            if age_n >= 65:
                nudges.append('Older age increases skin cancer risk — maintain regular dermatologist visits.')
        except Exception:
            pass

        return {
            'spf_recommendation': spf,
            'nudges': nudges,
            'alerts': alerts,
            'next_steps': [
                'Set a reminder to re-scan in 3 months for routine monitoring.',
                'If you notice rapid changes, seek dermatologic assessment.'
            ]
        }
    except Exception as e:
        logger.exception('preventive guidance failed')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/assistant/analyze')
async def assistant_analyze(payload: Dict):
    try:
        # 1. Pull data from payload correctly
        history_item = payload.get('history_item')
        question = payload.get('question', 'Explain these results.')

        # 2. Get credentials from environment
        llm_url = os.getenv('LLM_API_URL')
        llm_key = os.getenv('LLM_API_KEY')
        llm_model = os.getenv('LLM_MODEL', 'meta-llama/llama-3.1-8b-instruct:free')

        # If no external LLM is configured, fall back to a local KB-based reply
        if not llm_url:
            # Build a concise assistant reply using local MEDICAL_KB when possible
            try:
                pred = history_item.get('prediction') if history_item else 'Unknown'
                conf = history_item.get('confidence') if history_item else '0'
                raw = history_item.get('raw_class') if history_item else None
                kb_entry = None
                if raw and MEDICAL_KB:
                    # MEDICAL_KB keys may be raw class names or friendly names
                    kb_entry = MEDICAL_KB.get(raw) or MEDICAL_KB.get(pred)

                assistant_text = ''
                assistant_text += f"Prediction: {pred} (Confidence: {conf}%).\n\n"
                if kb_entry:
                    # If KB entry is a dict containing 'summary' or 'advice'
                    if isinstance(kb_entry, dict):
                        assistant_text += kb_entry.get('summary', '') + '\n\n'
                        assistant_text += kb_entry.get('advice', '')
                    else:
                        assistant_text += str(kb_entry)
                else:
                    assistant_text += "I don't have extended clinical notes for this condition locally. "
                    assistant_text += "This is an automated explanation and not a medical diagnosis. Please consult a dermatologist for professional evaluation."

                # Append user's question if provided
                if question:
                    assistant_text += f"\n\nUser question: {question}\nAnswer: Based on the information above, I recommend consulting a dermatologist for any concerns."

                return {"assistant_text": assistant_text}
            except Exception as e:
                logger.exception('Local KB assistant fallback failed')
                return {"assistant_text": f"Assistant unavailable: {str(e)}"}

        # 3. OpenRouter HEADERS (Crucial for Free Tier)
        headers = {
            "Authorization": f"Bearer {llm_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5173", # Tells OpenRouter where request is from
            "X-Title": "OncoDetect Assistant"
        }

        # 4. Structured Chat Body
        chat_body = {
            "model": llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional clinical assistant for OncoDetect. Explain the skin lesion analysis clearly. Be concise."
                },
                {
                    "role": "user",
                    "content": f"Prediction: {history_item.get('prediction') if history_item else 'Unknown'}. Confidence: {history_item.get('confidence') if history_item else '0'}%. User Question: {question}"
                }
            ]
        }

        # 5. Send Request with longer timeout
        response = requests.post(llm_url, headers=headers, json=chat_body, timeout=45)

        if response.status_code != 200:
            # Log the provider error for diagnostics but return a friendly local fallback to the client
            try:
                logger.warning(f"LLM provider returned {response.status_code}: {response.text}")
            except Exception:
                pass
            # Try local KB fallback when external LLM fails
            try:
                if history_item:
                    pred = history_item.get('prediction')
                    conf = history_item.get('confidence')
                    raw = history_item.get('raw_class')
                else:
                    pred = 'Unknown'; conf = '0'; raw = None
                kb_entry = None
                if raw and MEDICAL_KB:
                    kb_entry = MEDICAL_KB.get(raw) or MEDICAL_KB.get(pred)

                assistant_text = f"Using local knowledge to answer. Prediction: {pred} (Confidence: {conf}%).\n\n"
                if kb_entry:
                    if isinstance(kb_entry, dict):
                        assistant_text += (kb_entry.get('summary', '') or '') + '\n\n' + (kb_entry.get('advice', '') or '')
                    else:
                        assistant_text += str(kb_entry)
                else:
                    assistant_text += "No local KB entry found. Please consult a dermatologist for professional evaluation."

                return {"assistant_text": assistant_text}
            except Exception:
                return {"assistant_text": "Assistant is temporarily unavailable. Please try again shortly."}

        data = response.json()

        if 'choices' in data and len(data['choices']) > 0:
            assistant_message = data['choices'][0]['message']['content']
            return {"assistant_text": assistant_message}

        return {"assistant_text": "The AI model responded but had no content. Please try a different question."}

    except Exception as e:
        logger.exception('Assistant analyze failed')
        return {"assistant_text": f"Connection Error: {str(e)}"}
@app.get('/assistant/llm_test')
async def assistant_llm_test():
    """Quick diagnostic endpoint to test LLM connectivity and shapes."""
    llm_url = os.getenv('LLM_API_URL')
    llm_key = os.getenv('LLM_API_KEY')
    if not llm_url:
        # No external LLM configured — provide informative local-KB diagnostic
        kb_loaded = bool(MEDICAL_KB)
        sample_kb_keys = list(MEDICAL_KB.keys())[:5]
        return {
            "status": "no_llm_configured",
            "local_kb_loaded": kb_loaded,
            "kb_keys_sample": sample_kb_keys,
            "message": "No external LLM URL configured (LLM_API_URL). The assistant will use local KB fallbacks."
        }
    headers = {"Content-Type": "application/json"}
    if llm_key:
        headers['Authorization'] = f"Bearer {llm_key}"
    sample = {"inputs": "Hello from OncoDetect diagnostic check", "parameters": {"max_new_tokens": 50}}
    try:
        r = requests.post(llm_url, headers=headers, json=sample, timeout=10)
        # If provider responded OK for the generic shape, return that; otherwise attempt chat shape
        if 200 <= r.status_code < 300:
            return {"status": "ok", "code": r.status_code, "body_snippet": r.text[:200]}
        # generic shape returned non-2xx — try chat shape with model field
        try:
            chat_body = {"model": os.getenv('LLM_MODEL', ''), "messages": [{"role": "user", "content": "Hello from OncoDetect diagnostic check"}]}
            rc = requests.post(llm_url, headers=headers, json=chat_body, timeout=10)
            return {"status": "ok_chat_shape", "code": rc.status_code, "body_snippet": rc.text[:200]}
        except Exception as e2:
            logger.error(f"LLM test: generic returned {r.status_code}; chat attempt failed: {e2}")
            return {"status": "generic_non2xx", "code": r.status_code, "body_snippet": r.text[:200], "chat_error": str(e2)}
    except Exception as e:
        # Primary generic attempt raised — try chat shape as fallback
        try:
            chat_body = {"model": os.getenv('LLM_MODEL', ''), "messages": [{"role": "user", "content": "Hello from OncoDetect diagnostic check"}]}
            r = requests.post(llm_url, headers=headers, json=chat_body, timeout=10)
            return {"status": "ok_chat_shape", "code": r.status_code, "body_snippet": r.text[:200]}
        except Exception as e2:
            logger.error(f"LLM test failed: {e} ; chat attempt failed: {e2}")
            return JSONResponse(status_code=502, content={"status": "error", "detail": str(e2)})