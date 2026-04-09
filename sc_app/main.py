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
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "edgefusion_v3_hybrid_champion.pth")

CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
NUM_CLASSES = len(CLASS_NAMES)

SEX_CATEGORIES = ['female', 'male', 'unknown']
LOC_CATEGORIES = [
    'anterior torso', 'head/neck', 'lateral torso', 'lower extremity', 
    'oral/genital', 'palms/soles', 'posterior torso', 'unknown', 'upper extremity'
]
NUM_META_FEATURES = 1 + len(SEX_CATEGORIES) + len(LOC_CATEGORIES)

# ==============================================================================
# 2. MODEL ARCHITECTURE
# ==============================================================================
class EdgeFusionV3Net(nn.Module):
    def __init__(self, num_classes, num_meta_features):
        super(EdgeFusionV3Net, self).__init__()
        self.vision = timm.create_model('efficientformerv2_s1', pretrained=False, num_classes=0)
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 224, 224)
            vision_out_features = self.vision(dummy_img).shape[1]

        self.meta_mlp = nn.Sequential(
            nn.Linear(num_meta_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        combined_dim = vision_out_features + 32
        self.se_gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 4),
            nn.ReLU(),
            nn.Linear(combined_dim // 4, combined_dim),
            nn.Sigmoid()
        )

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
# 3. UTILS & DATA PERSISTENCE
# ==============================================================================
prediction_history: List[Dict] = []
verification_codes_phone: Dict[str, str] = {}
MEDICAL_KB: Dict = {}
HISTORY_PATH = os.path.join(os.path.dirname(__file__), 'data', 'history.json')

def load_history_from_disk():
    global prediction_history
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r', encoding='utf-8') as hf:
                prediction_history = json.load(hf)
    except Exception as e:
        logger.error(f"History load failed: {e}")

def save_history_to_disk():
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, 'w', encoding='utf-8') as hf:
            json.dump(prediction_history, hf, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"History save failed: {e}")

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Env Config
SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
EMAIL_FROM = os.getenv('EMAIL_FROM', SMTP_USER)
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER')
DUMMY_SMS = os.getenv('DUMMY_SMS', '0')

# ==============================================================================
# 4. FASTAPI SETUP & MODEL LOAD
# ==============================================================================
app = FastAPI(title="OncoDetect API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Torch Model
model = EdgeFusionV3Net(NUM_CLASSES, NUM_META_FEATURES)
if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully")
else:
    logger.warning(f"Model file {MODEL_SAVE_PATH} not found!")

vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==============================================================================
# 5. SCHEMAS
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

class PhoneRequest(BaseModel):
    phone_number: str

class VerifyPhoneRequest(BaseModel):
    phone_number: str
    code: str

# ==============================================================================
# 6. ENDPOINTS
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    load_history_from_disk()

@app.get("/health")
async def health():
    return {"status": "healthy", "device": DEVICE}

@app.post("/send_sms_code")
async def send_sms_code(request: PhoneRequest):
    code = f"{random.randint(100000, 999999)}"
    verification_codes_phone[request.phone_number] = code
    message = f"Your OncoDetect code is: {code}"
    
    if DUMMY_SMS == '1':
        logger.info(f"DUMMY SMS to {request.phone_number}: {code}")
        return {"status": "code_sent", "code": code}
    
    # Twilio API Call
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {'From': TWILIO_FROM_NUMBER, 'To': request.phone_number, 'Body': message}
    resp = requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    
    if resp.status_code < 300:
        return {"status": "code_sent"}
    raise HTTPException(status_code=500, detail="Twilio failed")

@app.post("/verify_phone_code")
async def verify_phone_code(request: VerifyPhoneRequest):
    stored = verification_codes_phone.get(request.phone_number)
    if stored and stored == request.code:
        verification_codes_phone.pop(request.phone_number)
        return {"status": "verified"}
    raise HTTPException(status_code=400, detail="Invalid code")

@app.post("/predict")
async def predict_lesion(
    request: Request, 
    file: UploadFile = File(None), 
    age: Optional[str] = Form(None), 
    gender: Optional[str] = Form(None), 
    location: Optional[str] = Form(None)
):
    try:
        # Handle Multipart (Mobile)
        if file:
            content = await file.read()
            image_data = f"data:{file.content_type};base64,{base64.b64encode(content).decode()}"
            meta = PatientMetadata(age=age or "0", gender=gender or "unknown", location=location or "unknown")
        # Handle JSON (Web)
        else:
            body = await request.json()
            ar = AssessmentRequest(**body)
            image_data = ar.image
            meta = ar.metadata

        result = run_inference(image_data, meta)
        prediction_history.append(result['history_item'])
        save_history_to_disk()
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    return sorted(prediction_history, key=lambda x: x['timestamp'], reverse=True)

@app.get("/stats")
async def get_stats():
    if not prediction_history:
        return {"total_predictions": 0, "avg_confidence": 0}
    confidences = [p['confidence'] for p in prediction_history]
    return {
        "total_predictions": len(prediction_history),
        "avg_confidence": round(sum(confidences)/len(confidences), 2)
    }

# ==============================================================================
# 7. INFERENCE ENGINE
# ==============================================================================

def run_inference(image_base64: str, metadata: PatientMetadata):
    pred_id = str(uuid.uuid4())[:8]
    # Process Image
    img_str = image_base64.split(',')[-1]
    img = Image.open(io.BytesIO(base64.b64decode(img_str))).convert("RGB")
    img_t = vision_transform(img).unsqueeze(0).to(DEVICE)

    # Process Meta
    age_s = float(metadata.age) / 100.0
    sex_v = [1.0 if metadata.gender.lower() == c else 0.0 for c in SEX_CATEGORIES]
    loc_v = [1.0 if metadata.location.lower() == c else 0.0 for c in LOC_CATEGORIES]
    meta_t = torch.tensor([[age_s] + sex_v + loc_v], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        out = model(img_t, meta_t)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        conf, idx = torch.max(probs, 0)

    friendly_names = {"AKIEC": "Actinic Keratosis", "BCC": "Basal Cell Carcinoma", "MEL": "Melanoma", "NV": "Nevus"}
    raw_class = CLASS_NAMES[idx.item()]
    
    res = {
        "id": pred_id,
        "prediction": friendly_names.get(raw_class, raw_class),
        "confidence": round(conf.item() * 100, 2),
        "raw_class": raw_class,
        "timestamp": datetime.now().isoformat(),
        "age": metadata.age,
        "gender": metadata.gender,
        "location": metadata.location
    }
    return {"id": pred_id, **res, "history_item": res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
