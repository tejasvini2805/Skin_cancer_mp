from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
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
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
DEVICE = "cpu"
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "edgefusion_v3_hybrid_champion.pth")

CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
FRIENDLY_NAMES = {
    "AKIEC": "Actinic Keratosis",
    "BCC": "Basal Cell Carcinoma",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "MEL": "Melanoma",
    "NV": "Nevus",
    "SCC": "Squamous Cell Carcinoma",
    "VASC": "Vascular Lesion"
}
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
            v_out = self.vision(dummy_img).shape[1]

        self.meta_mlp = nn.Sequential(
            nn.Linear(num_meta_features, 32), nn.ReLU(),
            nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 32), nn.ReLU()
        )
        combined_dim = v_out + 32
        self.se_gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 4), nn.ReLU(),
            nn.Linear(combined_dim // 4, combined_dim), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, meta):
        v_feat = self.vision(img)
        m_feat = self.meta_mlp(meta)
        fused = torch.cat((v_feat, m_feat), dim=1)
        fused = fused * self.se_gate(fused)
        return self.classifier(fused)

# ==============================================================================
# 3. GLOBAL STATE & SCHEMAS
# ==============================================================================
model = None
prediction_history = []
verification_codes_phone: Dict[str, str] = {}
HISTORY_PATH = os.path.join(os.path.dirname(__file__), 'data', 'history.json')

class PatientMetadata(BaseModel):
    age: str
    gender: str
    location: str

class AssessmentRequest(BaseModel):
    image: str 
    metadata: PatientMetadata

class PhoneRequest(BaseModel):
    phone_number: str

class VerifyPhoneRequest(BaseModel):
    phone_number: str
    code: str

# ==============================================================================
# 4. APP SETUP & STARTUP
# ==============================================================================
app = FastAPI(title="OncoDetect API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    global model, prediction_history
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r') as f:
                prediction_history = json.load(f)
    except:
        prediction_history = []
    
    model = EdgeFusionV3Net(NUM_CLASSES, NUM_META_FEATURES)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
        model.eval()
        logger.info("✓ Model loaded on CPU")
    else:
        logger.error(f"✗ Model file {MODEL_SAVE_PATH} not found")

vision_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==============================================================================
# 5. ENDPOINTS
# ==============================================================================

@app.get("/health")
def health(): return {"status": "healthy", "device": DEVICE}

@app.post("/send_sms_code")
async def send_sms_code(request: PhoneRequest):
    code = f"{random.randint(100000, 999999)}"
    verification_codes_phone[request.phone_number] = code
    
    if os.getenv("DUMMY_SMS") == '1':
        logger.info(f"DUMMY SMS to {request.phone_number}: {code}")
        return {"status": "code_sent", "code": code}
    
    try:
        sid = os.getenv('TWILIO_ACCOUNT_SID')
        token = os.getenv('TWILIO_AUTH_TOKEN')
        from_num = os.getenv('TWILIO_FROM_NUMBER')
        
        if not all([sid, token, from_num]):
            raise HTTPException(status_code=500, detail="Twilio credentials missing")

        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        data = {'From': from_num, 'To': request.phone_number, 'Body': f"Your OncoDetect code is: {code}"}
        resp = requests.post(url, data=data, auth=(sid, token), timeout=10)
        
        if resp.status_code < 300:
            return {"status": "code_sent"}
        else:
            logger.error(f"Twilio Error: {resp.text}")
            raise HTTPException(status_code=500, detail="Twilio gateway error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify_phone_code")
async def verify_phone_code(request: VerifyPhoneRequest):
    stored = verification_codes_phone.get(request.phone_number)
    if stored and stored == request.code:
        verification_codes_phone.pop(request.phone_number, None)
        return {"status": "verified"}
    raise HTTPException(status_code=400, detail="Invalid verification code")

@app.post('/assistant/analyze')
async def assistant_analyze(payload: Dict):
    """
    Enterprise-grade LLM handler with native model routing and offline degradation.
    """
    try:
        h_item = payload.get('history_item') or {}
        question = payload.get('question') or 'Analyze these results.'
        llm_key = os.getenv('LLM_API_KEY')
        
        pred = h_item.get('prediction', 'No image analyzed')
        conf = h_item.get('confidence', 'N/A')

        # The Ultimate Safety Net: If the AI completely fails, return this safe, professional text.
        offline_fallback = (
            f"**Automated Clinical Note:**\n"
            f"Prediction: {pred} (Confidence: {conf}%).\n\n"
            f"⚠️ *Our live AI servers are experiencing high traffic. Please note that this tool is for screening purposes only. "
            f"Based on your results, we strongly advise scheduling an appointment with a certified dermatologist for a professional biopsy and diagnosis.*"
        )

        if not llm_key: 
            return {"assistant_text": offline_fallback}

        headers = {
            "Authorization": f"Bearer {llm_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://oncodetect.com", 
            "X-Title": "OncoDetect Clinical App"
        }

        # NATIVE ROUTING: We pass an array of models. OpenRouter will instantly 
        # try them from top to bottom until one works.
        chat_data = {
            "models": [
                "google/gemini-2.0-flash-lite-preview-02-05:free", # Highly available 
                "meta-llama/llama-3.1-8b-instruct:free",           # Excellent reasoning
                "mistralai/mistral-7b-instruct:free",              # Solid backup
                "openchat/openchat-7b:free",
                "huggingfaceh4/zephyr-7b-beta:free"
            ],
            "route": "fallback", # Instructs OpenRouter to auto-failover
            "messages": [
                {"role": "system", "content": "You are OncoDetect Clinical Assistant. Explain results concisely and professionally. If no image is analyzed, answer general skin health questions."},
                {"role": "user", "content": f"Current Analysis: {pred} (Confidence: {conf}%). Question: {question}"}
            ]
        }

        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=chat_data, timeout=20)
        
        # Safely attempt to parse the JSON. If the server returned an HTML error page, this triggers the except block.
        res = r.json()
        
        if 'choices' in res and len(res['choices']) > 0:
            return {"assistant_text": res['choices'][0]['message']['content']}
        
        # If OpenRouter responded cleanly but said ALL models are down
        logger.warning(f"OpenRouter exhausted all models: {res.get('error', {}).get('message', 'Unknown Error')}")
        return {"assistant_text": offline_fallback}

    except (requests.exceptions.Timeout, requests.exceptions.RequestException, ValueError) as e:
        # Catches Timeouts, DNS failures, and Nginx HTML 502/504 errors
        logger.error(f"LLM Network/Parsing Error: {e}")
        return {"assistant_text": offline_fallback}
    except Exception as e:
        logger.exception("Fatal Assistant Error")
        return {"assistant_text": f"System Error: {str(e)}"}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(None), age: str = Form("0"), gender: str = Form("unknown"), location: str = Form("unknown")):
    try:
        if file:
            content = await file.read()
            img_b64 = f"data:image/png;base64,{base64.b64encode(content).decode()}"
            meta = PatientMetadata(age=age, gender=gender, location=location)
        else:
            body = await request.json()
            img_b64, meta = body['image'], PatientMetadata(**body['metadata'])

        img_str = img_b64.split(',')[-1]
        img = Image.open(io.BytesIO(base64.b64decode(img_str))).convert("RGB")
        img_t = vision_transform(img).unsqueeze(0).to(DEVICE)
        
        age_s = float(meta.age) / 100.0
        sex_v = [1.0 if meta.gender.lower() == c else 0.0 for c in SEX_CATEGORIES]
        loc_v = [1.0 if meta.location.lower() == c else 0.0 for c in LOC_CATEGORIES]
        meta_t = torch.tensor([[age_s] + sex_v + loc_v], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            output = model(img_t, meta_t)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            conf, idx = torch.max(probs, 0)

        raw_class = CLASS_NAMES[idx.item()]
        breakdown = {CLASS_NAMES[i]: round(probs[i].item() * 100, 2) for i in range(NUM_CLASSES)}

        res = {
            "id": str(uuid.uuid4())[:8],
            "prediction": FRIENDLY_NAMES.get(raw_class, raw_class),
            "confidence": round(conf.item() * 100, 2),
            "confidence_breakdown": breakdown,
            "timestamp": datetime.now().isoformat(),
            "age": meta.age, "gender": meta.gender, "location": meta.location
        }
        
        prediction_history.append(res)
        try:
            with open(HISTORY_PATH, 'w') as f: json.dump(prediction_history[-100:], f)
        except: pass
        
        return {**res, "history_item": res}
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(): 
    return sorted(prediction_history, key=lambda x: x['timestamp'], reverse=True)[:50]

@app.get("/stats")
async def get_stats():
    if not prediction_history: return {"total_predictions": 0, "avg_confidence": 0}
    confs = [p['confidence'] for p in prediction_history]
    return {"total_predictions": len(prediction_history), "avg_confidence": round(sum(confs)/len(confs), 2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
