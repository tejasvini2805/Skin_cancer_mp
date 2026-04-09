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
# 2. MODEL ARCHITECTURE (Exactly same as Training)
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
# 3. SETUP & APP
# ==============================================================================
app = FastAPI(title="OncoDetect API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
prediction_history = []
HISTORY_PATH = os.path.join(os.path.dirname(__file__), 'data', 'history.json')

@app.on_event("startup")
async def startup():
    global model, prediction_history
    # Load History
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r') as f: prediction_history = json.load(f)
    except: pass
    # Load Model
    model = EdgeFusionV3Net(NUM_CLASSES, NUM_META_FEATURES)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
        model.eval()
    logger.info("✓ Backend Startup Ready")

vision_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class PatientMetadata(BaseModel):
    age: str
    gender: str
    location: str

# ==============================================================================
# 4. ENDPOINTS
# ==============================================================================

@app.get("/health")
def health(): return {"status": "healthy"}

@app.post('/assistant/analyze')
async def assistant_analyze(payload: Dict):
    try:
        h_item = payload.get('history_item', {})
        question = payload.get('question', 'Analyze these results.')
        llm_key = os.getenv('LLM_API_KEY')
        
        if not llm_key: return {"assistant_text": "Assistant Error: API Key not set in Render."}

        headers = {"Authorization": f"Bearer {llm_key}", "Content-Type": "application/json"}
        chat_data = {
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "messages": [
                {"role": "system", "content": "You are OncoDetect Clinical Assistant. Explain results concisely."},
                {"role": "user", "content": f"Prediction: {h_item.get('prediction')}. Confidence: {h_item.get('confidence')}%. User Question: {question}"}
            ]
        }

        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=chat_data, timeout=15)
        res = r.json()
        
        # SAFELY check for choices to avoid the 'choices' error
        if 'choices' in res:
            return {"assistant_text": res['choices'][0]['message']['content']}
        else:
            error_msg = res.get('error', {}).get('message', 'Unknown API Error')
            return {"assistant_text": f"AI Error: {error_msg}"}
    except Exception as e:
        return {"assistant_text": f"Connection Error: {str(e)}"}
# --- Add these after your /assistant/analyze endpoint ---

@app.post("/send_sms_code")
async def send_sms_code(request: PhoneRequest):
    # Generate a 6-digit code
    code = f"{random.randint(100000, 999999)}"
    
    # Store it in memory (Note: In production, use Redis or a DB)
    verification_codes_phone[request.phone_number] = code
    
    # Check for Dummy Mode (Dev testing)
    if os.getenv("DUMMY_SMS") == '1':
        logger.info(f"DUMMY SMS to {request.phone_number}: {code}")
        return {"status": "code_sent", "code": code}
    
    # Real Twilio Logic
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{os.getenv('TWILIO_ACCOUNT_SID')}/Messages.json"
        data = {
            'From': os.getenv('TWILIO_FROM_NUMBER'),
            'To': request.phone_number,
            'Body': f"Your OncoDetect code is: {code}"
        }
        resp = requests.post(
            url, 
            data=data, 
            auth=(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN')),
            timeout=10
        )
        if resp.status_code < 300:
            return {"status": "code_sent"}
        else:
            logger.error(f"Twilio Error: {resp.text}")
            raise HTTPException(status_code=500, detail="Failed to send SMS via Twilio")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify_phone_code")
async def verify_phone_code(request: VerifyPhoneRequest):
    stored_code = verification_codes_phone.get(request.phone_number)
    if stored_code and stored_code == request.code:
        # Success! Remove code so it can't be reused
        verification_codes_phone.pop(request.phone_number, None)
        return {"status": "verified"}
    
    raise HTTPException(status_code=400, detail="Invalid or expired verification code")

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(None), age: str = Form("0"), gender: str = Form("unknown"), location: str = Form("unknown")):
    try:
        # Handle file or JSON
        if file:
            content = await file.read()
            img_b64 = f"data:image/png;base64,{base64.b64encode(content).decode()}"
            meta = PatientMetadata(age=age, gender=gender, location=location)
        else:
            body = await request.json()
            img_b64, meta = body['image'], PatientMetadata(**body['metadata'])

        # INFERENCE
        img_str = img_b64.split(',')[-1]
        img = Image.open(io.BytesIO(base64.b64decode(img_str))).convert("RGB")
        img_t = vision_transform(img).unsqueeze(0)
        
        age_s = float(meta.age) / 100.0
        sex_v = [1.0 if meta.gender.lower() == c else 0.0 for c in SEX_CATEGORIES]
        loc_v = [1.0 if meta.location.lower() == c else 0.0 for c in LOC_CATEGORIES]
        meta_t = torch.tensor([[age_s] + sex_v + loc_v], dtype=torch.float32)

        with torch.no_grad():
            output = model(img_t, meta_t)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            conf, idx = torch.max(probs, 0)

        raw_class = CLASS_NAMES[idx.item()]
        
        # Restore full breakdown
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
        # Save history
        try:
            os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
            with open(HISTORY_PATH, 'w') as f: json.dump(prediction_history[-100:], f)
        except: pass
        
        return {**res, "history_item": res}
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(): return sorted(prediction_history, key=lambda x: x['timestamp'], reverse=True)[:50]

@app.get("/stats")
async def get_stats():
    if not prediction_history: return {"total_predictions": 0, "avg_confidence": 0}
    confs = [p['confidence'] for p in prediction_history]
    return {"total_predictions": len(prediction_history), "avg_confidence": round(sum(confs)/len(confs), 2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
