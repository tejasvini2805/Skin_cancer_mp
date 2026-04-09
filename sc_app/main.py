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
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
DEVICE = "cpu" # Force CPU for Render stability
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
# 3. DATA PERSISTENCE
# ==============================================================================
prediction_history: List[Dict] = []
verification_codes_phone: Dict[str, str] = {}
HISTORY_PATH = os.path.join(os.path.dirname(__file__), 'data', 'history.json')

def load_history_from_disk():
    global prediction_history
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r', encoding='utf-8') as hf:
                prediction_history = json.load(hf)
    except Exception: pass

def save_history_to_disk():
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, 'w', encoding='utf-8') as hf:
            json.dump(prediction_history, hf, ensure_ascii=False, indent=2)
    except Exception: pass

# ==============================================================================
# 4. FASTAPI SETUP
# ==============================================================================
app = FastAPI(title="OncoDetect API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    global model
    load_history_from_disk()
    model = EdgeFusionV3Net(NUM_CLASSES, NUM_META_FEATURES)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
        model.eval()
        logger.info("✓ Model Loaded")
    else:
        logger.error("✗ Model file not found")

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

class PhoneRequest(BaseModel):
    phone_number: str

class VerifyPhoneRequest(BaseModel):
    phone_number: str
    code: str

# ==============================================================================
# 6. ENDPOINTS
# ==============================================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "device": DEVICE}

@app.post("/send_sms_code")
async def send_sms_code(request: PhoneRequest):
    code = f"{random.randint(100000, 999999)}"
    verification_codes_phone[request.phone_number] = code
    if os.getenv("DUMMY_SMS") == '1':
        return {"status": "code_sent", "code": code}
    
    url = f"https://api.twilio.com/2010-04-01/Accounts/{os.getenv('TWILIO_ACCOUNT_SID')}/Messages.json"
    data = {'From': os.getenv('TWILIO_FROM_NUMBER'), 'To': request.phone_number, 'Body': f"Code: {code}"}
    resp = requests.post(url, data=data, auth=(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN')))
    return {"status": "code_sent"} if resp.status_code < 300 else JSONResponse(status_code=500, content={"detail": "SMS fail"})

@app.post("/verify_phone_code")
async def verify_phone_code(request: VerifyPhoneRequest):
    if verification_codes_phone.get(request.phone_number) == request.code:
        return {"status": "verified"}
    raise HTTPException(status_code=400, detail="Invalid code")

# --- THE MISSING ASSISTANT ENDPOINT ---
@app.post('/assistant/analyze')
async def assistant_analyze(payload: Dict):
    try:
        history_item = payload.get('history_item', {})
        question = payload.get('question', 'Explain these results.')
        
        llm_key = os.getenv('LLM_API_KEY')
        if not llm_key:
            return {"assistant_text": "AI Assistant is not configured (Missing API Key)."}

        headers = {
            "Authorization": f"Bearer {llm_key}",
            "Content-Type": "application/json"
        }

        chat_body = {
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "messages": [
                {"role": "system", "content": "You are OncoDetect Clinical Assistant. Explain skin results clearly."},
                {"role": "user", "content": f"Result: {history_item.get('prediction')} ({history_item.get('confidence')}%). Question: {question}"}
            ]
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=chat_body, timeout=15)
        res_json = response.json()
        return {"assistant_text": res_json['choices'][0]['message']['content']}
    except Exception as e:
        return {"assistant_text": f"Assistant Error: {str(e)}"}

@app.post("/predict")
async def predict_lesion(request: Request, file: UploadFile = File(None), age: Optional[str] = Form(None), gender: Optional[str] = Form(None), location: Optional[str] = Form(None)):
    try:
        if file:
            content = await file.read()
            img_b64 = f"data:{file.content_type};base64,{base64.b64encode(content).decode()}"
            meta = PatientMetadata(age=age or "0", gender=gender or "unknown", location=location or "unknown")
        else:
            body = await request.json()
            ar = AssessmentRequest(**body)
            img_b64, meta = ar.image, ar.metadata

        result = run_inference(img_b64, meta)
        prediction_history.append(result['history_item'])
        save_history_to_disk()
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    return sorted(prediction_history, key=lambda x: x['timestamp'], reverse=True)[:50]

@app.get("/stats")
async def get_stats():
    if not prediction_history: return {"total_predictions": 0, "avg_confidence": 0}
    confs = [p['confidence'] for p in prediction_history]
    return {"total_predictions": len(prediction_history), "avg_confidence": round(sum(confs)/len(confs), 2)}

# ==============================================================================
# 7. INFERENCE ENGINE
# ==============================================================================
def run_inference(image_base64: str, metadata: PatientMetadata):
    # Standardize image data
    img_str = image_base64.split(',')[-1]
    img = Image.open(io.BytesIO(base64.b64decode(img_str))).convert("RGB")
    img_t = vision_transform(img).unsqueeze(0).to("cpu")

    # Meta Vector
    age_s = float(metadata.age) / 100.0
    sex_v = [1.0 if metadata.gender.lower() == c else 0.0 for c in SEX_CATEGORIES]
    loc_v = [1.0 if metadata.location.lower() == c else 0.0 for c in LOC_CATEGORIES]
    meta_t = torch.tensor([[age_s] + sex_v + loc_v], dtype=torch.float32).to("cpu")

    with torch.no_grad():
        out = model(img_t, meta_t)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        conf, idx = torch.max(probs, 0)

    raw_class = CLASS_NAMES[idx.item()]
    res = {
        "id": str(uuid.uuid4())[:8],
        "prediction": raw_class, # Friendly name mapping can be added here
        "confidence": round(conf.item() * 100, 2),
        "timestamp": datetime.now().isoformat(),
        "age": metadata.age, "gender": metadata.gender, "location": metadata.location
    }
    return {**res, "history_item": res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
