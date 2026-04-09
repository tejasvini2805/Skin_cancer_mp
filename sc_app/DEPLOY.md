# OncoDetect Backend - Production Deployment Guide

## 🚀 Quick Deploy to Render (5 minutes, Free)

### Step 1: Push to GitHub
```bash
cd sc_app
git init
git add .
git commit -m "Initial commit: OncoDetect API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/oncodetect-backend.git
git push -u origin main
```

### Step 2: Deploy to Render
1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Select your GitHub repo
5. Fill in:
   - **Name**: oncodetect-api
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Click "Create Web Service"
7. Done! Your API is live at `https://oncodetect-api.onrender.com`

---

## 🚀 Deploy to Azure (More robust)

### Step 1: Install Azure CLI
```bash
# Download from https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows
az login
```

### Step 2: Create Resource Group
```bash
az group create --name oncodetect-rg --location eastus
```

### Step 3: Create App Service Plan (Free tier)
```bash
az appservice plan create --name oncodetect-plan \
  --resource-group oncodetect-rg --sku F1 --is-linux
```

### Step 4: Create Web App
```bash
az webapp create --resource-group oncodetect-rg \
  --plan oncodetect-plan --name oncodetect-api \
  --runtime "PYTHON:3.11"
```

### Step 5: Deploy
```bash
cd sc_app
az webapp deployment source config-zip \
  --resource-group oncodetect-rg --name oncodetect-api \
  --src-path deployment.zip
```

---

## 🐳 Docker Deployment (Advanced)

### Build & Test Locally
```bash
docker build -t oncodetect-api .
docker run -p 8000:8000 oncodetect-api
# Test: http://localhost:8000/docs
```

### Deploy to Cloud (e.g., Railway)
1. Create account on https://railway.app
2. Connect GitHub repo
3. Railway auto-deploys!

---

## ✅ Verify Deployment

Test your deployed API:
```bash
curl -X POST "https://YOUR-DEPLOYED-URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "BASE64_ENCODED_IMAGE",
    "metadata": {
      "age": "45",
      "gender": "male",
      "location": "Head & Neck"
    }
  }'
```

Or visit: `https://YOUR-DEPLOYED-URL/docs` (Swagger UI)

---

## 📊 Environment Variables to Set

In your deployment dashboard, add:
```
DEVICE=cpu
MODEL_SAVE_PATH=edgefusion_v3_hybrid_champion.pth
```

---

## 💡 Cost-Saving Tips

✅ Use **Render Free Tier** (~30 seconds spinup, acceptable for MVP)
✅ Use **Railway Free Tier** ($5 credit/month, free for first deployment)
✅ Use **Azure Free Tier** (750 hours free per month)
✅ Set up **auto-scaling** or **sleep mode** for free tier

---

**Recommended**: Start with **Render** (simplest) or **Railway** (best free tier), then move to Azure as you scale.
