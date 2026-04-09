Render & Railway Quick Deploy Guide for OncoDetect Backend

This guide walks through two free/easy options to host the `sc_app` FastAPI backend.

Prerequisites
- GitHub account
- Docker (optional, for local testing)
- `sc_app` repo pushed to GitHub

Common notes
- The server expects `edgefusion_v3_hybrid_champion.pth` present in the app root. Commit it to the repo (not ideal for secrets) or upload it to the host's file storage or use an object store and update `MODEL_SAVE_PATH`.
- Set environment variables in the hosting dashboard: `SMTP_*`, `TWILIO_*`, `LLM_API_URL`, `LLM_API_KEY`, `REQUIRE_SMTP`, `DEVICE` (set to `cpu` on free tiers).

Option A — Render (very quick)
1. Push `sc_app` to GitHub (if not already):

```bash
cd sc_app
git init
git add .
git commit -m "OncoDetect API"
git branch -M main
# add your remote and push
git remote add origin https://github.com/YOUR_USER/oncodetect-backend.git
git push -u origin main
```

2. Create a new Web Service on Render:
- Go to https://render.com
- Click "New+" → "Web Service"
- Connect your GitHub repo and select the `sc_app` repo
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment: Python 3

3. Set Environment Variables in Render dashboard:
- `DEVICE=cpu`
- `MODEL_SAVE_PATH=edgefusion_v3_hybrid_champion.pth`
- `REQUIRE_SMTP=0` (or `1` if you will provide SMTP)
- Any `SMTP_*` or `TWILIO_*` values your app needs

4. Deploy and verify:
- Visit `https://<your-service>.onrender.com/docs`
- Test endpoints like `/health` and `/predict` (use sample data)

Notes:
- Free plan may put the service to sleep — cold starts expected.
- For larger model sizes or faster inference, upgrade to a paid plan with more CPU/GPU.

Option B — Railway (also easy)
1. Push `sc_app` to GitHub (same as Render step above).
2. On https://railway.app, create a new project and connect your GitHub repo.
3. Use the default deploy settings — Railway detects Python projects. If needed, set build command to `pip install -r requirements.txt` and start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Add Environment Variables in Railway project settings (same list as Render).
5. Deploy and verify at the Railway-provided URL (visit `/docs`).

Extra: Using Docker on Render/Railway
- Both platforms support container deployments. If you prefer to use your `Dockerfile`, set up a Docker-powered service and push to GitHub. Render will build the Docker image and run it.

Troubleshooting
- If the model file is too large for platform file limits, upload model to an object storage (S3) and modify `main.py` to download model at startup if not present.
- If you see `CUDA` errors on Render, set `DEVICE=cpu` or ensure GPU runtime is selected (paid).
- For LLM endpoints: add `LLM_API_URL` and `LLM_API_KEY` (or leave blank to use local KB fallback).

Security/Prod notes
- Replace `CORS` wildcard with specific allowed origins in production.
- Do not commit secrets to the repo. Use host environment variables or secret storage.
- Consider moving the model to an artifact storage or model server for large-scale deployments.

"Quick test" curl (replace URL):

```bash
curl -X GET "https://YOUR_HOST/docs"
```