# Frontend Deployment Guide for Vercel/Netlify

## 🚀 Deploy to Vercel (Recommended - 2 minutes)

### Step 1: Push to GitHub
```bash
cd oncodetect-ui
git init
git add .
git commit -m "Initial commit: OncoDetect UI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/oncodetect-ui.git
git push -u origin main
```

### Step 2: Deploy to Vercel
1. Go to https://vercel.com/import
2. Sign in with GitHub
3. Select `oncodetect-ui` repository
4. Click "Import"
5. Wait 1-2 minutes
6. Your site is live! 🎉

---

## ⚙️ Configure Environment Variables

### In Vercel Dashboard:
1. Go to Settings → Environment Variables
2. Add:
```
VITE_API_URL=https://your-backend-api.com
```

### Start using in React:
```javascript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function predictLesion(image, metadata) {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image, metadata })
  });
  return response.json();
}
```

---

## 🚀 Deploy to Netlify (Alternative)

1. Go to https://app.netlify.com/start
2. Connect your GitHub repo
3. Set build command: `npm run build`
4. Set publish directory: `dist`
5. Deploy!

---

## 📦 What Happens Automatically

✅ Runs `npm install`
✅ Runs `npm run build` (creates optimized build in `dist/`)
✅ Deploys to CDN with HTTPS
✅ Auto-deploys on every git push

---

## 🔧 Local Testing Before Deploying

```bash
cd oncodetect-ui

# Install dependencies
npm install

# Test local dev server
npm run dev
# Visit: http://localhost:5173

# Build for production
npm run build

# Test production build
npm run preview
# Visit: http://localhost:4173
```

---

## 📊 Your Live URLs

After deployment:
- **Website**: `https://oncodetect-ui.vercel.app` (or your custom domain)
- **API**: `https://oncodetect-api.onrender.com`
- **API Docs**: `https://oncodetect-api.onrender.com/docs`

---

## 🎨 Custom Domain (Optional)

1. In Vercel Settings → Domains
2. Add your domain (e.g., oncodetect.com)
3. Update DNS records as instructed
4. HTTPS automatically configured!

---

**Next**: After deploying, test by submitting an image through your frontend!
