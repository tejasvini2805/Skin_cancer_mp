PWABuilder Quick APK Guide (free) — OncoDetect Frontend

Goal: Generate a simple Android APK from the hosted OncoDetect frontend (PWA) using PWABuilder.
This is fast and free for basic usage — good for MVP apps that call your hosted API.

Prerequisites
- Deploy the frontend to a public HTTPS URL (recommended hosts: Vercel, Netlify, GitHub Pages + Cloudflare Pages). PWABuilder needs a public HTTPS URL.
- PWA support already added (manifest.json and service worker). Verify `https://YOUR_SITE/manifest.json` works.

Steps
1. Build and deploy the frontend

```bash
# from oncodetect-ui
npm run build
# follow your host docs (Vercel/Netlify recommended)
```

2. Run PWABuilder
- Open https://www.pwabuilder.com
- Enter your hosted site URL (e.g., `https://oncodetect-yourname.vercel.app`) and click "Start"
- PWABuilder will analyze your PWA and suggest improvements. Fix any required items (icons, HTTPS, manifest completeness).

3. Generate an Android package
- In PWABuilder, choose "Android" → "Build Service"
- Follow the guided flow. PWABuilder offers two options:
  - Build using their cloud builder (fast) — they provide an APK or AAB.
  - Download a generated project you can open in Android Studio (more control).

4. Signing and Publishing
- For a test APK you can use the unsigned APK produced and `adb install` on a device (Android 8+ may require enabling installs from unknown sources).
- For Play Store publishing, generate a signed AAB (Android App Bundle) and follow Google Play Console steps.

Quick local test (install on device using ADB):

```bash
# assuming device connected and adb in PATH
adb install path/to/generated.apk
```

Tips and caveats
- Because inference runs server-side, ensure the API is reachable from mobile (public URL). Update `VITE_API_URL` to point to the deployed API and rebuild if you host a static build.
- If you used `VITE_API_URL` set at build time, rebuild the site with the correct env prior to hosting.
- PWABuilder cloud builds are free for simple APKs. For Play Store production builds you may want to open the project in Android Studio and sign the AAB properly.

Alternative: Capacitor (more control)
- If you want a native wrapper and access to native APIs, use Capacitor to wrap the built site. That requires Android Studio locally to produce the signed APK/AAB.
- PWABuilder is faster and fully free for quick distribution.

I'll produce a short checklist to publish the frontend (host + set env var) if you want to proceed with a specific host (Vercel/Netlify/GitHub Pages).