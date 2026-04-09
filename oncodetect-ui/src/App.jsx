import React, { useState, useRef, useEffect } from 'react';
// Using Twilio / backend SMS flow; Firebase removed to simplify auth
import {
  ShieldCheck, Mail, Camera, AlertCircle, CheckCircle2, RefreshCcw,
  ChevronRight, Activity, User, Calendar, Lock, ArrowLeft, Info, Save,
  BarChart3, History, Download, Trash2, Eye, EyeOff, Loader, FileText, MessageSquare,
  LogOut, Edit2, Phone as PhoneIcon, MapPin, Heart, Settings,
  Mail as MailIcon // <--- MAKE SURE THIS IS HERE
} from 'lucide-react';

// API Configuration
// Default to backend port 8000 (uvicorn default). You can override with VITE_API_URL.
// When accessing from a mobile device, replace 'localhost' with the device-visible
// hostname (window.location.hostname) so API calls reach the dev machine.
const RAW_API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
function resolveApiBaseUrl(rawUrl) {
  try {
    const url = new URL(rawUrl);
    // If the configured host is localhost/127.0.0.1 but the client is remote,
    // replace with the client's hostname to reach the dev server over LAN.
    if ((url.hostname === 'localhost' || url.hostname === '127.0.0.1') && typeof window !== 'undefined') {
      const host = window.location.hostname;
      // Only replace when host is not localhost and looks like an IP
      if (host && host !== 'localhost' && host !== '127.0.0.1') {
        url.hostname = host;
        return url.toString();
      }
    }
    // Ensure no trailing slash to avoid double-slash when concatenating
    return url.toString().replace(/\/$/, '');
  } catch (e) {
    return rawUrl.replace(/\/$/, '');
  }
}
const API_BASE_URL = resolveApiBaseUrl(RAW_API_URL);
const API_KEY = import.meta.env.VITE_API_KEY || '';

function apiFetch(url, opts = {}) {
  const providedHeaders = opts.headers || {};
  const keyHeader = API_KEY ? { 'x-api-key': API_KEY } : {};
  const headers = { ...providedHeaders, ...keyHeader };
  return fetch(url, { ...opts, headers });
}
// When set to '1' in .env, frontend will use server-side SMS (Twilio) instead of Firebase
const USE_SERVER_SMS = import.meta.env.VITE_USE_SERVER_SMS === '1';

console.log(`🔌 API URL: ${API_BASE_URL}`);

// Utility: friendly timestamp formatting (ISO or epoch)
function formatTimestamp(ts) {
  if (!ts) return 'N/A';
  try {
    const d = new Date(ts);
    if (isNaN(d.getTime())) return String(ts);
    return d.toLocaleString();
  } catch (e) {
    return String(ts);
  }
}

// ============================================================================
// UTILS
// ============================================================================

// Helper: ensure Recaptcha container exists and create a RecaptchaVerifier safely
function ensureRecaptcha(containerId) {
  // When using server-side SMS (Twilio) we don't use Firebase Recaptcha.
  if (USE_SERVER_SMS) return null;
  // If a future client-side recaptcha flow is desired, implement here.
  return null;
}

// Fallback: request backend to send SMS (dev / Twilio path)
async function requestBackendSms(phone) {
  try {
    // Build absolute URL correctly (handles trailing slashes)
    const url = new URL('/send_sms_code', API_BASE_URL).toString();
    const res = await apiFetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone_number: phone }),
      mode: 'cors'
    });

    if (!res.ok) {
      let txt = '';
      try { txt = await res.text(); } catch (_) { txt = res.statusText; }
      throw new Error(`SMS API error ${res.status} at ${url}: ${txt}`);
    }
    return res.json();
  } catch (e) {
    // Network-level failure (DNS, unreachable, CORS preflight failure) surfaces here
    throw new Error(`Failed to reach backend at ${API_BASE_URL} (requested ${phone}): ${e.message}`);
  }
}

async function predictLesion(imageBase64, metadata) {
  // Convert data URL to Blob
  function dataURLtoBlob(dataurl) {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
  }

  const form = new FormData();
  form.append('file', dataURLtoBlob(imageBase64), 'upload.png');
  form.append('age', metadata.age);
  form.append('gender', metadata.gender);
  form.append('bodyLocationLabel', metadata.location);

  const predictUrl = new URL('/predict', API_BASE_URL).toString();
  const response = await apiFetch(predictUrl, {
    method: 'POST',
    body: form
  });

  if (!response.ok) {
    let errText;
    try { errText = await response.json(); } catch (e) { errText = await response.text(); }
    throw new Error(`Prediction API ${predictUrl} error ${response.status}: ${errText.detail || errText || 'Prediction failed'}`);
  }

  return response.json();
}

async function fetchHistory() {
  // If a user is logged in, prefer local per-user history stored in localStorage
  try {
    const current = JSON.parse(localStorage.getItem('currentUser') || 'null');
    if (current && current.phone) {
      const users = JSON.parse(localStorage.getItem('users') || '{}');
      const u = users[current.phone] || {};
      if (Array.isArray(u.history)) {
        // Return a copy to mimic API shape
        return [...u.history].sort((a,b) => new Date(b.timestamp) - new Date(a.timestamp));
      }
    }
  } catch (e) {
    // fallthrough to server fetch on any parse error
    console.warn('Local history read failed', e);
  }

  const response = await apiFetch(new URL('/history?limit=100', API_BASE_URL).toString());
  if (!response.ok) throw new Error('Failed to fetch history');
  return response.json();
}

async function fetchStats() {
  // Compute per-user stats from localStorage when a user is logged in
  try {
    const current = JSON.parse(localStorage.getItem('currentUser') || 'null');
    if (current && current.phone) {
      const users = JSON.parse(localStorage.getItem('users') || '{}');
      const u = users[current.phone] || {};
      const hist = Array.isArray(u.history) ? u.history : [];
      if (hist.length > 0) {
        const predictions = hist.map(h => h.prediction);
        const confidence_scores = hist.map(h => h.confidence || 0);
        const counts = {};
        predictions.forEach(p => counts[p] = (counts[p] || 0) + 1);
        const top = Object.entries(counts).sort((a,b)=>b[1]-a[1])[0];
        const avg = confidence_scores.length ? Math.round((confidence_scores.reduce((a,b)=>a+b,0)/confidence_scores.length)*100)/100 : 0;
        return {
          total_predictions: hist.length,
          top_prediction: top ? top[0] : null,
          avg_confidence: avg,
          prediction_breakdown: counts
        };
      }
    }
  } catch (e) {
    console.warn('Local stats computation failed', e);
  }

  const response = await apiFetch(new URL('/stats', API_BASE_URL).toString());
  if (!response.ok) throw new Error('Failed to fetch stats');
  return response.json();
}

async function sendNotification(email, subject, body) {
  const res = await apiFetch(new URL('/notifications/send', API_BASE_URL).toString(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, subject, body })
  });
  if (!res.ok) throw new Error('Failed to send notification');
  return res.json();
}

async function deletePrediction(predictionId) {
  const response = await apiFetch(new URL(`/history/${encodeURIComponent(predictionId)}`, API_BASE_URL).toString(), {
    method: 'DELETE'
  });
  if (!response.ok) throw new Error('Failed to delete');
  return response.json();
}

function generateReport(prediction, userData) {
  // Support both full-result and history_item shapes
  const displayPrediction = prediction?.prediction ?? prediction?.history_item?.prediction ?? 'Unknown';
  const displayConfidence = prediction?.confidence ?? prediction?.history_item?.confidence ?? 0;
  const displayId = prediction?.id ?? prediction?.history_item?.id ?? 'n/a';
  const displayBreakdown = prediction?.confidence_breakdown ?? prediction?.history_item?.confidence_breakdown ?? {};

  const timestamp = new Date().toLocaleString();
  const report = `
======================== ONCODETECT REPORT ========================
Generated: ${timestamp}
Report ID: ${displayId}

DISCLAIMER:
This report is generated by AI analysis and is NOT a substitute for 
professional medical diagnosis. Always consult a dermatologist.

====================================================================
PATIENT INFORMATION
====================================================================
Age: ${userData.age} years
Gender: ${userData.gender}
Lesion Location: ${userData.location}

====================================================================
ANALYSIS RESULTS
====================================================================
Predicted Condition: ${displayPrediction}
Confidence Score: ${displayConfidence}%
Risk Level: ${
    displayPrediction.toLowerCase().includes('melanoma') ? 'HIGH' :
    displayPrediction.toLowerCase().includes('carcinoma') ? 'MEDIUM' :
    'LOW'
  }

====================================================================
CONFIDENCE BREAKDOWN
====================================================================
${Object.entries(displayBreakdown)
  .sort(([,a], [,b]) => b - a)
  .map(([cls, conf]) => `${cls}: ${conf}%`)
  .join('\n')}

====================================================================
RECOMMENDATIONS
====================================================================
1. Schedule an appointment with a dermatologist at your earliest 
   convenience for professional evaluation.
2. Do not delay seeking medical attention if you have concerns.
3. Keep this report for your medical records.
4. Mention this AI analysis to your healthcare provider for context.

====================================================================
NEXT STEPS
====================================================================
• Print or save this report for your records
• Schedule a dermatology consultation
• Avoid self-medication or home remedies
• Monitor the lesion for any changes

Report Generated by OncoDetect v1.0
For more information, visit: https://oncodetect.com
====================================================================
  `;
  return report;
}

function downloadReport(prediction, userData) {
  const report = generateReport(prediction, userData);
  const element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(report));
  const displayId = prediction?.id ?? prediction?.history_item?.id ?? 'n-a';
  element.setAttribute('download', `OncoDetect_Report_${displayId}.txt`);
  element.style.display = 'none';
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
}


// ============================================================================
// HOMEPAGE (Public Landing) 
// ============================================================================

const HomePage = ({ navigate }) => (
  <div className="min-h-screen bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 text-white flex flex-col">
    {/* Header */}
    <div className="p-6 flex justify-between items-center">
      <div className="flex items-center gap-2">
        <Activity className="w-8 h-8" />
        <span className="text-2xl font-bold">OncoDetect</span>
      </div>
      <div className="flex gap-3">
        <button 
          onClick={() => navigate('login')}
          className="px-6 py-2 bg-white text-blue-600 rounded-lg font-semibold hover:bg-gray-100 transition"
        >
          Sign In
        </button>
        <button 
          onClick={() => navigate('signup')}
          className="px-6 py-2 bg-blue-500 text-white rounded-lg font-semibold hover:bg-blue-400 transition border-2 border-white"
        >
          Sign Up
        </button>
      </div>
    </div>

    {/* Hero */}
    <div className="flex-1 flex flex-col items-center justify-center text-center px-6 py-20">
      <h1 className="text-6xl font-bold mb-6">AI-Powered Skin Assessment</h1>
  <p className="text-xl text-gray-100 mb-4 max-w-full sm:max-w-2xl">
        Early detection saves lives. OncoDetect uses advanced machine learning to help identify potential skin lesions.
      </p>
  <p className="text-gray-200 mb-12 max-w-full sm:max-w-2xl">
        ⚠️ Always consult a dermatologist. This tool is for screening support only.
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mb-12">
        <div className="bg-white bg-opacity-10 backdrop-blur p-6 rounded-xl">
          <Activity className="w-12 h-12 mx-auto mb-4" />
          <h3 className="font-bold text-lg mb-2">Fast Analysis</h3>
          <p className="text-sm text-gray-100">Get AI predictions in seconds</p>
        </div>
        <div className="bg-white bg-opacity-10 backdrop-blur p-6 rounded-xl">
          <ShieldCheck className="w-12 h-12 mx-auto mb-4" />
          <h3 className="font-bold text-lg mb-2">Secure</h3>
          <p className="text-sm text-gray-100">Your data stays private</p>
        </div>
        <div className="bg-white bg-opacity-10 backdrop-blur p-6 rounded-xl">
          <BarChart3 className="w-12 h-12 mx-auto mb-4" />
          <h3 className="font-bold text-lg mb-2">Track History</h3>
          <p className="text-sm text-gray-100">Monitor your skin health</p>
        </div>
      </div>

      <button 
        onClick={() => navigate('signup')}
        className="px-8 py-4 bg-white text-blue-600 rounded-lg font-bold text-lg hover:bg-gray-100 transition flex items-center gap-2"
      >
        Get Started Free
        <ChevronRight className="w-6 h-6" />
      </button>
    </div>

    {/* Footer */}
    <div className="p-6 text-center text-gray-200 border-t border-white border-opacity-20">
      <p>© 2026 OncoDetect. Medical Disclaimer: Not a substitute for professional diagnosis.</p>
    </div>
  </div>
);

// ============================================================================
// SIGN UP PAGE
// ============================================================================


const COUNTRY_OPTIONS = [
	{ code: '+91', name: 'India' },
	{ code: '+1', name: 'United States' },
	{ code: '+44', name: 'United Kingdom' },
	{ code: '+61', name: 'Australia' },
	{ code: '+49', name: 'Germany' }
];

const SignUp = ({ navigate }) => {
  const [form, setForm] = useState({ name: '', countryCode: '+91', phone: '', password: '', confirmPassword: '' });
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const handleSignUp = async () => {
    setError('');
    setInfo('');

    // 1. Validation
    if (!form.name || !form.phone || !form.password) {
      setError('Please fill in all fields.');
      return;
    }

    if (form.password !== form.confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    // 2. Format Phone Number correctly for Twilio (E.164 format)
    // Extracts digits and adds the + symbol: e.g., +919849858407
    const digits = form.phone.replace(/\D/g, '');
    const phoneKey = `${form.countryCode}${digits}`;

    // 3. Store user details temporarily in LocalStorage
    localStorage.setItem('pendingPhone', phoneKey);
    localStorage.setItem('pendingUserData', JSON.stringify({
      name: form.name,
      phone: phoneKey,
      password: form.password
    }));

    try {
      // 4. Request backend to send SMS via Twilio Proxy
      // Note: requestBackendSms is already defined in your UTILS section
      const resp = await requestBackendSms(phoneKey);
	  if (resp.code) {
  alert("Demo OTP: " + resp.code);
}

      // 5. Handle the Response from main.py
      if (resp && (resp.status === 'code_sent' || resp.status === 'success')) {
        setInfo('Verification code sent via Twilio! Please check your phone.');

        // Navigate to verification screen after a short delay
        setTimeout(() => navigate('verifyphone'), 1200);
      } else {
        // If backend sends an error detail, show it; otherwise use fallback
        throw new Error(resp?.detail || 'Twilio gateway failed to send SMS.');
      }
    } catch (err) {
      console.error("Sign Up / Twilio Error:", err);
      setError(`Authentication Failed: ${err.message}`);
    }
  };
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white flex items-center justify-center p-6">
      <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full">
        <button onClick={() => navigate('homepage')} className="text-gray-600 hover:text-gray-900 mb-6">
          <ArrowLeft className="w-6 h-6" />
        </button>
        <h2 className="text-3xl font-bold text-gray-900 mb-6">Create Account</h2>

        {error && <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-6 text-sm flex gap-2"><AlertCircle className="w-5 h-5"/>{error}</div>}
        {info && <div className="bg-green-50 text-green-700 p-4 rounded-lg mb-6 text-sm">{info}</div>}

        <div className="space-y-4">
          <input type="text" placeholder="Full Name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} className="w-full px-4 py-3 border-2 rounded-lg focus:border-blue-500 outline-none" />
          <div className="flex gap-2">
            <select value={form.countryCode} onChange={(e) => setForm({ ...form, countryCode: e.target.value })} className="w-28 sm:w-32 px-3 py-3 border-2 rounded-lg bg-white">
              {COUNTRY_OPTIONS.map(c => <option key={c.code} value={c.code}>{c.code}</option>)}
            </select>
            <input type="tel" placeholder="Mobile Number" value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} className="flex-1 px-4 py-3 border-2 rounded-lg focus:border-blue-500 outline-none" />
          </div>
          <input type="password" placeholder="Password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} className="w-full px-4 py-3 border-2 rounded-lg focus:border-blue-500 outline-none" />
          <input type="password" placeholder="Confirm Password" value={form.confirmPassword} onChange={(e) => setForm({ ...form, confirmPassword: e.target.value })} className="w-full px-4 py-3 border-2 rounded-lg focus:border-blue-500 outline-none" />
        </div>

        <button onClick={handleSignUp} className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold mt-8 hover:bg-blue-700 transition">
          Send OTP
        </button>
      </div>
    </div>
  );
};
// ============================================================================
// PHONE VERIFICATION PAGE
// ============================================================================

const VerifyPhone = ({ navigate, setCurrentUser }) => {
  const [code, setCode] = useState('');
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');
  const phone = localStorage.getItem('pendingPhone');

  const handleVerify = async () => {
    setError('');
    setInfo('');

    try {
      // 1. Call Backend to verify the code against Twilio session/cache
      const verifyUrl = new URL('/verify_phone_code', API_BASE_URL).toString();
      const res = await apiFetch(verifyUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone_number: phone, code: code })
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Invalid OTP. Please try again.');
      }

      // 2. Success! Mark user verified locally
      const pendingUser = JSON.parse(localStorage.getItem('pendingUserData'));
      if (pendingUser) {
        const users = JSON.parse(localStorage.getItem('users') || '{}');
        users[pendingUser.phone] = { ...pendingUser, verified: true, createdAt: new Date().toISOString() };
        localStorage.setItem('users', JSON.stringify(users));
        localStorage.setItem('currentUser', JSON.stringify(users[pendingUser.phone]));
        if (typeof setCurrentUser === 'function') setCurrentUser(users[pendingUser.phone]);
      }

      // 3. Cleanup and Enter Dashboard
      localStorage.removeItem('pendingUserData');
      localStorage.removeItem('pendingPhone');
      setInfo('Phone Verified! Welcome.');
      setTimeout(() => navigate('dashboard'), 800);

    } catch (err) {
      setError(err.message);
    }
  };

  const handleResend = async () => {
    setError('');
    setInfo('');
    try {
      const resp = await requestBackendSms(phone);
	  if (resp.code) {
  alert("Resent OTP: " + resp.code);
}
      if (resp && (resp.status === 'code_sent' || resp.status === 'success')) {
        setInfo('A new OTP has been sent via Twilio.');
      } else {
        throw new Error('Resend failed');
      }
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white flex items-center justify-center p-6">
      <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full text-center">
        <MailIcon className="w-16 h-16 text-blue-600 mx-auto mb-6" />
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Verify Phone</h2>
        <p className="text-gray-600 mb-6">Enter the code sent to <br/><b>{phone}</b></p>

        {error && <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-4 text-sm text-left">{error}</div>}
        {info && <div className="bg-green-50 text-green-700 p-4 rounded-lg mb-4 text-sm text-left">{info}</div>}

        <input type="text" maxLength="6" placeholder="000000" value={code} onChange={(e) => setCode(e.target.value.replace(/\D/g, ''))} className="w-full px-4 py-3 border-2 rounded-lg text-center text-2xl tracking-widest mb-6 font-mono" />
        <button onClick={handleVerify} className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition">Verify</button>
        <p className="text-gray-600 mt-6 text-sm">Didn't receive code? <button onClick={handleResend} className="text-blue-600 font-semibold hover:underline">Resend</button></p>
      </div>
    </div>
  );
};

// ============================================================================
// LOGIN PAGE
// ============================================================================

const Login = ({ navigate, setCurrentUser }) => {
	const [form, setForm] = useState({ countryCode: '+91', phone: '', password: '' });
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const handleLogin = async () => {
		setError('');
		setInfo('');

		if (!form.phone || !form.password) {
			setError('Phone and password required');
	  return;
	}

		// Compose full phone
		const digits = (form.phone || '').replace(/\D/g, '');
		let phoneKey = `${form.countryCode}${digits}`;
		if (!phoneKey.startsWith('+')) phoneKey = `+${phoneKey}`;

		const users = JSON.parse(localStorage.getItem('users') || '{}');
		const user = users[phoneKey];

		if (!user || user.password !== form.password) {
			setError('Invalid credentials');
			return;
		}

		if (!user.verified) {
    try {
        // Use your backend Twilio proxy instead of Firebase auth
        const resp = await requestBackendSms(phoneKey);
        if (!resp || (resp.status !== 'code_sent' && resp.status !== 'success'))
            throw new Error(resp?.detail || 'SMS send failed');

        localStorage.setItem('pendingPhone', phoneKey);
        setInfo('Account needs verification. OTP sent via Twilio.');
        setTimeout(() => navigate('verifyphone'), 800);
    } catch (err) {
        setError(`Failed to send verification code: ${err.message}`);
    }
    return;

		}

		localStorage.setItem('currentUser', JSON.stringify({
			name: user.name,
			phone: user.phone,
			createdAt: user.createdAt
		}));

		setCurrentUser({
			name: user.name,
			phone: user.phone,
			createdAt: user.createdAt
		});

	navigate('dashboard');
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white flex items-center justify-center p-6">
      <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full">
        <button onClick={() => navigate('homepage')} className="text-gray-600 hover:text-gray-900 mb-6">
          <ArrowLeft className="w-6 h-6" />
        </button>

        <h2 className="text-3xl font-bold text-gray-900 mb-6">Sign In</h2>

				{error && (
          <div className="bg-red-50 border border-red-300 rounded-lg p-4 mb-6 flex gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

				{info && (
					<div className="bg-green-50 border border-green-300 rounded-lg p-4 mb-6 text-sm text-green-700">
						{info}
					</div>
				)}

				<div className="space-y-4">
					<div>
						<label className="block text-sm font-semibold text-gray-700 mb-2">Phone</label>
						<div className="flex gap-2">
              <select
                value={form.countryCode}
                onChange={(e) => setForm({...form, countryCode: e.target.value})}
                className="w-28 sm:w-32 px-3 py-3 border-2 border-gray-200 rounded-lg focus:border-blue-500 focus:outline-none bg-white"
              >
								{COUNTRY_OPTIONS.map(c => (
									<option key={c.code} value={c.code}>{c.name} ({c.code})</option>
								))}
							</select>
							<input
								type="tel"
								placeholder="9849858407"
								value={form.phone}
								onChange={(e) => setForm({...form, phone: e.target.value})}
								className="flex-1 px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-blue-500 focus:outline-none"
							/>
						</div>
					</div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Password</label>
            <div className="relative">
              <input
                type={showPassword ? 'text' : 'password'}
                placeholder="••••••"
                value={form.password}
                onChange={(e) => setForm({...form, password: e.target.value})}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-blue-500 focus:outline-none"
              />
              <button
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-3 text-gray-600"
              >
                {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            </div>
          </div>
				</div>

				{/* invisible reCAPTCHA container for login-triggered verifies */}
				<div id="recaptcha-container-login"></div>

				<button
					onClick={handleLogin}
					className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold mt-8 hover:bg-blue-700 transition"
				>
					Sign In
				</button>

        <p className="text-center text-gray-600 mt-6">
          Don't have an account?{' '}
          <button onClick={() => navigate('signup')} className="text-blue-600 font-semibold hover:underline">
            Create one
          </button>
        </p>
      </div>
    </div>
  );
};

// ============================================================================
// PROFILE PAGE
// ============================================================================

const Profile = ({ currentUser, navigate, setCurrentUser }) => {
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState({ name: currentUser.name });

  const handleSaveProfile = () => {
		const users = JSON.parse(localStorage.getItem('users') || '{}');
		users[currentUser.phone].name = form.name;
    localStorage.setItem('users', JSON.stringify(users));

    const updatedUser = {...currentUser, name: form.name};
    localStorage.setItem('currentUser', JSON.stringify(updatedUser));
    setCurrentUser(updatedUser);
    setEditing(false);
  };

  const handleLogout = () => {
    localStorage.removeItem('currentUser');
    setCurrentUser(null);
    navigate('homepage');
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <div className="max-w-full sm:max-w-2xl mx-auto p-6">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-gray-900">My Profile</h1>
          <button
            onClick={() => navigate('dashboard')}
            className="text-gray-600 hover:text-gray-900"
          >
            <ArrowLeft className="w-6 h-6" />
          </button>
        </div>

        {/* Profile Card */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-6">
          <div className="flex items-center gap-6 mb-8 pb-8 border-b">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center">
              <User className="w-10 h-10 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900">{currentUser.name}</h2>
	<p className="text-gray-600">{currentUser.phone}</p>
              <p className="text-sm text-gray-500 mt-2">
                Member since {formatTimestamp(currentUser.createdAt)}
              </p>
            </div>
          </div>

          {/* Profile Fields */}
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Full Name</label>
              {editing ? (
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => setForm({...form, name: e.target.value})}
                  className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:border-blue-500 focus:outline-none"
                />
              ) : (
                <p className="text-gray-900 py-2">{currentUser.name}</p>
              )}
            </div>

		<div>
			<label className="block text-sm font-semibold text-gray-700 mb-2">Phone (Cannot change)</label>
			<p className="text-gray-500 py-2">{currentUser.phone}</p>
		</div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4 mt-8">
            {editing ? (
              <>
                <button
                  onClick={() => setEditing(false)}
                  className="flex-1 border-2 border-gray-300 text-gray-700 py-3 rounded-lg font-semibold hover:bg-gray-50 transition"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveProfile}
                  className="flex-1 bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition flex items-center justify-center gap-2"
                >
                  <Save className="w-5 h-5" />
                  Save Changes
                </button>
              </>
            ) : (
              <button
                onClick={() => setEditing(true)}
                className="flex-1 border-2 border-blue-300 text-blue-600 py-3 rounded-lg font-semibold hover:bg-blue-50 transition flex items-center justify-center gap-2"
              >
                <Edit2 className="w-5 h-5" />
                Edit Profile
              </button>
            )}
          </div>
        </div>

        {/* Logout Button */}
        <button
          onClick={handleLogout}
          className="w-full bg-red-600 text-white py-3 rounded-lg font-semibold hover:bg-red-700 transition flex items-center justify-center gap-2"
        >
          <LogOut className="w-5 h-5" />
          Sign Out
        </button>
      </div>
    </div>
  );
};

// ============================================================================
// DETAILED HISTORY PAGE
// ============================================================================

const DetailedHistory = ({ navigate, currentUser }) => {
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    const loadData = async () => {
      try {
        const [historyData, statsData] = await Promise.all([
          fetchHistory(),
          fetchStats()
        ]);
        setHistory(historyData);
        setStats(statsData);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [currentUser]);

  const filteredHistory =
    filter === 'all'
      ? history
      : history.filter(h => h.prediction.toLowerCase().includes(filter.toLowerCase()));

  const handleDelete =async (id) => {
    try {
      await deletePrediction(id);
      setHistory(history.filter(h => h.id !== id));
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <div className="max-w-4xl mx-auto p-6">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Analysis History</h1>
          <button
            onClick={() => navigate('dashboard')}
            className="text-gray-600 hover:text-gray-900"
          >
            <ArrowLeft className="w-6 h-6" />
          </button>
        </div>

        {/* Summary Stats */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="bg-white rounded-lg shadow p-6 border-l-4 border-blue-500">
              <p className="text-gray-600 text-sm">Total Analyses</p>
              <p className="text-3xl font-bold text-blue-600">{stats.total_predictions}</p>
            </div>
            <div className="bg-white rounded-lg shadow p-6 border-l-4 border-green-500">
              <p className="text-gray-600 text-sm">Most Common</p>
              <p className="text-lg font-bold text-green-600 truncate">{stats.top_prediction || 'N/A'}</p>
            </div>
            <div className="bg-white rounded-lg shadow p-6 border-l-4 border-purple-500">
              <p className="text-gray-600 text-sm">Avg Confidence</p>
              <p className="text-3xl font-bold text-purple-600">{stats.avg_confidence}%</p>
            </div>
          </div>
        )}

        {/* Filter Buttons */}
        <div className="bg-white rounded-lg shadow p-4 mb-6">
          <p className="text-sm font-semibold text-gray-700 mb-3">Filter:</p>
          <div className="flex flex-wrap gap-2">
            {['all', 'Melanoma', 'Nevus', 'Carcinoma'].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-4 py-2 rounded-lg font-medium transition capitalize ${
                  filter === f
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {/* History List */}
        {loading ? (
          <div className="flex justify-center py-12">
            <Loader className="w-8 h-8 text-blue-600 animate-spin" />
          </div>
        ) : filteredHistory.length > 0 ? (
          <div className="space-y-4">
            {filteredHistory.map((item, idx) => (
              <div key={idx} className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-xl font-bold text-gray-900">{item.prediction}</h3>
                    <p className="text-sm text-gray-600 mt-1">
                      {formatTimestamp(item.timestamp)}{' '} 
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                  <button
                    onClick={() => handleDelete(item.id)}
                    className="p-2 text-gray-400 hover:text-red-600 transition"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div>
                    <p className="text-xs text-gray-600">Confidence</p>
                    <p className="text-lg font-bold text-blue-600">{item.confidence}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Age</p>
                    <p className="text-lg font-bold text-gray-900">{item.age} yrs</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Gender</p>
                    <p className="text-lg font-bold text-gray-900 capitalize">{item.gender}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Location</p>
                    <p className="text-sm font-bold text-gray-900">{item.location}</p>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded p-3">
                  <p className="text-xs font-semibold text-gray-700 mb-2">Confidence Breakdown:</p>
                  <div className="text-xs text-gray-600 space-y-1">
                    {Object.entries(item.confidence_breakdown || {})
                      .sort(([,a], [,b]) => b - a)
                      .slice(0, 3)
                      .map(([cls, conf]) => (
                        <p key={cls}>• {cls}: {conf}%</p>
                      ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12 bg-white rounded-lg">
            <History className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-600">No analyses found</p>
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// DASHBOARD (Main App After Login)
// ============================================================================

function Dashboard({ navigate, currentUser }) {
  const [stats, setStats] = useState({ total_predictions: 0, top_prediction: null, avg_confidence: 0 });
  const [historyCount, setHistoryCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);

  const load = async () => {
    try {
      const s = await fetchStats();
      const h = await fetchHistory();
      setStats(s || { total_predictions: 0, top_prediction: null, avg_confidence: 0 });
      setHistoryCount((h || []).length);
      setLastUpdated(new Date().toISOString());
    } catch (err) {
      console.error('Failed to load dashboard data', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [currentUser]);

  return (
  <div className="flex flex-col min-h-screen bg-white overflow-auto">
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-6 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Welcome, {currentUser.name}!</h1>
          <p className="text-blue-100">Ready for your next analysis?</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => navigate('profile')}
            className="p-3 bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg transition"
          >
            <User className="w-6 h-6" />
          </button>
          <button
            onClick={() => navigate('detailedhistory')}
            className="p-3 bg-white bg-opacity-20 hover:bg-opacity-30 rounded-lg transition"
          >
            <History className="w-6 h-6" />
          </button>
        </div>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center p-6 bg-gradient-to-b from-blue-50 via-white to-indigo-50">
        <div className="w-24 h-24 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-3xl flex items-center justify-center mb-6 shadow-xl">
          <Activity className="w-12 h-12 text-white" />
        </div>
        <h2 className="text-4xl font-bold text-gray-900 mb-3 text-center">OncoDetect</h2>
        <p className="text-gray-500 text-center text-lg mb-2 max-w-sm">AI-Powered Skin Assessment</p>
        <p className="text-xs text-gray-400 mb-6">Last updated: {formatTimestamp(lastUpdated)}</p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl w-full mb-8">
          <div className="bg-white rounded-lg shadow p-8 border-l-4 border-blue-500">
            <p className="text-gray-600 text-sm font-medium">Total Analyses</p>
            <p className="text-4xl font-extrabold text-blue-600 mt-2">{stats.total_predictions ?? historyCount}</p>
            <p className="text-xs text-gray-500 mt-1">(Local count: {historyCount})</p>
          </div>

          <div className="bg-white rounded-lg shadow p-8 border-l-4 border-green-500">
            <p className="text-gray-600 text-sm font-medium">Most Common</p>
            <p className="text-xl font-extrabold text-green-600 mt-2 truncate">{stats.top_prediction || 'N/A'}</p>
          </div>

          <div className="bg-white rounded-lg shadow p-8 border-l-4 border-purple-500">
            <p className="text-gray-600 text-sm font-medium">Avg Confidence</p>
            <p className="text-4xl font-extrabold text-purple-600 mt-2">{stats.avg_confidence ?? 0}%</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-md w-full">
          <button
            onClick={() => navigate('landing')}
            className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-7 rounded-2xl font-bold shadow-lg hover:shadow-xl transition flex items-center justify-center gap-3 text-lg"
          >
            <Camera className="w-6 h-6" />
            New Analysis
          </button>

          <button
            onClick={() => navigate('detailedhistory')}
            className="border-2 border-gray-300 text-gray-700 py-7 rounded-2xl font-bold hover:bg-gray-50 transition flex items-center justify-center gap-3 text-lg"
          >
            <History className="w-6 h-6" />
            View History
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// SUB-COMPONENTS (Updated)
// ============================================================================
const Assistant = ({ navigate, context }) => {
  const [messages, setMessages] = useState([
    { type: 'assistant', text: '👋 Hi! I\'m OncoDetect Assistant. I can help answer questions about skin lesions and our analysis. What would you like to know?' }
  ]);

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [lastAssistantPayload, setLastAssistantPayload] = useState(null);

  const quickQuestions = [
    { q: 'What is Melanoma?', a: 'Melanoma is the most dangerous type of skin cancer, arising from pigment-producing cells called melanocytes. Early detection and treatment are crucial for better outcomes.' },
    { q: 'When should I see a dermatologist?', a: 'You should see a dermatologist if you notice any new, changing, or concerning moles or lesions. If OncoDetect flags a HIGH risk, schedule an appointment immediately.' },
    { q: 'What are warning signs (ABCDE)?', a: 'Asymmetry: One half doesn\'t match the other. Border: Irregular or notched edges. Color: Multiple colors or uneven distribution. Diameter: Larger than a pencil eraser. Evolving: Changes over time.' },
    { q: 'Is OncoDetect accurate?', a: 'OncoDetect uses advanced AI but is not a substitute for professional medical diagnosis. It\'s a screening tool to help identify lesions that need professional evaluation.' },
    { q: 'How do I protect my skin?', a: 'Use SPF 30+ sunscreen daily, avoid peak sun (10am-4pm), wear protective clothing, and avoid tanning beds. Regular self-examination helps catch changes early.' },
  ];

  const handleQuestion = (answer) => {
    setMessages([...messages, 
      { type: 'user', text: answer },
      { type: 'assistant', text: answer }
    ]);
  };

  const askAssistant = async (question) => {
    if (!question) return;
    setLoading(true);
    setMessages(prev => [...prev, { type: 'user', text: question }]);

      try {
      // Build a robust history_item payload: accept full result, nested history_item, or history_item directly
      const derivedHistoryItem = context?.prediction?.history_item ?? context?.history_item ?? context?.prediction ?? null;
      const payload = {
        history_item: derivedHistoryItem,
        heatmap_summary: (context?.prediction?.heatmap || context?.history_item?.heatmap) ? 'Heatmap available' : '',
        question
      };

      const res = await apiFetch(new URL('/assistant/analyze', API_BASE_URL).toString(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      // save last payload so user can retry with longer timeout
      setLastAssistantPayload(payload);

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || 'Assistant API failed');
      }

      const data = await res.json();
      let assistant_text = '';
      if (data.assistant) {
        const a = data.assistant;
        assistant_text += a.summary ? a.summary + '\n\n' : '';
        assistant_text += a.explanation ? a.explanation + '\n\n' : '';
        assistant_text += 'Recommendation: ' + (a.recommendation || '') + '\n\n';
        if (a.sources && a.sources.length) assistant_text += 'Sources: ' + a.sources.join(', ');
      } else {
        assistant_text = data.assistant_text || (typeof data === 'string' ? data : JSON.stringify(data));
      }
      setMessages(prev => [...prev, { type: 'assistant', text: assistant_text }]);
    } catch (err) {
      setMessages(prev => [...prev, { type: 'assistant', text: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const retryLastWithTimeout = async () => {
    if (!lastAssistantPayload) return alert('No previous assistant request to retry');
    const payload = { ...lastAssistantPayload, llm_timeout: 60, llm_retries: 1 };
    setLoading(true);
    setMessages(prev => [...prev, { type: 'user', text: '(Retrying with longer timeout...)' }]);
    try {
      const res = await apiFetch(new URL('/assistant/analyze', API_BASE_URL).toString(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || 'Assistant API failed');
      }
      const data = await res.json();
      let assistant_text = '';
      if (data.assistant) {
        const a = data.assistant;
        assistant_text += a.summary ? a.summary + '\n\n' : '';
        assistant_text += a.explanation ? a.explanation + '\n\n' : '';
        assistant_text += 'Recommendation: ' + (a.recommendation || '') + '\n\n';
        if (a.sources && a.sources.length) assistant_text += 'Sources: ' + a.sources.join(', ');
      } else {
        assistant_text = data.assistant_text || (typeof data === 'string' ? data : JSON.stringify(data));
      }
      setMessages(prev => [...prev, { type: 'assistant', text: assistant_text }]);
    } catch (err) {
      setMessages(prev => [...prev, { type: 'assistant', text: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
  <div className="flex flex-col min-h-screen bg-white overflow-auto">
      <div className="flex items-center p-6 border-b">
        <button onClick={() => navigate('landing')} className="text-gray-600 hover:text-gray-900">
          <ArrowLeft className="w-6 h-6" />
        </button>
        <h2 className="flex-1 text-2xl font-bold text-gray-900 ml-4 flex items-center gap-2">
          <MessageSquare className="w-6 h-6" />
          Assistant
        </h2>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${
              msg.type === 'user' 
                ? 'bg-blue-600 text-white rounded-br-none' 
                : 'bg-gray-100 text-gray-900 rounded-bl-none'
            }`}>
              <p className="text-sm">{msg.text}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Questions */}
      <div className="p-6 border-t bg-gray-50 max-h-64 overflow-y-auto">
        <p className="text-sm font-semibold text-gray-700 mb-3">Common Questions:</p>
        <div className="space-y-2">
          {quickQuestions.map((item, idx) => (
            <button
              key={idx}
              onClick={() => handleQuestion(item.a)}
              className="w-full text-left p-3 bg-white border border-gray-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition text-sm font-medium text-gray-900"
            >
              {item.q}
            </button>
          ))}
        </div>
        <div className="mt-4">
          <label className="text-sm font-semibold text-gray-700 mb-2 block">Ask about the current analysis</label>
          <div className="flex gap-2">
            <input value={input} onChange={e => setInput(e.target.value)} placeholder="Ask a question about the prediction or heatmap" className="flex-1 px-3 py-2 border rounded" />
            <button onClick={() => { askAssistant(input); setInput(''); }} disabled={loading} className="px-4 py-2 bg-blue-600 text-white rounded">{loading ? 'Asking...' : 'Ask'}</button>
          </div>
          <div className="mt-3">
            <button onClick={retryLastWithTimeout} className="text-sm text-gray-600 hover:underline">Retry last with longer timeout</button>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Sub-Components (Updated)
// ============================================================================

const Landing = ({ navigate }) => (
  <div className="flex flex-col items-center justify-center h-full p-6 bg-gradient-to-b from-blue-50 via-white to-indigo-50">
    <div className="w-24 h-24 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-3xl flex items-center justify-center mb-6 shadow-xl transform hover:scale-105 transition">
      <Activity className="w-12 h-12 text-white" />
    </div>
    <h1 className="text-5xl font-bold text-gray-900 mb-3 text-center">OncoDetect</h1>
    <p className="text-gray-500 text-center text-lg mb-4">AI-Powered Skin Assessment</p>
    <p className="text-gray-400 text-center text-sm mb-12 max-w-sm">Advanced machine learning analysis for early detection support</p>
    
    <button 
      onClick={() => navigate('welcome')}
      className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4 rounded-2xl font-semibold text-lg shadow-lg hover:shadow-xl hover:from-blue-700 hover:to-indigo-700 transition active:scale-95 mb-3"
    >
      Get Started
      <ChevronRight className="inline ml-2 w-5 h-5" />
    </button>

    <button 
      onClick={() => navigate('stats')}
      className="w-full border-2 border-gray-300 text-gray-700 py-3 rounded-2xl font-semibold hover:border-gray-400 hover:bg-gray-50 transition mb-3"
    >
      <BarChart3 className="inline mr-2 w-5 h-5" />
      View Statistics
    </button>

    <button 
      onClick={() => navigate('assistant')}
      className="w-full border-2 border-blue-400 text-blue-600 py-3 rounded-2xl font-semibold hover:border-blue-500 hover:bg-blue-50 transition"
    >
      <MessageSquare className="inline mr-2 w-5 h-5" />
      Ask Assistant
    </button>
  </div>
);

const Welcome = ({ navigate }) => {
  const [agreed, setAgreed] = useState(false);
  
  return (
    <div className="flex flex-col h-full bg-white p-6">
      <div className="flex-1 flex flex-col justify-center overflow-y-auto">
        <ShieldCheck className="w-16 h-16 text-blue-600 mb-6" />
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Important Disclaimer</h2>
        
        <div className="space-y-4 mb-8">
          <p className="text-gray-700 text-base leading-relaxed">
            OncoDetect uses advanced artificial intelligence to help analyze skin lesions. However, this tool is <b>NOT</b> a substitute for professional medical diagnosis.
          </p>
          
          <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded">
            <p className="text-amber-900 text-sm">
              ⚠️ <b>Always consult a dermatologist</b> for proper diagnosis and treatment. AI analysis should only be used as a supplementary screening tool.
            </p>
          </div>

          <p className="text-gray-700 text-base leading-relaxed">
            Your data is processed securely and stored locally. We do not sell or share your medical information.
          </p>
        </div>

        <div className="bg-blue-50 p-4 rounded-xl flex items-start gap-3 border border-blue-100 mb-8">
          <input 
            type="checkbox" 
            id="agree" 
            className="mt-1 w-5 h-5 rounded border-gray-300 text-blue-600 cursor-pointer" 
            checked={agreed} 
            onChange={e => setAgreed(e.target.checked)} 
          />
          <label htmlFor="agree" className="text-sm text-gray-700 cursor-pointer">
            I understand the limitations of this tool and will consult a healthcare professional for medical advice.
          </label>
        </div>
      </div>

      <div className="flex gap-3">
        <button 
          onClick={() => navigate('landing')} 
          className="flex-1 border-2 border-gray-300 text-gray-700 py-3 rounded-xl font-semibold hover:bg-gray-50 transition"
        >
          Back
        </button>
        <button 
          onClick={() => navigate('metadata')} 
          disabled={!agreed}
          className={`flex-1 py-3 rounded-xl font-semibold transition flex items-center justify-center gap-2 ${agreed ? 'bg-blue-600 text-white hover:bg-blue-700 active:scale-95' : 'bg-gray-200 text-gray-400 cursor-not-allowed'}`}
        >
          Continue
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};

const HumanBodyDiagram = ({ side, selectedPoint, onSelectPoint }) => {
  const areas = {
    front: [
      { name: 'Head & Neck', x: 50, y: 10 },
      { name: 'Upper Extremity', x: 30, y: 30 },
      { name: 'Upper Extremity', x: 70, y: 30 },
      { name: 'Anterior Trunk', x: 50, y: 45 },
      { name: 'Lower Extremity', x: 35, y:75 },
      { name: 'Lower Extremity', x: 65, y: 75 },
      { name: 'Acral (Soles/Palms)', x: 50, y: 95 },
    ],
    back: [
      { name: 'Head & Neck', x: 50, y: 10 },
      { name: 'Upper Extremity', x: 30, y: 30 },
      { name: 'Upper Extremity', x: 70, y: 30 },
      { name: 'Back', x: 50, y: 45 },
      { name: 'Lower Extremity', x: 35, y: 75 },
      { name: 'Lower Extremity', x: 65, y: 75 },
      { name: 'Acral (Soles/Palms)', x: 50, y: 95 },
    ]
  };

  return (
    <svg viewBox="0 0 100 100" className="w-48 h-64 mx-auto">
      {/* Head */}
      <circle cx="50" cy="15" r="8" fill="#e5e7eb" stroke="#9ca3af" strokeWidth="2" />
      {/* Body */}
      <rect x="45" y="24" width="10" height="20" fill="#e5e7eb" stroke="#9ca3af" strokeWidth="2" />
      {/* Arms */}
      <line x1="45" y1="28" x2="20" y2="35" stroke="#9ca3af" strokeWidth="3" />
      <line x1="55" y1="28" x2="80" y2="35" stroke="#9ca3af" strokeWidth="3" />
      {/* Legs */}
      <line x1="47" y1="44" x2="40" y2="75" stroke="#9ca3af" strokeWidth="3" />
      <line x1="53" y1="44" x2="60" y2="75" stroke="#9ca3af" strokeWidth="3" />

      {/* Clickable points */}
      {areas[side].map((area,idx) => (
        <g key={idx}>
          <circle
            cx={area.x}
            cy={area.y}
            r="6"
            fill={selectedPoint === area.name ? '#3b82f6' : '#dbeafe'}
            stroke={selectedPoint === area.name ? '#1e40af' : '#93c5fd'}
            strokeWidth="2"
            opacity="0.7"
            style={{ cursor: 'pointer' }}
            onClick={() => onSelectPoint(area.name)}
          />
          <circle
            cx={area.x}
            cy={area.y}
            r="6"
            fill="none"
            stroke={selectedPoint === area.name ? '#1e40af' : '#93c5fd'}
            strokeWidth="1"
            opacity="0"
            onMouseEnter={(e) => e.target.style.opacity = '1'}
            onMouseLeave={(e) => e.target.style.opacity = '0'}
            style={{ cursor: 'pointer' }}
            onClick={() => onSelectPoint(area.name)}
          />
        </g>
      ))}
    </svg>
  );
};

const Metadata = ({ userData, setUserData, navigate }) => {
  const [focused, setFocused] = useState(null);
  const [bodyViewMode, setBodyViewMode] = useState('front');
  
  const genders = ["Female", "Male", "Other"];

  const handleContinue = () => {
    if (userData.age && userData.gender && userData.location) {
      navigate('capture');
    }
  };

  return (
  <div className="flex flex-col min-h-screen bg-white overflow-auto">
      <div className="flex items-center p-6 border-b">
        <button onClick={() => navigate('welcome')} className="text-gray-600 hover:text-gray-900">
          <ArrowLeft className="w-6 h-6" />
        </button>
        <h2 className="flex-1 text-2xl font-bold text-gray-900 ml-4">Patient Information</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Age Input */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Age</label>
          <div className="relative">
            <Calendar className="absolute left-3 top-3.5 w-5 h-5 text-gray-400" />
            <input
              type="number"
              min="1"
              max="120"
              value={userData.age}
              onChange={(e) => setUserData({...userData, age: e.target.value})}
              onFocus={() => setFocused('age')}
              onBlur={() => setFocused(null)}
              placeholder="Enter your age"
              className={`w-full pl-10 pr-4 py-3 rounded-lg border-2 transition focus:outline-none ${
                focused === 'age' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white'
              }`}
            />
          </div>
        </div>

        {/* Gender Selection */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-3">Gender</label>
          <div className="grid grid-cols-3 gap-3">
            {genders.map(gender => (
              <button
                key={gender}
                onClick={() => setUserData({...userData, gender: gender.toLowerCase()})}
                className={`p-3 rounded-lg border-2 font-medium transition ${
                  userData.gender === gender.toLowerCase()
                    ? 'border-blue-600 bg-blue-50 text-blue-600'
                    : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
                }`}
              >
                <User className="w-5 h-5 mx-auto mb-1" />
                {gender}
              </button>
            ))}
          </div>
        </div>

        {/* Body Diagram Selection */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-3">Select Lesion Location</label>
          
          {/* Body View Toggle */}
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setBodyViewMode('front')}
              className={`flex-1 py-2 rounded-lg border-2 font-medium transition ${
                bodyViewMode === 'front'
                  ? 'border-blue-600 bg-blue-50 text-blue-600'
                  : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
              }`}
            >
              Front
            </button>
            <button
              onClick={() => setBodyViewMode('back')}
              className={`flex-1 py-2 rounded-lg border-2 font-medium transition ${
                bodyViewMode === 'back'
                  ? 'border-blue-600 bg-blue-50 text-blue-600'
                  : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
              }`}
            >
              Back
            </button>
          </div>

          {/* Body Diagram */}
          <div className="bg-gradient-to-b from-blue-50 to-indigo-50 rounded-xl p-6 border-2 border-blue-200">
            <HumanBodyDiagram
              side={bodyViewMode}
              selectedPoint={userData.location}
              onSelectPoint={(point) => setUserData({...userData, location: point})}
            />
            {userData.location && (
              <p className="text-center text-sm font-semibold text-blue-600 mt-4">
                Selected: {userData.location}
              </p>
            )}
          </div>
        </div>
      </div>

      <div className="flex gap-3 p-6 border-t bg-gray-50">
        <button 
          onClick={() => navigate('welcome')} 
          className="flex-1 border-2 border-gray-300 text-gray-700 py-3 rounded-xl font-semibold hover:bg-gray-100 transition"
        >
          Back
        </button>
        <button 
          onClick={handleContinue}
          disabled={!userData.age || !userData.gender || !userData.location}
          className={`flex-1 py-3 rounded-xl font-semibold transition flex items-center justify-center gap-2 ${
            userData.age && userData.gender && userData.location
              ? 'bg-blue-600 text-white hover:bg-blue-700 active:scale-95'
              : 'bg-gray-200 text-gray-400 cursor-not-allowed'
          }`}
        >
          Next: Upload Image
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};

const Capture = ({ userData, navigate }) => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  
  // Real-time metrics state for AR guidance
  const [metrics, setMetrics] = useState({ lighting: 'Good', focus: 'Sharp', ready: false });
  
  const videoRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const frameIdRef = useRef(null);

  // EDGE PROCESSING: Analyze video frame for brightness and contrast
  const analyzeFrame = () => {
    if (!videoRef.current || !hiddenCanvasRef.current || !cameraActive) return;
    
    const video = videoRef.current;
    const canvas = hiddenCanvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    if (video.videoWidth > 0) {
      // Draw a low-res version of the frame for hyper-fast processing (64x64 is sufficient for luminance)
      ctx.drawImage(video, 0, 0, 64, 64);
      const imageData = ctx.getImageData(0, 0, 64, 64).data;
      
      let totalLuminance = 0;
      let minLum = 255;
      let maxLum = 0;

      for (let i = 0; i < imageData.length; i += 4) {
        // Standard relative luminance formula (Rec. 709)
        const lum = 0.299 * imageData[i] + 0.587 * imageData[i + 1] + 0.114 * imageData[i + 2];
        totalLuminance += lum;
        if (lum < minLum) minLum = lum;
        if (lum > maxLum) maxLum = lum;
      }
      
      const avgLuminance = totalLuminance / (imageData.length / 4);
      const contrast = maxLum - minLum;

      let lightingStatus = 'Good';
      let focusStatus = 'Sharp';
      let isReady = true;

      // Threshold adjustments
      if (avgLuminance < 60) { lightingStatus = 'Too Dark'; isReady = false; }
      else if (avgLuminance > 210) { lightingStatus = 'Too Bright'; isReady = false; }

      // Contrast acts as a lightweight mathematical proxy for focus/blur
      if (contrast < 40) { focusStatus = 'Blurry'; isReady = false; }

      setMetrics({ lighting: lightingStatus, focus: focusStatus, ready: isReady });
    }
    
    // Loop the analysis tightly bound to the browser's render cycle
    frameIdRef.current = requestAnimationFrame(analyzeFrame);
  };

  const startCamera = async () => {
    try {
      setError(null);
      // Request optimal mobile back camera
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment', width: { ideal: 1280 } }, 
        audio: false 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          analyzeFrame(); // Start AR loop once video plays
        };
      }
      setCameraActive(true);
    } catch (e) {
      setError('Unable to access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (frameIdRef.current) cancelAnimationFrame(frameIdRef.current);
    const stream = videoRef.current?.srcObject;
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
  };

  // Cleanup to prevent memory leaks if user navigates away while camera is on
  useEffect(() => {
    return () => stopCamera();
  }, []);

  const handleCapture = async () => {
    if (!preview) return;
    setLoading(true);
    setError(null);
    try {
      const result = await predictLesion(preview, userData);
      navigate('results', { prediction: result, userData, preview });
    } catch (err) {
      setError(`Analysis failed: ${err.message}`);
      setLoading(false);
    }
  };

  const captureFromCamera = async () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    setPreview(canvas.toDataURL('image/png'));
    stopCamera();
  };

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        setError('Image size must be less than 10MB');
        return;
      }
      const reader = new FileReader();
      reader.onload = (event) => setPreview(event.target.result);
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-white">
      <div className="flex items-center p-6 border-b">
        <button onClick={() => { stopCamera(); navigate('metadata'); }} className="text-gray-600 hover:text-gray-900">
          <ArrowLeft className="w-6 h-6" />
        </button>
        <h2 className="flex-1 text-2xl font-bold text-gray-900 ml-4">Capture Lesion</h2>
      </div>

      <div className="flex-1 p-6 flex flex-col">
        {/* Viewport */}
        {preview ? (
          <div className="mb-6 relative w-full aspect-square bg-black rounded-2xl overflow-hidden border-2 border-gray-200">
            <img src={preview} alt="Selected" className="absolute inset-0 w-full h-full object-contain" />
            <button onClick={() => setPreview(null)} className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-white px-6 py-2 rounded-full shadow-lg font-bold text-blue-600">
              Retake
            </button>
          </div>
        ) : (
          <div className="mb-4 relative w-full aspect-square bg-gray-100 rounded-2xl overflow-hidden border-2 border-gray-300 shadow-inner">
            {!cameraActive ? (
               <div className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-200 transition" onClick={() => fileInputRef.current?.click()}>
                 <Camera className="w-16 h-16 text-gray-400 mb-4" />
                 <p className="font-semibold text-gray-600">Tap to browse files</p>
               </div>
            ) : (
               <>
                 <video ref={videoRef} playsInline className="absolute inset-0 w-full h-full object-cover" />
                 
                 {/* AR Guidance Targeting Reticle */}
                 <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                    <div className={`w-2/3 aspect-square rounded-full border-4 transition-all duration-300 ${metrics.ready ? 'border-green-500 scale-105 shadow-[0_0_20px_rgba(34,197,94,0.6)]' : 'border-red-500 shadow-[0_0_15px_rgba(239,68,68,0.5)]'}`}></div>
                 </div>

                 {/* Real-time Data HUD */}
                 <div className="absolute top-4 left-4 right-4 flex justify-between gap-2 pointer-events-none">
                    <div className={`px-3 py-1 rounded-full text-xs font-bold text-white shadow-md ${metrics.lighting === 'Good' ? 'bg-green-600/80' : 'bg-red-600/80'}`}>
                      Lighting: {metrics.lighting}
                    </div>
                    <div className={`px-3 py-1 rounded-full text-xs font-bold text-white shadow-md ${metrics.focus === 'Sharp' ? 'bg-green-600/80' : 'bg-red-600/80'}`}>
                      Focus: {metrics.focus}
                    </div>
                 </div>
                 
                 {/* Footer Status */}
                 <div className="absolute bottom-6 w-full text-center pointer-events-none">
                    <p className="text-white text-sm font-semibold bg-black/60 inline-block px-4 py-2 rounded-full backdrop-blur-sm">
                      {metrics.ready ? "Hold steady... Ready to capture." : "Adjust position to fit target"}
                    </p>
                 </div>
               </>
            )}
            {/* Hidden canvas for pixel extraction */}
            <canvas ref={hiddenCanvasRef} width="64" height="64" className="hidden" />
          </div>
        )}

        <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageSelect} className="hidden" />

        {!preview && (
          <div className="flex gap-3 mt-4 mb-6">
            <button onClick={() => fileInputRef.current?.click()} className="flex-1 py-3 bg-white border-2 border-gray-300 font-semibold text-gray-700 hover:bg-gray-50 rounded-xl transition">
              Upload Image
            </button>
            {!cameraActive ? (
              <button onClick={startCamera} className="flex-1 py-3 bg-blue-600 font-semibold text-white rounded-xl shadow-md hover:bg-blue-700 transition">
                Start Camera
              </button>
            ) : (
              <button onClick={captureFromCamera} disabled={!metrics.ready} className={`flex-1 py-3 font-semibold text-white rounded-xl shadow-md transition ${metrics.ready ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-400 cursor-not-allowed'}`}>
                Capture
              </button>
            )}
          </div>
        )}

        {/* Static Tips Below */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
          <p className="text-sm font-semibold text-blue-900 mb-2">💡 Best Practices:</p>
          <ul className="text-sm text-blue-800 space-y-1 ml-4 list-disc">
            <li>Ensure the lesion is centered in the circle.</li>
            <li>Maintain a distance of about 4-6 inches.</li>
            <li>Avoid shadows directly over the skin.</li>
          </ul>
        </div>

        {error && (
          <div className="bg-red-50 p-4 rounded-xl flex items-center gap-3 border border-red-200 mb-4">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}
      </div>

      <div className="p-6 border-t bg-gray-50 flex gap-3 mt-auto">
        <button onClick={() => { stopCamera(); navigate('metadata'); }} className="flex-1 border-2 border-gray-300 py-3 rounded-xl font-semibold text-gray-700 hover:bg-gray-100 transition">
          Back
        </button>
        <button onClick={handleCapture} disabled={!preview || loading} className={`flex-1 py-3 rounded-xl font-semibold flex justify-center items-center gap-2 transition ${preview && !loading ? 'bg-blue-600 text-white shadow-md hover:bg-blue-700 active:scale-95' : 'bg-gray-200 text-gray-400 cursor-not-allowed'}`}>
          {loading ? <Loader className="w-5 h-5 animate-spin" /> : 'Analyze Image'}
        </button>
      </div>
    </div>
  );
};

const PreventiveGuidance = ({ userData, navigate }) => {
  const [uv, setUv] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleGetGuidance = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await apiFetch(new URL('/guidance/preventive', API_BASE_URL).toString(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metadata: userData || {}, uv_index: uv, skin_type: (userData?.skin_type || '') })
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || 'Guidance failed');
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      <div className="flex items-center gap-3 mb-4">
        <button onClick={() => navigate('results', { prediction: null, userData })} className="text-gray-600 hover:text-gray-900">
          <ArrowLeft className="w-6 h-6" />
        </button>
        <h2 className="text-2xl font-bold">Preventive Guidance</h2>
      </div>

      <div className="bg-white p-4 rounded-lg shadow">
        <label className="text-sm font-semibold">Estimated UV Index (optional)</label>
        <input value={uv} onChange={e => setUv(e.target.value)} placeholder="e.g. 7.5" className="w-full px-3 py-2 border rounded mt-2" />
        <div className="flex gap-3 mt-4">
          <button onClick={handleGetGuidance} disabled={loading} className="px-4 py-2 bg-blue-600 text-white rounded">{loading ? 'Checking...' : 'Get Guidance'}</button>
          <button onClick={() => navigate('results', { prediction: null, userData })} className="px-4 py-2 bg-gray-100 rounded">Close</button>
        </div>

        {error && <div className="mt-3 text-red-600">{error}</div>}

        {result && (
          <div className="mt-4">
            <p className="font-semibold">SPF Recommendation: SPF {result.spf_recommendation}</p>
            {result.alerts && result.alerts.length > 0 && (
              <div className="mt-2 bg-yellow-50 border border-yellow-200 p-2 rounded">
                <p className="font-semibold">Alerts</p>
                <ul className="list-disc ml-5">
                  {result.alerts.map((a,i) => <li key={i}>{a}</li>)}
                </ul>
              </div>
            )}

						{(result.nudges || []).length > 0 && (
							<div className="mt-3">
								<p className="font-semibold">Nudges</p>
								<ul className="list-disc ml-5">
									{(result.nudges || []).map((n,i) => <li key={i}>{n}</li>)}
								</ul>
							</div>
						)}

						{(result.next_steps || []).length > 0 && (
							<div className="mt-3">
								<p className="font-semibold">Next Steps</p>
								<ul className="list-disc ml-5">
									{(result.next_steps || []).map((s,i) => <li key={i}>{s}</li>)}
								</ul>
							</div>
						)}
          </div>
        )}
      </div>
    </div>
  );
};

const Admin = ({ navigate }) => {
  const [health, setHealth] = useState(null);
  const [smtp, setSmtp] = useState(null);
  const [llmTest, setLlmTest] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    (async () => {
      try {
        const h = await apiFetch(new URL('/health', API_BASE_URL).toString());
        setHealth(await h.json());
      } catch (e) {
        setError('Health check failed: ' + e.message);
      }

      try {
        const s = await apiFetch(new URL('/smtp_health', API_BASE_URL).toString());
        setSmtp(await s.json());
      } catch (e) {
        setSmtp({ status: 'error', detail: e.message });
      }

      try {
        const l = await apiFetch(new URL('/assistant/llm_test', API_BASE_URL).toString());
        setLlmTest(await l.json());
      } catch (e) {
        setLlmTest({ status: 'no_llm' });
      }

			try {
      const r = await apiFetch(new URL('/history?limit=10', API_BASE_URL).toString());
				if (r.ok) {
					setHistory(await r.json());
				} else {
					setHistory([]);
				}
			} catch (e) {
				setHistory([]);
			}
    })();
  }, []);

  return (
    <div className="p-6">
      <div className="flex items-center gap-3 mb-4">
        <button onClick={() => navigate('dashboard')} className="text-gray-600 hover:text-gray-900">
          <ArrowLeft className="w-6 h-6" />
        </button>
        <h2 className="text-2xl font-bold">Admin — Health Check</h2>
      </div>

      {error && <div className="text-red-600 mb-3">{error}</div>}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white p-4 rounded shadow">
          <h3 className="font-semibold mb-2">API Health</h3>
          <pre className="text-sm bg-gray-50 p-2 rounded">{JSON.stringify(health, null, 2)}</pre>
        </div>

        <div className="bg-white p-4 rounded shadow">
          <h3 className="font-semibold mb-2">SMTP Health</h3>
          <pre className="text-sm bg-gray-50 p-2 rounded">{JSON.stringify(smtp, null, 2)}</pre>
        </div>

        <div className="bg-white p-4 rounded shadow md:col-span-2">
          <h3 className="font-semibold mb-2">LLM Test</h3>
          <pre className="text-sm bg-gray-50 p-2 rounded">{JSON.stringify(llmTest, null, 2)}</pre>
        </div>

        <div className="bg-white p-4 rounded shadow md:col-span-2">
          <h3 className="font-semibold mb-2">Recent History (preview)</h3>
          {history.length === 0 && <p className="text-sm text-gray-500">No recent history</p>}
          {history.map(h => (
            <div key={h.id} className="border-b py-2">
              <div className="flex justify-between text-sm">
                <div>{h.id} — {h.prediction}</div>
                <div className="font-semibold">{h.confidence}%</div>
              </div>
              <div className="text-xs text-gray-500">{new Date(h.timestamp).toLocaleString()}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const Results = ({ prediction, userData, navigate, preview }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.6);
  const [blendMode, setBlendMode] = useState('screen');
  const [hueRotate, setHueRotate] = useState(0);
  const [saturation, setSaturation] = useState(2);

  // Support multiple backend response shapes
  const displayPrediction = prediction?.prediction ?? prediction?.history_item?.prediction ?? 'Unknown';
  const displayConfidence = prediction?.confidence ?? prediction?.history_item?.confidence ?? 0;
  const displayId = prediction?.id ?? prediction?.history_item?.id ?? 'n/a';
  const displayBreakdown = prediction?.confidence_breakdown ?? prediction?.history_item?.confidence_breakdown ?? {};
  const heatmapSrc = prediction?.heatmap ?? prediction?.history_item?.heatmap ?? null;
  
  const confidenceColor = 
    displayConfidence >= 80 ? 'text-green-600' :
    displayConfidence >= 60 ? 'text-yellow-600' :
    'text-orange-600';

  const riskLevel = 
    displayPrediction.toLowerCase().includes('melanoma') ? 'HIGH' :
    displayPrediction.toLowerCase().includes('carcinoma') ? 'MEDIUM' :
    'LOW';

  const riskColor = 
    riskLevel === 'HIGH' ? 'bg-red-100 border-red-300 text-red-900' :
    riskLevel === 'MEDIUM' ? 'bg-yellow-100 border-yellow-300 text-yellow-900' :
    'bg-green-100 border-green-300 text-green-900';

  // Persist this prediction into per-user local history
  useEffect(() => {
    try {
      const cur = JSON.parse(localStorage.getItem('currentUser') || 'null');
      if (!cur || !cur.phone) return;
      const users = JSON.parse(localStorage.getItem('users') || '{}');
      if (!users[cur.phone]) users[cur.phone] = { name: cur.name, phone: cur.phone };

      const histItem = prediction?.history_item ? prediction.history_item : {
        id: displayId === 'n/a' ? `local-${Date.now()}` : displayId,
        prediction: displayPrediction,
        confidence: displayConfidence,
        age: userData?.age || '',
        gender: userData?.gender || '',
        location: userData?.location || '',
        timestamp: prediction?.timestamp || new Date().toISOString(),
        confidence_breakdown: displayBreakdown || {}
      };

      users[cur.phone].history = users[cur.phone].history || [];
      if (!users[cur.phone].history.find(h => h.id === histItem.id)) {
        users[cur.phone].history.unshift(histItem);
        users[cur.phone].history = users[cur.phone].history.slice(0, 200);
        localStorage.setItem('users', JSON.stringify(users));
      }
    } catch (e) {
      console.warn('Failed to persist per-user history:', e);
    }
  }, [prediction, displayId, displayPrediction, displayConfidence, userData, displayBreakdown]);

  return (
    <div className="flex flex-col items-center justify-center p-6 bg-gradient-to-b from-white to-gray-50 gap-6">
      <CheckCircle2 className="w-16 h-16 text-green-500 mb-2 animate-bounce" />
      
      <h2 className="text-3xl font-bold text-gray-900 mb-1 text-center">Analysis Complete</h2>
      <p className="text-gray-500 text-center text-sm mb-4">ID: {displayId}</p>

      {/* Main Result Card */}
      <div className="w-full max-w-md bg-white rounded-2xl shadow-lg p-6 mb-2 border-t-4 border-blue-500">
        <p className="text-gray-500 text-sm font-semibold mb-2">PREDICTION</p>
        <h3 className="text-2xl font-bold text-gray-900 mb-4">{displayPrediction}</h3>
        
        <div className="flex items-center gap-3 mb-4">
          <div className="flex-1">
            <p className="text-xs text-gray-500 mb-1">Confidence Score</p>
            <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
              <div
                className={`h-full transition-all duration-1000 ${
                  displayConfidence >= 80 ? 'bg-green-500' :
                  displayConfidence >= 60 ? 'bg-yellow-500' :
                  'bg-orange-500'
                }`}
                style={{ width: `${displayConfidence}%` }}
              />
            </div>
          </div>
          <span className={`text-2xl font-bold ${confidenceColor}`}>
            {displayConfidence}%
          </span>
        </div>

        <div className={`mt-4 border rounded-lg p-3 ${riskColor} text-center font-semibold tracking-wide`}>
          Risk Level: {riskLevel}
        </div>
      </div>

      {/* Clinical Image & Heatmap View */}
      {preview && (
        <div className="w-full max-w-md bg-white rounded-2xl shadow-lg p-6">
          <p className="text-gray-500 text-sm font-semibold mb-3">CLINICAL VIEW</p>
          
          {/* Locked Aspect Ratio Container using object-contain */}
          <div className="relative w-full aspect-square bg-black rounded-xl overflow-hidden border border-gray-200 shadow-inner">
            <img 
              src={preview} 
              alt="lesion base" 
              className="absolute inset-0 w-full h-full object-contain" 
            />
            
            {heatmapSrc && showHeatmap && (
              <img
                src={heatmapSrc}
                alt="AI heatmap overlay"
                style={{
                  opacity: heatmapOpacity,
                  mixBlendMode: blendMode,
                  filter: `hue-rotate(${hueRotate}deg) saturate(${saturation})`
                }}
                className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-300"
              />
            )}
          </div>

          {/* Interactive Control Deck */}
          {heatmapSrc && (
            <div className="mt-5 p-4 bg-gray-50 rounded-xl border border-gray-200 space-y-5">
              
              {/* Top Row: Toggle & Blend Mode */}
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 font-semibold text-gray-700 cursor-pointer">
                  <input 
                    type="checkbox" 
                    checked={showHeatmap} 
                    onChange={() => setShowHeatmap(!showHeatmap)} 
                    className="w-4 h-4 rounded text-blue-600 accent-blue-600" 
                  />
                  AI Overlay
                </label>
                <select 
                  value={blendMode} 
                  onChange={(e) => setBlendMode(e.target.value)} 
                  className="text-sm border border-gray-300 rounded-lg px-2 py-1.5 bg-white outline-none cursor-pointer focus:ring-2 focus:ring-blue-500"
                >
                  <option value="screen">Screen (Highlight)</option>
                  <option value="multiply">Multiply (Darken)</option>
                  <option value="overlay">Overlay (Contrast)</option>
                </select>
              </div>

              {/* Opacity Slider */}
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between text-xs font-semibold text-gray-500 uppercase">
                  <span>Transparent</span>
                  <span>Opaque</span>
                </div>
                <input
                  type="range" min="0" max="1" step="0.05"
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
              </div>

              {/* Color Adjustments */}
              <div className="grid grid-cols-2 gap-4 pt-2 border-t border-gray-200">
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-semibold text-gray-500 uppercase">Hue Shift</label>
                  <input 
                    type="range" min="0" max="360" step="10" 
                    value={hueRotate} 
                    onChange={(e) => setHueRotate(parseInt(e.target.value))} 
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500" 
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-semibold text-gray-500 uppercase">Intensity</label>
                  <input 
                    type="range" min="0" max="5" step="0.5" 
                    value={saturation} 
                    onChange={(e) => setSaturation(parseFloat(e.target.value))} 
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-red-500" 
                  />
                </div>
              </div>

            </div>
          )}
        </div>
      )}

      {/* Patient Information Panel */}
      <div className="w-full max-w-md bg-white rounded-2xl shadow-lg p-6">
        <p className="text-gray-500 text-sm font-semibold mb-4">PATIENT INFORMATION</p>
        <div className="space-y-3">
          <div className="flex justify-between items-center py-2 border-b border-gray-100">
            <span className="text-gray-600">Age</span>
            <span className="font-semibold text-gray-900">{userData.age} years</span>
          </div>
          <div className="flex justify-between items-center py-2 border-b border-gray-100">
            <span className="text-gray-600">Gender</span>
            <span className="font-semibold text-gray-900 capitalize">{userData.gender}</span>
          </div>
          <div className="flex justify-between items-center py-2">
            <span className="text-gray-600">Location</span>
            <span className="font-semibold text-gray-900">{userData.location}</span>
          </div>
        </div>

        <div className="flex gap-3 mt-5">
          <button 
            onClick={() => navigate('guidance', { userData, prediction })} 
            className="flex-1 py-3 bg-green-50 text-green-700 border border-green-200 rounded-xl font-semibold hover:bg-green-100 transition"
          >
            Care Guidance
          </button>
        </div>
      </div>

      {/* Medical Disclaimer */}
      <div className="w-full max-w-md bg-amber-50 border border-amber-300 rounded-lg p-4">
        <p className="text-xs text-amber-900 text-center leading-relaxed">
          ⚠️ This analysis is generated by AI and is not a medical diagnosis. Please consult a board-certified dermatologist for professional evaluation.
        </p>
      </div>

      {/* Action Dashboard */}
      <div className="w-full max-w-md space-y-3 mb-8">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="w-full border-2 border-blue-300 text-blue-600 py-3 rounded-xl font-semibold hover:bg-blue-50 transition flex items-center justify-center gap-2"
        >
          {showDetails ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
          {showDetails ? 'Hide Statistical Breakdown' : 'View Statistical Breakdown'}
        </button>

        {showDetails && (
          <div className="bg-white rounded-xl shadow-inner border border-gray-200 p-5 mb-4">
            {Object.entries(displayBreakdown).length > 0 ? (
              Object.entries(displayBreakdown).sort(([,a], [,b]) => b - a).map(([cls, conf]) => (
                <div key={cls} className="mb-3 last:mb-0">
                  <div className="flex justify-between mb-1.5">
                    <span className="text-sm font-medium text-gray-700">{cls}</span>
                    <span className="text-sm font-bold text-gray-900">{conf}%</span>
                  </div>
                  <div className="w-full bg-gray-100 rounded-full h-2">
                    <div className="h-2 rounded-full bg-blue-500" style={{ width: `${conf}%` }} />
                  </div>
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-500 text-center">No statistical breakdown available.</p>
            )}
          </div>
        )}

        <button
          onClick={() => navigate('assistant', { prediction, userData, preview })}
          className="w-full border-2 border-purple-300 text-purple-700 bg-purple-50 py-3 rounded-xl font-semibold hover:bg-purple-100 transition flex items-center justify-center gap-2"
        >
          <MessageSquare className="w-5 h-5" />
          Discuss with AI Assistant
        </button>

        <button
          onClick={() => navigate('landing')}
          className="w-full bg-blue-600 text-white py-3.5 rounded-xl font-bold shadow-md hover:bg-blue-700 transition flex items-center justify-center gap-2"
        >
          <RefreshCcw className="w-5 h-5" />
          Start New Assessment
        </button>
      </div>
    </div>
  );
};

const Statistics = ({ navigate, currentUser }) => {
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [statsData, historyData] = await Promise.all([fetchStats(), fetchHistory()]);
        setStats(statsData);
        setHistory(historyData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [currentUser]);

  const handleDelete = async (id) => {
    try {
      await deletePrediction(id);
      setHistory(history.filter(h => h.id !== id));
    } catch (err) {
      setError(err.message);
    }
  };

  return (
  <div className="flex flex-col min-h-screen bg-white overflow-auto">
      <div className="flex items-center p-6 border-b">
        <button onClick={() => navigate('landing')} className="text-gray-600 hover:text-gray-900">
          <ArrowLeft className="w-6 h-6" />
        </button>
        <h2 className="flex-1 text-2xl font-bold text-gray-900 ml-4">Statistics</h2>
      </div>

      {loading ? (
        <div className="flex-1 flex items-center justify-center">
          <Loader className="w-8 h-8 text-blue-600 animate-spin" />
        </div>
      ) : error ? (
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="text-center">
            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <p className="text-red-600">{error}</p>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Summary Cards */}
          {stats && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-4 border border-blue-200">
                <p className="text-gray-600 text-sm font-semibold">Total Assessments</p>
                <p className="text-3xl font-bold text-blue-600 mt-2">{stats.total_predictions}</p>
              </div>
              <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-4 border border-green-200">
                <p className="text-gray-600 text-sm font-semibold">Most Common</p>
                <p className="text-lg font-bold text-green-600 mt-2 truncate">{stats.top_prediction || 'N/A'}</p>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-4 border border-purple-200">
                <p className="text-gray-600 text-sm font-semibold">Avg Confidence</p>
                <p className="text-3xl font-bold text-purple-600 mt-2">{stats.avg_confidence}%</p>
              </div>
            </div>
          )}

          {/* Prediction Breakdown */}
          {stats?.prediction_breakdown && Object.keys(stats.prediction_breakdown).length > 0 && (
            <div className="bg-white rounded-xl border p-4">
              <p className="text-gray-600 text-sm font-semibold mb-4">Prediction Distribution</p>
              <div className="space-y-3">
                {Object.entries(stats.prediction_breakdown)
                  .sort(([,a], [,b]) => b - a)
                  .map(([pred, count]) => (
                    <div key={pred}>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium text-gray-700">{pred}</span>
                        <span className="text-sm font-semibold text-gray-900">{count} cases</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="h-2 rounded-full bg-blue-500"
                          style={{ width: `${(count / stats.total_predictions) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* History */}
          {history.length > 0 && (
            <div className="bg-white rounded-xl border p-4">
              <p className="text-gray-600 text-sm font-semibold mb-4">Recent Assessments</p>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {history.map(item => (
                  <div key={item.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition">
                    <div className="flex-1">
                      <p className="font-semibold text-gray-900">{item.prediction}</p>
                      <p className="text-xs text-gray-500">{formatTimestamp(item.timestamp)} | {item.confidence}% confidence</p>
                    </div>
                    <button
                      onClick={() => handleDelete(item.id)}
                      className="p-2 text-gray-400 hover:text-red-600 transition"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {history.length === 0 && (
            <div className="text-center py-12">
              <History className="w-12 h-12 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500">No assessments yet</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// MAIN APP
// ============================================================================

export default function App() {
  const [currentScreen, setCurrentScreen] = useState('homepage');
  const [currentUser, setCurrentUser] = useState(null);
  const [userData, setUserData] = useState({ age: '', gender: '', location: '' });
  const [resultData, setResultData] = useState(null);

  // Quick health-check helper for mobile debugging
  const checkApiHealth = async () => {
    try {
      const res = await apiFetch(new URL('/health', API_BASE_URL).toString());
      const txt = await res.text();
      alert(`API ${API_BASE_URL} responded: ${res.status}\n${txt}`);
    } catch (e) {
      alert(`API health check failed: ${e.message}`);
    }
  };

  // Reset session-scoped UI when the active profile changes so each profile has a fresh view
  useEffect(() => {
    setUserData({ age: '', gender: '', location: '' });
    setResultData(null);
  }, [currentUser]);

  // Using server-side Twilio SMS; no Firebase session to persist

  // Load user from localStorage on mount
  useEffect(() => {
    const savedUser = localStorage.getItem('currentUser');
    if (savedUser) {
      setCurrentUser(JSON.parse(savedUser));
      setCurrentScreen('dashboard');
    }
  }, []);

  // CLEANUP: remove all registered profiles and session data for a fresh demo state


  const navigate = (screen, data = null) => {
    // If any navigation provides data, store it as the current result context
    if (data) setResultData(data);
    setCurrentScreen(screen);
  };

  // If not logged in, show public pages
	if (!currentUser) {
    return (
      <div className="w-full min-h-screen bg-gray-50 flex flex-col overflow-auto">
        {currentScreen === 'homepage' && <HomePage navigate={navigate} />}
                {currentScreen === 'signup' && <SignUp navigate={navigate} setCurrentUser={setCurrentUser} />}
                {currentScreen === 'verifyphone' && <VerifyPhone navigate={navigate} setCurrentUser={setCurrentUser} />}
                {currentScreen === 'login' && <Login navigate={navigate} setCurrentUser={setCurrentUser} />}

        {/* Debug: persistent floating API health button for public pages */}
        <div style={{position: 'fixed', right: 12, bottom: 14, zIndex: 60, display: 'flex', flexDirection: 'column', gap: 8}}>
          <button onClick={async () => {
            try {
              const res = await fetch(new URL('/health', API_BASE_URL).toString());
              const txt = await res.text();
              alert(`API ${API_BASE_URL} status ${res.status}\n${txt}`);
            } catch (err) {
              alert(`API health check failed: ${err.message}`);
            }
          }} className="bg-black text-white px-3 py-2 rounded shadow text-sm">API</button>

          <button onClick={async () => {
            const samplePhone = '+919849858407';
            const url = new URL('/send_sms_code', API_BASE_URL).toString();
            try {
              const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phone_number: samplePhone }),
                mode: 'cors'
              });
              const text = await res.text();
              alert(`POST ${url} => ${res.status}\n${text}`);
            } catch (err) {
              alert(`SMS test failed (POST ${url}): ${err.message}`);
            }
          }} className="bg-blue-600 text-white px-3 py-2 rounded shadow text-sm">Test SMS</button>

          <button onClick={async () => {
            const phone = prompt('Phone to verify (E.164):', '+919849858407');
            if (!phone) return alert('Cancelled');
            const code = prompt('Enter code to verify (6 digits):', '');
            if (!code) return alert('Cancelled');
            const url = new URL('/verify_phone_code', API_BASE_URL).toString();
            try {
              const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phone_number: phone, code: code }),
                mode: 'cors'
              });
              const text = await res.text();
              alert(`POST ${url} => ${res.status}\n${text}`);
            } catch (err) {
              alert(`Verify test failed (POST ${url}): ${err.message}`);
            }
          }} className="bg-green-600 text-white px-3 py-2 rounded shadow text-sm">Test Verify</button>
        </div>
      </div>
    );
  }

  // If logged in, show main app
 return (
    <div className="w-full h-screen bg-gray-50 flex flex-col overflow-hidden">
      {/* Top navigation for authenticated pages */}
      {currentUser && (
        <div className="bg-white border-b shadow-sm">
          <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button onClick={() => navigate('dashboard')} className="text-gray-700 hover:text-gray-900 font-bold text-lg px-3 py-2">
                Home
              </button>
              <button onClick={() => navigate('landing')} className="text-gray-600 hover:text-gray-900 font-bold text-lg px-3 py-2">
                New Analysis
              </button>
              <button onClick={() => navigate('stats')} className="text-gray-600 hover:text-gray-900 font-bold text-lg px-3 py-2">
                Stats
              </button>
              <button onClick={() => navigate('detailedhistory')} className="text-gray-600 hover:text-gray-900 font-bold text-lg px-3 py-2">
                History
              </button>
              <button onClick={() => navigate('admin')} className="text-gray-600 hover:text-gray-900 font-bold text-lg px-3 py-2">
                Admin
              </button>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm sm:text-base font-bold text-gray-700">{currentUser.name}</span>
              <button onClick={() => navigate('profile')} className="text-gray-700 hover:text-gray-900 font-bold sm:text-base">Profile</button>
              <button onClick={checkApiHealth} title="API Health" className="ml-2 text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">API</button>
              <div className="ml-2 text-xs text-gray-500">{API_BASE_URL.replace(/^https?:\/\//,'')}</div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content - FIXED SCROLLING HERE */}
      <div className="flex-1 min-h-0 max-w-full sm:max-w-2xl mx-auto w-full bg-white rounded-t-3xl overflow-y-auto flex flex-col shadow-2xl border-t border-gray-100">
        {currentScreen === 'dashboard' && <Dashboard navigate={navigate} currentUser={currentUser} />}
        {currentScreen === 'landing' && <Landing navigate={navigate} />}
        {currentScreen === 'welcome' && <Welcome navigate={navigate} />}
        {currentScreen === 'metadata' && <Metadata userData={userData} setUserData={setUserData} navigate={navigate} />}
        {currentScreen === 'capture' && <Capture userData={userData} navigate={navigate} />}
        {currentScreen === 'results' && resultData && (
          <Results prediction={resultData.prediction} userData={resultData.userData} preview={resultData.preview} navigate={navigate} />
        )}
        {currentScreen === 'stats' && <Statistics navigate={navigate} currentUser={currentUser} />}
        {currentScreen === 'assistant' && <Assistant navigate={navigate} context={resultData} />}
        {currentScreen === 'guidance' && <PreventiveGuidance userData={resultData?.userData ?? userData} navigate={navigate} />}
        {currentScreen === 'admin' && <Admin navigate={navigate} />}
        {currentScreen === 'profile' && <Profile currentUser={currentUser} navigate={navigate} setCurrentUser={setCurrentUser} />}
        {currentScreen === 'detailedhistory' && <DetailedHistory navigate={navigate} currentUser={currentUser} />}
      </div>

      {/* Footer */}
      {currentScreen !== 'dashboard' && (
        <div className="bg-gray-900 text-gray-400 text-xs text-center py-3 border-t">
          <p>Medical Disclaimer: This tool is NOT a substitute for professional medical advice. Always consult a dermatologist.</p>
        </div>
      )}
    </div>
  );
}
