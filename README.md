## Sentinel AI (Hackathon Project)

Sentinel AI is a real-time multimodal safety agent built with the Vision Agents SDK.  
It watches live video, listens to audio, classifies workplace risk, and logs incidents automatically.

### What It Detects
- Helmet not worn -> **HIGH** risk
- Phone detected -> **MEDIUM** risk
- Loud sound detected -> **LOW / MEDIUM / HIGH** risk (by thresholds)

### Built With
- Vision Agents SDK (`Agent`, `Runner`, `getstream.Edge`, processors, tool-calling)
- Stream Video (real-time transport)
- OpenAI Realtime (reasoning + responses)
- Roboflow hosted models (helmet + phone)
- Custom audio RMS processor (noise risk detection)

## Run Locally (Exact Steps)

### 1) Clone and install
```bash
git clone https://github.com/realxpen/Sentinel_AI.git
cd Sentinel_AI
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
2) Create environment file
Copy-Item examples/10_sentinel_ai/.env.example .env
Fill these in .env:

STREAM_API_KEY
STREAM_API_SECRET
OPENAI_API_KEY
ROBOFLOW_API_KEY
Keep:

SENTINEL_FRONTEND_CALL_ID=sentinel-live-001
3) Start backend (Terminal 1)
.\.venv\Scripts\Activate.ps1
python examples/10_sentinel_ai/sentinel_ai.py run --no-demo --call-id sentinel-live-001 --log-level info
4) Start frontend (Terminal 2)
.\.venv\Scripts\Activate.ps1
python examples/10_sentinel_ai/frontend/dev_server.py
5) Open app
http://localhost:5500/preview.html
In UI:

Enable Helmet Mode (send camera to agent)
Click Connect to Call
Click Start Monitoring
Judge Test Checklist
No helmet in camera -> incident log shows HIGH + helmet type
Hold phone visibly -> incident log shows MEDIUM + phone type
Clap/loud sound -> incident log shows sound type + risk
Click Generate Summary -> summary entry appears in incident log
Notes / Limitations
Internet is required (Stream + Roboflow APIs).
Accuracy depends on camera angle, lighting, and model quality.
Frontend/backend must use the same call ID, otherwise incidents will not appear.
Secrets are not committed; use .env locally and .env.example as template.
