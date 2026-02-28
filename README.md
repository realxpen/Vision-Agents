<<<<<<< HEAD
# Sentinel AI
Real-Time Safety and Risk Detection Agent built with Vision Agents SDK.

## Overview
Sentinel AI is a multimodal safety agent that monitors live video and audio, detects workplace safety violations, classifies risk severity, and logs incidents in real time.
=======
## Sentinel AI (Hackathon Project)
>>>>>>> 8983685d370362388ae0dec593d5bdd4b1c80031

### Core Detection Scenarios
1. Helmet not worn -> `HIGH` risk
2. Phone detected -> `MEDIUM` risk
3. Loud sound detected -> `LOW`, `MEDIUM`, or `HIGH` risk

## Tech Stack
| Layer | Technology |
|---|---|
| Agent Framework | Vision Agents SDK (`Agent`, `Runner`, processors, tool-calling) |
| Realtime Transport | Stream Video (`getstream.Edge`) |
| Reasoning | OpenAI Realtime (`gpt-realtime-1.5`) |
| Vision Detection | Roboflow hosted models (helmet + phone), YOLO support path |
| Audio Detection | Custom RMS processor (NumPy) |
| Media Processing | OpenCV, AV |
| Frontend | React + `@stream-io/video-client` |
| Config/Token API | Python `http.server` (`dev_server.py`) |
| Deployment | Railway (agent + config API), Vercel (frontend) |

## Architecture
```text
Camera + Mic (Frontend)
        |
        v
Stream Call (shared call ID)
        |
        +--> Sentinel Agent (Vision Agents SDK)
              |
              +--> Video processor -> Roboflow helmet/phone inference
              +--> Audio processor -> RMS noise risk
              +--> OpenAI Realtime reasoning + tool-calling
              |
              +--> Custom events: incident_log, risk_status, warning
        |
        v
React Incident Panel (newest-first logs + summary)
```

## Local Run (Judge-Friendly)
Run from repo root.

### 1) Setup
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
Copy-Item examples/10_sentinel_ai/.env.example .env
```

Fill real keys in `.env`:
- `STREAM_API_KEY`
- `STREAM_API_SECRET`
- `OPENAI_API_KEY`
- `ROBOFLOW_API_KEY`

### 2) Start backend
```powershell
.\.venv\Scripts\Activate.ps1
python examples/10_sentinel_ai/sentinel_ai.py run --no-demo --call-id sentinel-live-001 --log-level info
```

### 3) Start frontend (new terminal)
```powershell
.\.venv\Scripts\Activate.ps1
python examples/10_sentinel_ai/frontend/dev_server.py
```

<<<<<<< HEAD
### 4) Open app
- `http://localhost:5500/preview.html`

In UI:
1. Enable `Helmet Mode (send camera to agent)`
2. Click `Connect to Call`
3. Click `Start Monitoring`

## What To Test
1. Show no helmet in camera -> `HIGH` helmet incident
2. Hold phone in hand -> `MEDIUM` phone incident
3. Make loud sound (clap) -> sound incident with risk level
4. Click `Generate Summary` -> summary appears in incident log

## Project Highlights
- Real-time multimodal safety monitoring (video + audio)
- Live incident feed and risk status updates
- Tool-calling for autonomous logging and warning workflows
- False-positive controls for phone detection:
  - confidence threshold
  - person-required gate
  - multi-frame confirmation

## Known Limitations
- Detection quality depends on lighting, angle, and model quality
- Requires internet (Stream + Roboflow APIs)
- Realtime session can drop on unstable networks
- Frontend and backend must use the same call ID

## Important Links
- Sentinel example code: `examples/10_sentinel_ai/`
- Example env template: `examples/10_sentinel_ai/.env.example`
- Frontend preview: `examples/10_sentinel_ai/frontend/preview.html`

## Security Note
- `.env` is ignored and should never be committed
- Rotate API keys if they were ever exposed
=======
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
>>>>>>> 8983685d370362388ae0dec593d5bdd4b1c80031
