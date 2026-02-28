# Sentinel AI
Real-Time Safety and Risk Detection Agent.

## Built For This Hackathon
- Uses **Vision Agents SDK** (`Agent`, `Runner`, `getstream.Edge`, processors, tool-calling).
- Multi-modal monitoring: **video + audio** in real time.
- Detects and logs:
1. Helmet not worn -> `HIGH`
2. Phone detected -> `MEDIUM`
3. Loud noise detected -> `LOW/MEDIUM/HIGH`

## Tech Stack
- Vision Agents SDK
- Stream Video (edge transport)
- OpenAI Realtime
- Roboflow hosted detection models (helmet + phone)
- Custom audio RMS noise processor

## Quick Start (Judge Runbook)
Run from repo root.

1. Create/activate env and install:
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

2. Create `.env` from template:
```powershell
Copy-Item examples/10_sentinel_ai/.env.example .env
```
Fill API keys in `.env`: `STREAM_API_KEY`, `STREAM_API_SECRET`, `OPENAI_API_KEY`, `ROBOFLOW_API_KEY`.

3. Start backend:
```powershell
python examples/10_sentinel_ai/sentinel_ai.py run --no-demo --call-id sentinel-live-001 --log-level info
```

4. Start frontend (new terminal):
```powershell
.\.venv\Scripts\Activate.ps1
python examples/10_sentinel_ai/frontend/dev_server.py
```

5. Open:
- `http://localhost:5500/preview.html`

6. In UI:
1. Enable `Helmet Mode (send camera to agent)` before connecting.
2. Click `Connect to Call`.
3. Click `Start Monitoring`.

## What Judges Should Test
1. Stand without helmet -> incident log shows `HIGH` + helmet risk type.
2. Hold phone in view -> incident log shows `MEDIUM` + phone risk type.
3. Make loud noise (clap) -> incident log shows sound risk type.
4. Click `Generate Summary` -> summary entry appears in log.

## Config Notes
- Keep frontend/backend call IDs the same.
- Stable demo defaults are in:
  - `examples/10_sentinel_ai/.env.example`
  - root `.env.example` Sentinel section

## Limitations
- Requires internet connectivity (Stream + Roboflow APIs).
- Detection accuracy depends on camera angle, lighting, and model quality.
- Network instability can cause RTC/session drops.
