# Sentinel AI

Real-Time Safety & Risk Detection Agent built with Vision Agents SDK.

## What This Demo Does

- Detects missing helmet from video and logs a HIGH-risk incident.
- Detects phone usage from video and logs a MEDIUM-risk incident.
- Detects loud noise from audio and logs LOW/MEDIUM/HIGH risk by RMS thresholds.
- Updates risk status and incident log in the frontend in real time (newest first).
- Supports `Generate Summary` from recent incidents in the UI.

## Setup

1. Create and activate a Python 3.13 virtual environment at repo root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

3. Configure environment variables:

- Copy values from `examples/10_sentinel_ai/.env.example` into your repo-root `.env`.
- Fill in real API keys for Stream, OpenAI, and Roboflow.

Recommended: copy `examples/10_sentinel_ai/.env.example` to `.env` at repo root and edit values.

## .env Example

Use `examples/10_sentinel_ai/.env.example` as your template.

## Exact Run Commands

Run from repo root (`C:\Users\HP\Documents\sentinel_ai_agent\Vision-Agents`).

1. Start backend agent:

```powershell
.\.venv\Scripts\Activate.ps1
python examples/10_sentinel_ai/sentinel_ai.py run --no-demo --call-id sentinel-live-001 --log-level info
```

2. Start frontend dev server in a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python examples/10_sentinel_ai/frontend/dev_server.py
```

3. Open frontend:

- `http://localhost:5500/preview.html`

4. Click `Connect to Call` and keep helmet mode/video enabled for vision detection.

## Feature List

- Vision + audio safety monitoring in one agent.
- Live incident feed with risk badge and risk type.
- Helmet state transitions:
  - `High risk - helmet missing`
  - `Low risk - helmet worn`
- Phone detection with anti-false-positive guards:
  - confidence threshold control
  - person-required gate
  - multi-frame confirmation
- Loud-noise thresholds:
  - `SENTINEL_NOISE_RMS_LOW`
  - `SENTINEL_NOISE_RMS_MEDIUM`
  - `SENTINEL_NOISE_RMS_HIGH`
- Quiet demo logging mode:
  - `SENTINEL_QUIET_LOGS=1`

## Limitations

- Detection quality depends on camera angle, lighting, and model quality.
- Roboflow hosted inference requires stable internet.
- Phone false positives can still occur if confidence is too low.
- If backend/frontend call IDs differ, no incidents will appear.
- Realtime transport can fail under unstable network conditions.
- Current logic is optimized for demo speed, not full production hardening.
