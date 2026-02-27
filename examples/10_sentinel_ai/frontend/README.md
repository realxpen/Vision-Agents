# Sentinel UI (Dark Command Center)

Use this as a drop-in page section in your existing Stream demo frontend.

## Integrate

1. Import:

```js
import "./sentinel_panel.css";
import { mountSentinelPanel } from "./sentinel_panel";
```

2. Ensure you have a mount node:

```html
<div id="sentinel-root"></div>
```

3. Mount after call is ready:

```js
const unmountSentinelPanel = mountSentinelPanel({
  root: document.getElementById("sentinel-root"),
  call,
  onStartMonitoring: () => {
    // optional: connect camera/join call/start tracks
  },
});
```

4. Cleanup:

```js
unmountSentinelPanel();
```

## Events Consumed

- `risk_status` with `{ risk: "LOW" | "MEDIUM" | "HIGH" }`
- `incident_log` with `{ message: string }`
- `warning` with `{ message: string }`

## Quick Preview (React, Real Stream Call, Auto .env Config)

Open this file in your browser:

- `examples/10_sentinel_ai/frontend/preview.html`

This preview uses React (ESM CDN) and connects to a real Stream call.
It fetches API key + user token from a local backend endpoint, so you do not type keys in the browser.

### End-to-end steps

1. Start the agent with a fixed call ID:

```bash
python examples/10_sentinel_ai/sentinel_ai.py run --no-demo --call-id sentinel-live
```

2. Run the frontend dev server:

```bash
cd examples/10_sentinel_ai/frontend
python dev_server.py
```

3. Open:
- `http://localhost:5500/preview.html`

4. Click `Connect to Call`.

The incident panel and risk status will now be driven by real agent custom events.

## Keep Agent Active (No Idle Timeout)

In `.env`, add:

```env
AGENT_IDLE_TIMEOUT_SECONDS=0
```

Behavior:
- `0` or negative => effectively disabled (set to ~10 years internally)
- positive value => used as idle timeout in seconds
