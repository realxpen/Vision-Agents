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

## Quick Preview (React)

Open this file in your browser:

- `examples/10_sentinel_ai/frontend/preview.html`

This preview now uses React (via ESM CDN) and includes simulation buttons for LOW / MEDIUM / HIGH events.

## Keep Agent Active (No Idle Timeout)

In `.env`, add:

```env
AGENT_IDLE_TIMEOUT_SECONDS=0
```

Behavior:
- `0` or negative => effectively disabled (set to ~10 years internally)
- positive value => used as idle timeout in seconds
