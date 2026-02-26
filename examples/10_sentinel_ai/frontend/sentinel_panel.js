export function mountSentinelPanel({ root, call, onStartMonitoring }) {
  const panel = document.createElement("section");
  panel.className = "sentinel-ui";
  panel.innerHTML = `
    <header class="sentinel-topbar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">~</div>
        <div>
          <h1>VisionGuard AI</h1>
          <p>Real-Time Safety &amp; Risk Detection Agent</p>
        </div>
      </div>
      <div class="top-actions">
        <span id="sentinel-system-pill" class="system-pill">SYSTEM STANDBY</span>
        <button id="sentinel-start-btn" class="start-btn" type="button">Start Monitoring</button>
      </div>
    </header>

    <main class="sentinel-grid">
      <section class="camera-col">
        <div class="camera-stage">
          <div class="camera-overlay">
            <div class="camera-icon">[CAM]</div>
            <p id="sentinel-camera-state">Camera offline. Click Start Monitoring.</p>
          </div>
        </div>

        <div class="session-report">
          <div>
            <h3>Session Report</h3>
            <p>Generate a summary of all detected incidents.</p>
          </div>
          <button type="button" id="sentinel-report-btn" class="report-btn">Generate Summary</button>
        </div>
      </section>

      <aside class="incident-col">
        <div class="incident-head">
          <h2>Incident Log</h2>
          <span id="sentinel-event-count" class="event-count">0 Events</span>
        </div>
        <div class="incident-panel">
          <div id="sentinel-empty" class="incident-empty">No incidents detected yet.</div>
          <ul id="sentinel-incidents"></ul>
        </div>
        <div class="risk-line">Risk Status: <strong id="sentinel-risk">LOW</strong></div>
      </aside>
    </main>
  `;

  root.appendChild(panel);

  const riskEl = panel.querySelector("#sentinel-risk");
  const incidentsEl = panel.querySelector("#sentinel-incidents");
  const emptyEl = panel.querySelector("#sentinel-empty");
  const countEl = panel.querySelector("#sentinel-event-count");
  const systemPillEl = panel.querySelector("#sentinel-system-pill");
  const cameraStateEl = panel.querySelector("#sentinel-camera-state");
  const startBtnEl = panel.querySelector("#sentinel-start-btn");

  startBtnEl?.addEventListener("click", () => {
    systemPillEl.textContent = "MONITORING LIVE";
    systemPillEl.classList.add("live");
    cameraStateEl.textContent = "Monitoring started. Waiting for activity...";
    if (typeof onStartMonitoring === "function") onStartMonitoring();
  });

  function setRisk(risk) {
    const normalized = String(risk || "LOW").toUpperCase();
    riskEl.textContent = normalized;
    riskEl.dataset.risk = normalized;
  }

  function addIncident(text) {
    const li = document.createElement("li");
    li.className = "incident-item";
    li.innerHTML = `
      <span class="dot"></span>
      <div>
        <strong>Safety Alert</strong>
        <p>${text}</p>
      </div>
    `;
    incidentsEl.prepend(li);
    emptyEl.style.display = "none";
    while (incidentsEl.children.length > 50) {
      incidentsEl.removeChild(incidentsEl.lastChild);
    }
    countEl.textContent = `${incidentsEl.children.length} Events`;
  }

  const detach = call?.on?.("custom", (event) => {
    const payload = event?.custom || event;
    if (!payload || typeof payload !== "object") return;

    if (payload.type === "risk_status" && payload.risk) {
      setRisk(payload.risk);
    }

    if (payload.type === "incident_log" && payload.message) {
      addIncident(String(payload.message));
    }

    if (payload.type === "warning" && payload.message) {
      addIncident(String(payload.message));
    }
  });

  return () => {
    if (typeof detach === "function") detach();
    panel.remove();
  };
}
