import React from "https://esm.sh/react@18.3.1";

const e = React.createElement;

export function SentinelPanel({ call, onStartMonitoring, onStopMonitoring }) {
  const [risk, setRisk] = React.useState("LOW");
  const [incidents, setIncidents] = React.useState([]);
  const [live, setLive] = React.useState(false);
  const [mediaReady, setMediaReady] = React.useState(false);
  const [micLevel, setMicLevel] = React.useState(0);
  const [cameraText, setCameraText] = React.useState(
    "Camera offline. Click Start Monitoring."
  );
  const videoRef = React.useRef(null);
  const streamRef = React.useRef(null);
  const audioCtxRef = React.useRef(null);
  const rafRef = React.useRef(null);
  const incidentPanelRef = React.useRef(null);

  function detectRiskFromText(text) {
    const t = String(text || "").toLowerCase();
    if (t.includes("high risk") || t.includes("high")) return "HIGH";
    if (t.includes("medium risk") || t.includes("medium")) return "MEDIUM";
    if (t.includes("low risk") || t.includes("low")) return "LOW";
    return null;
  }

  function detectRiskTypeFromText(text) {
    const t = String(text || "").toLowerCase();
    if (t.includes("helmet")) return "HELMET_NOT_WORN_DETECTED";
    if (t.includes("phone")) return "PHONE_DETECTED";
    if (t.includes("noise") || t.includes("sound")) return "SOUND_LEVEL_DETECTED";
    return "GENERAL";
  }

  function addIncident(message, riskHint, riskTypeHint) {
    const msg = String(message || "").trim();
    if (!msg) return;
    const detected = detectRiskFromText(msg);
    const level = String(riskHint || detected || risk || "MEDIUM").toUpperCase();
    const type = String(
      riskTypeHint || detectRiskTypeFromText(msg) || "GENERAL"
    ).toUpperCase();
    const incident = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      level: ["HIGH", "MEDIUM", "LOW"].includes(level) ? level : "MEDIUM",
      riskType: [
        "HELMET_NOT_WORN_DETECTED",
        "PHONE_DETECTED",
        "SOUND_LEVEL_DETECTED",
        "GENERAL",
      ].includes(type)
        ? type
        : "GENERAL",
      ts: Date.now(),
    };
    setIncidents((prev) => [incident, ...prev].slice(0, 100));
  }

  React.useEffect(() => {
    if (!call || typeof call.on !== "function") return undefined;
    const detach = call.on("custom", (event) => {
      const payload = event?.custom || event;
      if (!payload || typeof payload !== "object") return;

      if (payload.type === "risk_status" && payload.risk) {
        setRisk(String(payload.risk).toUpperCase());
      }

      if (payload.type === "incident_log" && payload.message) {
        addIncident(payload.message, payload.risk, payload.risk_type);
      }

      if (payload.type === "warning" && payload.message) {
        addIncident(payload.message, payload.risk, payload.risk_type);
      }
    });

    return () => {
      if (typeof detach === "function") detach();
    };
  }, [call]);

  function stopMedia() {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) track.stop();
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setMediaReady(false);
    setMicLevel(0);
  }

  async function startLocalMedia() {
    try {
      stopMedia();
      let stream = null;
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        });
      } else {
        const legacyGetUserMedia =
          navigator.getUserMedia ||
          navigator.webkitGetUserMedia ||
          navigator.mozGetUserMedia;
        if (legacyGetUserMedia) {
          stream = await new Promise((resolve, reject) => {
            legacyGetUserMedia.call(
              navigator,
              { video: true, audio: true },
              resolve,
              reject
            );
          });
        }
      }
      if (!stream) {
        throw new Error(
          "Camera/Mic API unavailable. On phone, open this app over HTTPS (not plain HTTP IP) and allow permissions."
        );
      }
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setMediaReady(true);
      setCameraText("Live camera + microphone active.");

      const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextCtor) return;
      const audioCtx = new AudioContextCtor();
      audioCtxRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      const data = new Uint8Array(analyser.fftSize);

      const tick = () => {
        analyser.getByteTimeDomainData(data);
        let sum = 0;
        for (let i = 0; i < data.length; i += 1) {
          const v = (data[i] - 128) / 128;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / data.length);
        const normalized = Math.max(0, Math.min(1, rms * 6));
        setMicLevel(normalized);
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();
    } catch (err) {
      setMediaReady(false);
      setCameraText(`Media access failed: ${String(err?.message || err)}`);
    }
  }

  async function startMonitoring() {
    setLive(true);
    setCameraText("Monitoring started. Waiting for activity...");
    await startLocalMedia();
    if (typeof onStartMonitoring === "function") onStartMonitoring();
  }

  async function stopMonitoring() {
    stopMedia();
    setLive(false);
    setCameraText("Camera offline. Click Start Monitoring.");
    if (typeof onStopMonitoring === "function") onStopMonitoring();
  }

  function generateSummary() {
    if (incidents.length === 0) {
      addIncident("Low risk - no incidents recorded this session", "LOW", "GENERAL");
      return;
    }
    const counts = { HIGH: 0, MEDIUM: 0, LOW: 0 };
    const typeCounts = {
      HELMET_NOT_WORN_DETECTED: 0,
      PHONE_DETECTED: 0,
      SOUND_LEVEL_DETECTED: 0,
      GENERAL: 0,
    };
    for (const item of incidents) {
      if (counts[item.level] !== undefined) counts[item.level] += 1;
      if (typeCounts[item.riskType] !== undefined) typeCounts[item.riskType] += 1;
    }
    const overallRisk =
      counts.HIGH > 0 ? "HIGH" : counts.MEDIUM > 0 ? "MEDIUM" : "LOW";
    const summary =
      `Session summary - H:${counts.HIGH} M:${counts.MEDIUM} L:${counts.LOW} ` +
      `| Helmet:${typeCounts.HELMET_NOT_WORN_DETECTED} Phone:${typeCounts.PHONE_DETECTED} Sound:${typeCounts.SOUND_LEVEL_DETECTED}`;
    addIncident(summary, overallRisk, "GENERAL");
  }

  function formatRiskType(type) {
    if (type === "HELMET_NOT_WORN_DETECTED") return "Helmet Not Worn Detected";
    if (type === "PHONE_DETECTED") return "Phone Detected";
    if (type === "SOUND_LEVEL_DETECTED") return "Sound Level Detected";
    return "General";
  }

  React.useEffect(() => {
    return () => {
      stopMedia();
    };
  }, []);

  React.useEffect(() => {
    if (!incidentPanelRef.current) return;
    incidentPanelRef.current.scrollTop = 0;
  }, [incidents.length]);

  return e(
    "section",
    { className: "sentinel-ui" },
    e(
      "header",
      { className: "sentinel-topbar" },
      e(
        "div",
        { className: "brand" },
        e("div", { className: "brand-mark", "aria-hidden": "true" }, "~"),
          e("div", null, [
          e("h1", { key: "h" }, "Sentinel AI"),
          e("p", { key: "p" }, "Real-Time Safety & Risk Detection Agent"),
          e("span", { key: "sdk", className: "sdk-badge" }, "Vision Agents SDK"),
        ])
      ),
      e("div", { className: "top-actions" }, [
        e(
          "span",
          {
            key: "pill",
            className: `system-pill${live ? " live" : ""}`,
          },
          live ? "MONITORING LIVE" : "SYSTEM STANDBY"
        ),
        live
          ? e(
              "button",
              {
                key: "stop",
                className: "start-btn",
                type: "button",
                onClick: stopMonitoring,
              },
              "Stop Monitoring"
            )
          : e(
              "button",
              {
                key: "start",
                className: "start-btn",
                type: "button",
                onClick: startMonitoring,
              },
              "Start Monitoring"
            ),
      ])
    ),
    e(
      "main",
      { className: "sentinel-grid" },
      e("section", { className: "camera-col" }, [
        e(
          "div",
          { key: "stage", className: "camera-stage" },
          [
            e("video", {
              key: "v",
              ref: videoRef,
              className: `camera-video${mediaReady ? " live" : ""}`,
              autoPlay: true,
              playsInline: true,
              muted: true,
            }),
            e("div", { key: "o", className: "camera-overlay" }, [
              e("div", { key: "icon", className: "camera-icon" }, "[CAM]"),
              e("p", { key: "txt" }, cameraText),
            ]),
            e("div", { key: "m", className: "mic-meter" }, [
              e("span", { key: "l" }, "MIC"),
              e("div", { key: "track", className: "mic-track" }, [
                e("div", {
                  key: "fill",
                  className: "mic-fill",
                  style: { width: `${Math.round(micLevel * 100)}%` },
                }),
              ]),
            ]),
          ]
        ),
        e("div", { key: "report", className: "session-report" }, [
          e("div", { key: "txt" }, [
            e("h3", { key: "h" }, "Session Report"),
            e("p", { key: "p" }, "Generate a summary of all detected incidents."),
          ]),
          e(
            "button",
            {
              key: "b",
              type: "button",
              className: "report-btn",
              onClick: generateSummary,
            },
            "Generate Summary"
          ),
        ]),
      ]),
      e("aside", { className: "incident-col" }, [
        e("div", { key: "head", className: "incident-head" }, [
          e("h2", { key: "h" }, "Incident Log"),
          e("span", { key: "c", className: "event-count" }, `${incidents.length} Events`),
        ]),
        e(
          "div",
          { key: "panel", className: "incident-panel", ref: incidentPanelRef },
          incidents.length === 0
            ? e("div", { className: "incident-empty" }, "No incidents detected yet.")
            : e(
                "ul",
                { className: "incident-group-list" },
                incidents.map((item) =>
                  e("li", { key: item.id, className: `incident-item ${item.level.toLowerCase()}` }, [
                    e("span", { key: "dot", className: "dot" }),
                    e("div", { key: "body", className: "incident-compact" }, [
                      e("strong", { key: "sev", className: `risk-pill ${item.level.toLowerCase()}` }, item.level),
                      e("strong", { key: "type", className: "type-pill" }, formatRiskType(item.riskType)),
                    ]),
                  ])
                )
              )
        ),
        e("div", { key: "risk", className: "risk-line" }, [
          "Risk Status: ",
          e("strong", { id: "sentinel-risk", "data-risk": risk }, risk),
        ]),
      ])
    )
  );
}
