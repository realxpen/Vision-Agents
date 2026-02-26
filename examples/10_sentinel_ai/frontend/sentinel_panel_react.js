import React from "https://esm.sh/react@18.3.1";

const e = React.createElement;

export function SentinelPanel({ call, onStartMonitoring }) {
  const [risk, setRisk] = React.useState("LOW");
  const [incidents, setIncidents] = React.useState([]);
  const [live, setLive] = React.useState(false);
  const [cameraText, setCameraText] = React.useState(
    "Camera offline. Click Start Monitoring."
  );

  React.useEffect(() => {
    if (!call || typeof call.on !== "function") return undefined;
    const detach = call.on("custom", (event) => {
      const payload = event?.custom || event;
      if (!payload || typeof payload !== "object") return;

      if (payload.type === "risk_status" && payload.risk) {
        setRisk(String(payload.risk).toUpperCase());
      }

      if (payload.type === "incident_log" && payload.message) {
        setIncidents((prev) => [String(payload.message), ...prev].slice(0, 50));
      }

      if (payload.type === "warning" && payload.message) {
        setIncidents((prev) => [String(payload.message), ...prev].slice(0, 50));
      }
    });

    return () => {
      if (typeof detach === "function") detach();
    };
  }, [call]);

  function startMonitoring() {
    setLive(true);
    setCameraText("Monitoring started. Waiting for activity...");
    if (typeof onStartMonitoring === "function") onStartMonitoring();
  }

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
          e("h1", { key: "h" }, "VisionGuard AI"),
          e("p", { key: "p" }, "Real-Time Safety & Risk Detection Agent"),
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
        e(
          "button",
          {
            key: "btn",
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
          e("div", { className: "camera-overlay" }, [
            e("div", { key: "icon", className: "camera-icon" }, "[CAM]"),
            e("p", { key: "txt" }, cameraText),
          ])
        ),
        e("div", { key: "report", className: "session-report" }, [
          e("div", { key: "txt" }, [
            e("h3", { key: "h" }, "Session Report"),
            e("p", { key: "p" }, "Generate a summary of all detected incidents."),
          ]),
          e(
            "button",
            { key: "b", type: "button", className: "report-btn" },
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
          { key: "panel", className: "incident-panel" },
          incidents.length === 0
            ? e("div", { className: "incident-empty" }, "No incidents detected yet.")
            : e(
                "ul",
                { id: "sentinel-incidents" },
                incidents.map((text, idx) =>
                  e("li", { key: `${idx}-${text}`, className: "incident-item" }, [
                    e("span", { key: "dot", className: "dot" }),
                    e("div", { key: "body" }, [
                      e("strong", { key: "title" }, "Safety Alert"),
                      e("p", { key: "text" }, text),
                    ]),
                  ])
                )
              )
        ),
        e("div", { key: "risk", className: "risk-line" }, [
          "Risk Status: ",
          e("strong", { "data-risk": risk }, risk),
        ]),
      ])
    )
  );
}
