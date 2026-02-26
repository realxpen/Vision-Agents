import React from "https://esm.sh/react@18.3.1";
import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";
import { SentinelPanel } from "./sentinel_panel_react.js";

const e = React.createElement;

class MockCall {
  constructor() {
    this.handlers = [];
  }

  on(eventType, handler) {
    if (eventType !== "custom") return () => {};
    this.handlers.push(handler);
    return () => {
      this.handlers = this.handlers.filter((h) => h !== handler);
    };
  }

  emitCustom(custom) {
    for (const handler of this.handlers) {
      handler({ custom });
    }
  }
}

const call = new MockCall();

function App() {
  return e("div", null, [
    e("div", { key: "actions", className: "preview-actions" }, [
      e(
        "button",
        {
          key: "low",
          type: "button",
          onClick: () => {
            call.emitCustom({ type: "risk_status", risk: "LOW" });
            call.emitCustom({
              type: "incident_log",
              message: "Low risk - routine activity.",
            });
          },
        },
        "Simulate LOW"
      ),
      e(
        "button",
        {
          key: "medium",
          type: "button",
          onClick: () => {
            call.emitCustom({ type: "risk_status", risk: "MEDIUM" });
            call.emitCustom({
              type: "incident_log",
              message: "Medium risk - unsafe phone usage.",
            });
          },
        },
        "Simulate MEDIUM"
      ),
      e(
        "button",
        {
          key: "high",
          type: "button",
          onClick: () => {
            call.emitCustom({ type: "risk_status", risk: "HIGH" });
            call.emitCustom({
              type: "incident_log",
              message: "High risk - helmet missing",
            });
            call.emitCustom({
              type: "warning",
              message: "High risk detected. Worker without helmet.",
            });
          },
        },
        "Simulate HIGH"
      ),
    ]),
    e(SentinelPanel, { key: "panel", call }),
  ]);
}

createRoot(document.getElementById("sentinel-root")).render(e(App));
