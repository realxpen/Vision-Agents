import React from "https://esm.sh/react@18.3.1";
import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";
import { SentinelPanel } from "./sentinel_panel_react.js";

const e = React.createElement;

async function getStreamVideoClientClass() {
  const mod = await import("https://esm.sh/@stream-io/video-client");
  if (mod.StreamVideoClient) return mod.StreamVideoClient;
  if (mod.default?.StreamVideoClient) return mod.default.StreamVideoClient;
  if (typeof mod.default === "function") return mod.default;
  throw new Error("Could not load StreamVideoClient from @stream-io/video-client");
}

function App() {
  const [config, setConfig] = React.useState(null);
  const [status, setStatus] = React.useState("Loading config...");
  const [error, setError] = React.useState("");
  const [connecting, setConnecting] = React.useState(false);
  const [connectedCall, setConnectedCall] = React.useState(null);
  const [publishVideo, setPublishVideo] = React.useState(false);

  const clientRef = React.useRef(null);
  const callRef = React.useRef(null);

  async function disconnectCurrent() {
    const call = callRef.current;
    const client = clientRef.current;

    callRef.current = null;
    clientRef.current = null;
    setConnectedCall(null);

    try {
      if (call && typeof call.leave === "function") await call.leave();
    } catch (_) {}

    try {
      if (client && typeof client.disconnectUser === "function") {
        await client.disconnectUser();
      }
    } catch (_) {}

    setStatus("Disconnected");
  }

  async function loadConfig() {
    setError("");
    setStatus("Loading config...");
    try {
      const resp = await fetch("/frontend-config");
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Failed to load frontend config");
      setConfig(data);
      setStatus("Config loaded");
    } catch (err) {
      setError(String(err?.message || err));
      setStatus("Config error");
    }
  }

  async function connect() {
    if (!config) {
      setError("Config not loaded yet.");
      return;
    }

    setError("");
    setConnecting(true);
    setStatus("Connecting...");

    try {
      await disconnectCurrent();

      const StreamVideoClient = await getStreamVideoClientClass();
      const client = new StreamVideoClient({
        apiKey: config.apiKey,
        user: { id: config.userId, name: config.userId },
        token: config.token,
      });

      const call = client.call(config.callType, config.callId);
      // Prefer VP8 for widest decoder compatibility on the Python side.
      if (typeof call.setPreferredCodec === "function") {
        await call.setPreferredCodec("vp8");
      } else if (typeof call.setPreferredVideoCodec === "function") {
        await call.setPreferredVideoCodec("vp8");
      }
      await call.join({ create: true });

      clientRef.current = client;
      callRef.current = call;
      setConnectedCall(call);
      setStatus(`Connected: ${config.callType}/${config.callId}`);
    } catch (err) {
      setError(String(err?.message || err));
      setStatus("Connection failed");
      await disconnectCurrent();
    } finally {
      setConnecting(false);
    }
  }

  React.useEffect(() => {
    loadConfig();
    return () => {
      disconnectCurrent().catch(() => {});
    };
  }, []);

  return e("div", null, [
    e("div", { key: "cfg", className: "preview-config" }, [
      e("h3", { key: "h" }, "Real Stream Connection"),
      config
        ? e("div", { key: "meta", className: "meta" }, [
            e("div", { key: "k" }, `API Key: ${config.apiKey}`),
            e("div", { key: "u" }, `User ID: ${config.userId}`),
            e("div", { key: "c" }, `Call: ${config.callType}/${config.callId}`),
          ])
        : null,
      e("div", { key: "actions", className: "preview-actions" }, [
        e(
          "button",
          {
            key: "reload",
            type: "button",
            onClick: loadConfig,
            disabled: connecting,
          },
          "Reload Config"
        ),
        e(
          "button",
          {
            key: "connect",
            type: "button",
            onClick: connect,
            disabled: connecting || !config || !!connectedCall,
          },
          connecting ? "Connecting..." : "Connect to Call"
        ),
        e(
          "button",
          {
            key: "disconnect",
            type: "button",
            onClick: () => disconnectCurrent(),
            disabled: connecting || !connectedCall,
          },
          "Disconnect"
        ),
        e("label", { key: "videoMode", className: "video-mode-toggle" }, [
          e("input", {
            key: "chk",
            type: "checkbox",
            checked: publishVideo,
            onChange: (ev) => setPublishVideo(Boolean(ev.target.checked)),
            disabled: connecting || !!connectedCall,
          }),
          e("span", { key: "txt" }, "Helmet Mode (send camera to agent)"),
        ]),
        e("span", { key: "status", className: "status" }, `Status: ${status}`),
      ]),
      error ? e("div", { key: "err", className: "error" }, error) : null,
    ]),
    e(SentinelPanel, {
      key: "panel",
      call: connectedCall,
      onStartMonitoring: async () => {
        if (!connectedCall) return;
        if (publishVideo) {
          try {
            if (connectedCall.camera?.enable) await connectedCall.camera.enable();
          } catch (e2) {
            console.warn("Could not enable call camera:", e2);
          }
        } else {
          // Keep call media audio-only by default for stability.
          // The panel still shows local camera preview from getUserMedia.
          try {
            if (connectedCall.camera?.disable) await connectedCall.camera.disable();
          } catch (e2) {
            console.warn("Could not disable call camera:", e2);
          }
        }
        try {
          if (connectedCall.microphone?.enable) {
            await connectedCall.microphone.enable();
          }
        } catch (e2) {
          console.warn("Could not enable call microphone:", e2);
        }
      },
      onStopMonitoring: async () => {
        if (!connectedCall) return;
        try {
          if (connectedCall.camera?.disable) await connectedCall.camera.disable();
        } catch (e2) {
          console.warn("Could not disable call camera:", e2);
        }
        try {
          if (connectedCall.microphone?.disable) {
            await connectedCall.microphone.disable();
          }
        } catch (e2) {
          console.warn("Could not disable call microphone:", e2);
        }
      },
    }),
  ]);
}

createRoot(document.getElementById("sentinel-root")).render(e(App));
