import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import av
import numpy as np
from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.processors.base_processor import AudioProcessor, VideoProcessor
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.plugins import deepgram, getstream, openai

load_dotenv()
logger = logging.getLogger(__name__)


SENTINEL_INSTRUCTIONS = """
You are a real-time workplace safety AI.

You analyze:
- Missing helmet
- Unsafe phone usage
- Loud noise events

Classify risk level as LOW, MEDIUM, or HIGH.
Call tools when needed.
For a missing helmet with active worker, treat as HIGH risk.
When helmet is missing, say exactly: "High risk detected. Worker without helmet."
Then call `log_incident` with: "High risk - helmet missing".
"""


def log_incident(data: str) -> None:
    print("Incident Logged:", data)


def speak_warning(message: str) -> None:
    print("Warning:", message)


def _normalize_audio(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32)
    if np.issubdtype(samples.dtype, np.integer):
        max_val = float(np.iinfo(samples.dtype).max)
        return samples.astype(np.float32) / max_val
    return samples.astype(np.float32)


class LoudNoiseProcessor(AudioProcessor):
    name = "loud_noise"

    def __init__(
        self,
        rms_low_threshold: float = 0.18,
        rms_medium_threshold: float = 0.28,
        rms_high_threshold: float = 0.45,
        cooldown_seconds: float = 3.0,
    ):
        self.rms_low_threshold = rms_low_threshold
        self.rms_medium_threshold = rms_medium_threshold
        self.rms_high_threshold = rms_high_threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_trigger = 0.0
        self._agent: Optional[Agent] = None
        self._inflight_task: Optional[asyncio.Task] = None

    def attach_agent(self, agent: Agent) -> None:
        self._agent = agent

    async def process_audio(self, audio_data) -> None:
        samples = getattr(audio_data, "samples", None)
        if samples is None:
            return
        if hasattr(samples, "reshape"):
            samples = samples.reshape(-1)
        samples = _normalize_audio(samples)
        if samples.size == 0:
            return
        rms = float(np.sqrt(np.mean(samples * samples)))
        logger.debug(
            "LoudNoiseProcessor RMS=%.3f bands(low>=%.3f, med>=%.3f, high>=%.3f)",
            rms,
            self.rms_low_threshold,
            self.rms_medium_threshold,
            self.rms_high_threshold,
        )
        if rms < self.rms_low_threshold:
            return

        risk = self._risk_from_rms(rms)
        now = time.monotonic()
        if now - self._last_trigger < self.cooldown_seconds:
            return
        self._last_trigger = now
        if not self._agent:
            return
        if self._inflight_task and not self._inflight_task.done():
            return

        async def _trigger() -> None:
            if not self._agent:
                return
            logger.info("Loud noise trigger fired (rms=%.3f, risk=%s)", rms, risk)
            # Emit a guaranteed UI event so Incident Log updates even if the LLM chooses not to call tools.
            await self._agent.send_custom_event(
                {
                    "type": "incident_log",
                    "message": f"{risk.title()} risk - loud noise detected (rms={rms:.2f})",
                }
            )
            await self._agent.send_custom_event({"type": "risk_status", "risk": risk})
            await self._agent.simple_response(
                f"Loud noise event detected with {risk} risk (rms={rms:.2f}). "
                "Respond briefly and call tools if needed."
            )

        self._inflight_task = asyncio.create_task(_trigger())

    async def close(self) -> None:
        return

    def _risk_from_rms(self, rms: float) -> str:
        if rms >= self.rms_high_threshold:
            return "HIGH"
        if rms >= self.rms_medium_threshold:
            return "MEDIUM"
        return "LOW"


class YoloSafetyProcessor(VideoProcessor):
    name = "yolo_safety"

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        imgsz: int = 640,
        device: str = "cpu",
        fps: int = 2,
        person_class: str = "person",
        helmet_class: str = "helmet",
        phone_class: str = "cell phone",
        require_person_for_helmet: bool = True,
        cooldown_seconds: float = 5.0,
        classes: Optional[Sequence[str]] = None,
    ):
        from ultralytics import YOLO

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device
        self.fps = fps
        self.person_class = person_class
        self.helmet_class = helmet_class
        self.phone_class = phone_class
        self.require_person_for_helmet = require_person_for_helmet
        self.cooldown_seconds = cooldown_seconds
        self.classes = list(classes or ["person", "helmet", "cell phone"])

        self._agent: Optional[Agent] = None
        self._video_forwarder: Optional[VideoForwarder] = None
        self._shutdown = False
        self._last_helmet_trigger = 0.0
        self._last_phone_trigger = 0.0
        self._last_summary_trigger = 0.0
        self._warned_missing_helmet_class = False
        self._last_counts_signature: Optional[tuple[int, int, int]] = None

        self._model = YOLO(self.model_path)
        self._model.to(self.device)
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="yolo_safety"
        )
        model_names = getattr(self._model, "names", {}) or {}
        if isinstance(model_names, dict):
            self._available_classes = set(str(v) for v in model_names.values())
        else:
            self._available_classes = set(str(v) for v in model_names)
        self._selected_class_ids = self._resolve_class_ids(model_names, self.classes)

        self._supports_person = self.person_class in self._available_classes
        self._supports_helmet = self.helmet_class in self._available_classes
        self._supports_phone = self.phone_class in self._available_classes

    def attach_agent(self, agent: Agent) -> None:
        self._agent = agent

    async def process_video(
        self,
        track,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._on_frame)

        self._video_forwarder = (
            shared_forwarder
            if shared_forwarder
            else VideoForwarder(
                track,
                max_buffer=max(1, self.fps),
                fps=self.fps,
                name="yolo_safety_forwarder",
            )
        )
        self._video_forwarder.add_frame_handler(
            self._on_frame, fps=float(self.fps), name="yolo_safety"
        )

    async def _on_frame(self, frame: av.VideoFrame) -> None:
        if self._shutdown:
            return
        frame_array = frame.to_ndarray(format="rgb24")
        event_loop = asyncio.get_event_loop()
        classes = await event_loop.run_in_executor(
            self._executor, self._infer_classes, frame_array
        )
        if not classes:
            return
        await self._handle_detections(classes)

    def _infer_classes(self, frame_array: np.ndarray) -> Sequence[str]:
        results = self._model(
            frame_array,
            verbose=False,
            conf=self.conf_threshold,
            device=self.device,
            imgsz=self.imgsz,
            classes=self._selected_class_ids if self._selected_class_ids else None,
        )
        if not results:
            return []
        result = results[0]
        if result.boxes is None or result.boxes.cls is None:
            return []
        names = result.names or {}
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        return [names.get(int(cls_id), str(cls_id)) for cls_id in class_ids]

    def _resolve_class_ids(
        self, names: dict | list, requested_names: Sequence[str]
    ) -> list[int]:
        name_to_id: dict[str, int] = {}
        if isinstance(names, dict):
            for class_id, class_name in names.items():
                name_to_id[str(class_name)] = int(class_id)
        else:
            for class_id, class_name in enumerate(names):
                name_to_id[str(class_name)] = int(class_id)

        class_ids: list[int] = []
        for class_name in requested_names:
            if class_name in name_to_id:
                class_ids.append(name_to_id[class_name])
            else:
                logger.warning(
                    "Requested class '%s' not found in model '%s'",
                    class_name,
                    self.model_path,
                )
        return class_ids

    async def _handle_detections(self, classes: Sequence[str]) -> None:
        if not self._agent:
            return
        now = time.monotonic()
        people_count = sum(1 for c in classes if c == self.person_class)
        helmet_count = sum(1 for c in classes if c == self.helmet_class)
        phone_count = sum(1 for c in classes if c == self.phone_class)
        person_present = self._supports_person and people_count > 0
        helmet_present = self._supports_helmet and helmet_count > 0
        phone_present = self._supports_phone and phone_count > 0

        if not self._supports_helmet and not self._warned_missing_helmet_class:
            self._warned_missing_helmet_class = True
            logger.warning(
                "Helmet class '%s' not found in model '%s'. Available classes: %s",
                self.helmet_class,
                self.model_path,
                sorted(self._available_classes),
            )

        if (
            self._supports_helmet
            and (not self.require_person_for_helmet or person_present)
            and not helmet_present
        ):
            if now - self._last_helmet_trigger >= self.cooldown_seconds:
                self._last_helmet_trigger = now
                await self._agent.send_custom_event(
                    {
                        "type": "incident_log",
                        "risk": "HIGH",
                        "message": "High risk - helmet missing",
                    }
                )
                await self._agent.send_custom_event({"type": "risk_status", "risk": "HIGH"})
                await self._agent.simple_response(
                    "Helmet missing detected in video. Respond with the required phrase and call tools."
                )

        if phone_present and now - self._last_phone_trigger >= self.cooldown_seconds:
            self._last_phone_trigger = now
            await self._agent.simple_response(
                "Unsafe phone usage detected. Classify risk and call tools if needed."
            )

        # Structured safety summary for LLM reasoning.
        counts_signature = (people_count, helmet_count, phone_count)
        if counts_signature != self._last_counts_signature and (
            now - self._last_summary_trigger >= self.cooldown_seconds
        ):
            self._last_counts_signature = counts_signature
            self._last_summary_trigger = now
            payload = {
                "people_detected": people_count,
                "helmets_detected": helmet_count,
                "phones_detected": phone_count,
            }
            await self._agent.simple_response(
                "You are a workplace safety AI. If a person is detected without a helmet, "
                "classify HIGH risk. If a person is using a phone, classify MEDIUM risk. "
                f"Detection payload: {json.dumps(payload)}"
            )

    async def stop_processing(self) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._on_frame)
            self._video_forwarder = None

    async def close(self) -> None:
        self._shutdown = True
        await self.stop_processing()
        self._executor.shutdown(wait=False)


async def create_agent(**kwargs) -> Agent:
    llm = openai.Realtime(fps=2)

    yolo_model_path = os.getenv("SENTINEL_YOLO_MODEL", "yolo11n.pt")
    yolo_conf = float(os.getenv("SENTINEL_YOLO_CONF", "0.35"))
    yolo_imgsz = int(os.getenv("SENTINEL_YOLO_IMGSZ", "640"))
    yolo_device = os.getenv("SENTINEL_YOLO_DEVICE", "cpu")
    yolo_fps = int(os.getenv("SENTINEL_YOLO_FPS", "2"))
    yolo_classes = [
        x.strip()
        for x in os.getenv("SENTINEL_YOLO_CLASSES", "person,helmet,cell phone").split(",")
        if x.strip()
    ]
    person_class = os.getenv("SENTINEL_PERSON_CLASS", "person")
    helmet_class = os.getenv("SENTINEL_HELMET_CLASS", "helmet")
    phone_class = os.getenv("SENTINEL_PHONE_CLASS", "cell phone")
    require_person = os.getenv("SENTINEL_REQUIRE_PERSON", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    event_cooldown = float(os.getenv("SENTINEL_EVENT_COOLDOWN_SECONDS", "5"))

    noise_low = float(os.getenv("SENTINEL_NOISE_RMS_LOW", "0.18"))
    noise_medium = float(os.getenv("SENTINEL_NOISE_RMS_MEDIUM", "0.28"))
    noise_high = float(os.getenv("SENTINEL_NOISE_RMS_HIGH", "0.45"))
    noise_cooldown = float(os.getenv("SENTINEL_NOISE_COOLDOWN", "8"))
    audio_only = os.getenv("SENTINEL_AUDIO_ONLY", "1").lower() in ("1", "true", "yes")

    # Fallback for placeholder/invalid absolute paths in .env.
    if os.path.isabs(yolo_model_path) and not os.path.exists(yolo_model_path):
        print(
            f"SENTINEL_YOLO_MODEL path not found: {yolo_model_path}. Falling back to yolo11n.pt"
        )
        yolo_model_path = "yolo11n.pt"

    processors = [
        LoudNoiseProcessor(
            rms_low_threshold=noise_low,
            rms_medium_threshold=noise_medium,
            rms_high_threshold=noise_high,
            cooldown_seconds=noise_cooldown,
        ),
    ]
    if not audio_only:
        processors.insert(
            0,
            YoloSafetyProcessor(
                model_path=yolo_model_path,
                conf_threshold=yolo_conf,
                imgsz=yolo_imgsz,
                device=yolo_device,
                fps=yolo_fps,
                person_class=person_class,
                helmet_class=helmet_class,
                phone_class=phone_class,
                require_person_for_helmet=require_person,
                cooldown_seconds=event_cooldown,
                classes=yolo_classes,
            ),
        )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Sentinel AI", id="sentinel"),
        instructions=SENTINEL_INSTRUCTIONS,
        processors=processors,
        llm=llm,
        stt=deepgram.STT(),
    )

    @llm.register_function(
        name="log_incident",
        description="Log a safety incident and update UI incident panel.",
    )
    async def log_incident_tool(data: str) -> str:
        log_incident(data)
        await agent.send_custom_event(
            {
                "type": "incident_log",
                "message": data,
            }
        )
        lowered = data.lower()
        risk = "LOW"
        if "high" in lowered:
            risk = "HIGH"
        elif "medium" in lowered:
            risk = "MEDIUM"
        await agent.send_custom_event({"type": "risk_status", "risk": risk})
        return f"incident_logged:{data}"

    @llm.register_function(
        name="speak_warning",
        description="Speak a concise warning message and update risk status UI if severity is present.",
    )
    async def speak_warning_tool(message: str) -> str:
        speak_warning(message)
        await agent.send_custom_event({"type": "warning", "message": message})
        lowered = message.lower()
        if "high risk" in lowered:
            await agent.send_custom_event({"type": "risk_status", "risk": "HIGH"})
        elif "medium risk" in lowered:
            await agent.send_custom_event({"type": "risk_status", "risk": "MEDIUM"})
        elif "low risk" in lowered:
            await agent.send_custom_event({"type": "risk_status", "risk": "LOW"})
        return f"warning_spoken:{message}"

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    # Workaround for current getstream transport: ensure agent user is created
    # before create_call so created_by_id is populated server-side.
    if hasattr(agent.edge, "create_user"):
        await agent.edge.create_user(agent.agent_user)
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    # Set AGENT_IDLE_TIMEOUT_SECONDS=0 (or negative) in .env to effectively disable idle timeout.
    idle_timeout_seconds = float(os.getenv("AGENT_IDLE_TIMEOUT_SECONDS", "0"))
    if idle_timeout_seconds <= 0:
        idle_timeout_seconds = 315360000.0  # ~10 years

    Runner(
        AgentLauncher(
            create_agent=create_agent,
            join_call=join_call,
            agent_idle_timeout=idle_timeout_seconds,
        )
    ).cli()



