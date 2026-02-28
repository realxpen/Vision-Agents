import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import av
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.processors.base_processor import AudioProcessor, VideoProcessor
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.plugins import getstream, openai

load_dotenv()
logger = logging.getLogger(__name__)


def _configure_demo_logging() -> None:
    # Keep terminal output focused on key safety events during demos.
    quiet_logs = os.getenv("SENTINEL_QUIET_LOGS", "1").lower() in ("1", "true", "yes")
    if not quiet_logs:
        return

    noisy_loggers = (
        "getstream",
        "getstream.video",
        "websockets",
        "httpx",
        "httpcore",
        "aiortc",
        "pyee",
    )
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


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


class RoboflowSafetyProcessor(VideoProcessor):
    name = "roboflow_safety"

    def __init__(
        self,
        api_key: str,
        helmet_model_id: str,
        phone_model_id: Optional[str] = None,
        conf_threshold: float = 0.35,
        helmet_conf_threshold: Optional[float] = None,
        phone_conf_threshold: Optional[float] = None,
        fps: int = 2,
        person_class: str = "person",
        helmet_class: str = "helmet",
        phone_class: str = "cell phone",
        require_person_for_helmet: bool = True,
        require_person_for_phone: bool = True,
        phone_confirm_frames: int = 3,
        cooldown_seconds: float = 5.0,
        timeout_seconds: float = 20.0,
    ):
        self.api_key = api_key
        self.helmet_model_id = helmet_model_id
        self.phone_model_id = phone_model_id
        self.conf_threshold = conf_threshold
        self.helmet_conf_threshold = (
            helmet_conf_threshold if helmet_conf_threshold is not None else conf_threshold
        )
        self.phone_conf_threshold = (
            phone_conf_threshold if phone_conf_threshold is not None else conf_threshold
        )
        self.fps = fps
        self.person_class = person_class
        self.helmet_class = helmet_class
        self.phone_class = phone_class
        self.require_person_for_helmet = require_person_for_helmet
        self.require_person_for_phone = require_person_for_phone
        self.cooldown_seconds = cooldown_seconds
        self.timeout_seconds = timeout_seconds

        self._agent: Optional[Agent] = None
        self._video_forwarder: Optional[VideoForwarder] = None
        self._shutdown = False
        self._last_helmet_trigger = 0.0
        self._last_phone_trigger = 0.0
        self._last_summary_trigger = 0.0
        self._last_counts_signature: Optional[tuple[int, int, int]] = None
        self._helmet_missing_active = False
        self._phone_streak = 0
        self._phone_confirm_frames = max(1, phone_confirm_frames)
        self._last_raw_log_ts = 0.0
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="roboflow_safety"
        )
        self._person_aliases = {
            "person",
            "worker",
            "human",
            "people",
        }
        self._helmet_aliases = {
            "helmet",
            "hardhat",
            "hard_hat",
            "helmet_worn",
            "helmet worn",
            "safety_helmet",
            "head-helmet",
            "head_helmet",
        }
        self._phone_aliases = {
            "cell phone",
            "cellphone",
            "mobile phone",
            "mobile_phone",
            "phone",
            "smartphone",
        }
        self._no_helmet_aliases = {
            "no helmet",
            "no_helmet",
            "without helmet",
            "without_helmet",
            "helmet not worn",
            "helmet_not_worn",
            "a helmet not worn",
            "rejected helmet",
            "rejected_helmet",
        }

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
                name="roboflow_safety_forwarder",
            )
        )
        self._video_forwarder.add_frame_handler(
            self._on_frame, fps=float(self.fps), name="roboflow_safety"
        )

    async def _on_frame(self, frame: av.VideoFrame) -> None:
        if self._shutdown:
            return
        frame_array = frame.to_ndarray(format="bgr24")
        event_loop = asyncio.get_event_loop()
        classes = await event_loop.run_in_executor(
            self._executor, self._infer_classes, frame_array
        )
        if classes is None:
            return
        await self._handle_detections(classes)

    def _infer_classes(self, frame_array: np.ndarray) -> Optional[list[str]]:
        ok, encoded = cv2.imencode(".jpg", frame_array)
        if not ok:
            return None
        image_bytes = encoded.tobytes()
        classes: list[str] = []
        helmet_classes = self._infer_classes_from_model(
            model_id=self.helmet_model_id,
            image_bytes=image_bytes,
            min_conf=self.helmet_conf_threshold,
        )
        if helmet_classes is None:
            return None
        classes.extend(helmet_classes)

        if self.phone_model_id and self.phone_model_id != self.helmet_model_id:
            phone_classes = self._infer_classes_from_model(
                model_id=self.phone_model_id,
                image_bytes=image_bytes,
                min_conf=self.phone_conf_threshold,
            )
            if phone_classes:
                classes.extend(phone_classes)
        return classes

    def _infer_classes_from_model(
        self, model_id: str, image_bytes: bytes, min_conf: float
    ) -> Optional[list[str]]:
        try:
            response = requests.post(
                f"https://detect.roboflow.com/{model_id}",
                params={
                    "api_key": self.api_key,
                    "confidence": int(min_conf * 100),
                },
                files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Roboflow inference failed for model '%s': %s", model_id, exc)
            return None

        predictions = payload.get("predictions", []) if isinstance(payload, dict) else []
        now = time.monotonic()
        if now - self._last_raw_log_ts >= 3.0:
            preview = []
            for pred in predictions[:5]:
                if isinstance(pred, dict):
                    preview.append(
                        {
                            "class": pred.get("class"),
                            "confidence": pred.get("confidence"),
                        }
                    )
            logger.debug(
                "Roboflow raw model=%s predictions=%d sample=%s",
                model_id,
                len(predictions),
                preview,
            )
            self._last_raw_log_ts = now
        classes: list[str] = []
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            cls = pred.get("class")
            conf = float(pred.get("confidence", 0.0))
            if cls and conf >= min_conf:
                classes.append(self._normalize_class_name(str(cls)))
        return classes

    def _normalize_class_name(self, raw: str) -> str:
        normalized = raw.strip().lower().replace("-", " ").replace("_", " ")
        if normalized in self._person_aliases:
            return self.person_class
        if normalized in self._helmet_aliases:
            return self.helmet_class
        if normalized in self._phone_aliases:
            return self.phone_class
        if normalized in self._no_helmet_aliases:
            return "__NO_HELMET__"
        # Fallback substring matching for custom label variants.
        if "phone" in normalized:
            return self.phone_class
        if "helmet" in normalized or "hardhat" in normalized or "hard hat" in normalized:
            if "not" in normalized or "without" in normalized or "no " in normalized:
                return "__NO_HELMET__"
            return self.helmet_class
        if "person" in normalized or "worker" in normalized or "human" in normalized:
            return self.person_class
        return raw

    async def _handle_detections(self, classes: Sequence[str]) -> None:
        if not self._agent:
            return
        now = time.monotonic()
        people_count = sum(1 for c in classes if c == self.person_class)
        helmet_count = sum(1 for c in classes if c == self.helmet_class)
        phone_count = sum(1 for c in classes if c == self.phone_class)
        no_helmet_count = sum(1 for c in classes if c == "__NO_HELMET__")
        person_present = people_count > 0
        helmet_present = helmet_count > 0
        phone_present = phone_count > 0
        no_helmet_present = no_helmet_count > 0
        counts_signature = (people_count, helmet_count, phone_count)
        if counts_signature != self._last_counts_signature:
            logger.debug(
                "Roboflow detections: people=%d helmets=%d phones=%d classes=%s",
                people_count,
                helmet_count,
                phone_count,
                list(classes),
            )

        helmet_missing = no_helmet_present or (
            (not self.require_person_for_helmet or person_present)
            and not helmet_present
            and (person_present or phone_present)
        )
        if helmet_missing and now - self._last_helmet_trigger >= self.cooldown_seconds:
            self._last_helmet_trigger = now
            self._helmet_missing_active = True
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
        elif self._helmet_missing_active and helmet_present:
            self._helmet_missing_active = False
            await self._agent.send_custom_event(
                {
                    "type": "incident_log",
                    "risk": "LOW",
                    "message": "Low risk - helmet worn",
                }
            )
            await self._agent.send_custom_event({"type": "risk_status", "risk": "LOW"})

        phone_gate = phone_present and (
            person_present or not self.require_person_for_phone
        )
        if phone_gate:
            self._phone_streak += 1
        else:
            self._phone_streak = 0

        if (
            phone_gate
            and
            self._phone_streak >= self._phone_confirm_frames
            and now - self._last_phone_trigger >= self.cooldown_seconds
        ):
            self._last_phone_trigger = now
            await self._agent.send_custom_event(
                {
                    "type": "incident_log",
                    "risk": "MEDIUM",
                    "message": "Medium risk - phone usage detected",
                }
            )
            await self._agent.send_custom_event(
                {"type": "risk_status", "risk": "MEDIUM"}
            )
            await self._agent.simple_response(
                "Unsafe phone usage detected. Classify risk and call tools if needed."
            )

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
    helmet_conf = float(os.getenv("SENTINEL_HELMET_CONF", str(yolo_conf)))
    phone_conf = float(os.getenv("SENTINEL_PHONE_CONF", "0.60"))
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
    require_person_for_phone = os.getenv(
        "SENTINEL_REQUIRE_PERSON_FOR_PHONE", "1"
    ).lower() in ("1", "true", "yes")
    phone_confirm_frames = max(1, int(os.getenv("SENTINEL_PHONE_CONFIRM_FRAMES", "3")))
    event_cooldown = float(os.getenv("SENTINEL_EVENT_COOLDOWN_SECONDS", "5"))

    noise_low = float(os.getenv("SENTINEL_NOISE_RMS_LOW", "0.18"))
    noise_medium = float(os.getenv("SENTINEL_NOISE_RMS_MEDIUM", "0.28"))
    noise_high = float(os.getenv("SENTINEL_NOISE_RMS_HIGH", "0.45"))
    noise_cooldown = float(os.getenv("SENTINEL_NOISE_COOLDOWN", "8"))
    audio_only = os.getenv("SENTINEL_AUDIO_ONLY", "1").lower() in ("1", "true", "yes")
    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    roboflow_model_id = os.getenv("ROBOFLOW_MODEL_ID", "").strip()
    roboflow_helmet_model_id = os.getenv("ROBOFLOW_HELMET_MODEL_ID", "").strip()
    roboflow_phone_model_id = os.getenv("ROBOFLOW_PHONE_MODEL_ID", "").strip()
    if not roboflow_helmet_model_id:
        roboflow_helmet_model_id = roboflow_model_id
    use_roboflow = os.getenv("SENTINEL_USE_ROBOFLOW", "1").lower() in (
        "1",
        "true",
        "yes",
    ) and bool(roboflow_api_key and roboflow_helmet_model_id)

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
        if use_roboflow:
            logger.info(
                "Using Roboflow hosted model(s): helmet=%s phone=%s",
                roboflow_helmet_model_id,
                roboflow_phone_model_id or roboflow_helmet_model_id,
            )
            processors.insert(
                0,
                RoboflowSafetyProcessor(
                    api_key=roboflow_api_key,
                    helmet_model_id=roboflow_helmet_model_id,
                    phone_model_id=roboflow_phone_model_id or None,
                    conf_threshold=yolo_conf,
                    helmet_conf_threshold=helmet_conf,
                    phone_conf_threshold=phone_conf,
                    fps=yolo_fps,
                    person_class=person_class,
                    helmet_class=helmet_class,
                    phone_class=phone_class,
                    require_person_for_helmet=require_person,
                    require_person_for_phone=require_person_for_phone,
                    phone_confirm_frames=phone_confirm_frames,
                    cooldown_seconds=event_cooldown,
                ),
            )
        else:
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
    max_retries = int(os.getenv("SENTINEL_JOIN_RETRIES", "3"))
    retry_delay = float(os.getenv("SENTINEL_JOIN_RETRY_DELAY_SECONDS", "2"))
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            call = await agent.create_call(call_type, call_id)
            async with agent.join(call, participant_wait_timeout=0):
                await agent.finish()
            return
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Join attempt %d/%d failed: %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)

    if last_exc:
        raise last_exc


if __name__ == "__main__":
    _configure_demo_logging()
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



