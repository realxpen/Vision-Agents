import os

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.plugins import deepgram, getstream, openai, ultralytics

load_dotenv()


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


async def create_agent(**kwargs) -> Agent:
    llm = openai.Realtime(fps=2)

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Sentinel AI", id="sentinel"),
        instructions=SENTINEL_INSTRUCTIONS,
        processors=[
            ultralytics.YOLOPoseProcessor(
                model_path="yolo26n-pose.pt",
                fps=2,
            )
        ],
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



