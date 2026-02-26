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

    if hasattr(agent, "register_function"):
        agent.register_function(log_incident)
        agent.register_function(speak_warning)
    else:
        llm.register_function()(log_incident)
        llm.register_function()(speak_warning)

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()



