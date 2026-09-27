import base64
import logging
from collections.abc import Callable
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.types import Command
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from pydantic import BaseModel, Field

from src.llm import get_llm
from src.prompts import REPORT_WRITER_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT, VISION_AGENT_SYSTEM_PROMPT
from src.state import MSAState

logger = logging.getLogger(__name__)
register_heif_opener()

TODAY = f"{date.today():%B %d, %Y}"
MAX_STEPS = 6
AGENTS = ["vision_agent", "audio_agent", "research_agent"]
REQUIRES = {"vision_agent": "image_path", "audio_agent": "audio_path"}
MODEL_FORMATS = {"PNG", "JPEG", "GIF", "WEBP"}


def specialist_agent(role: str, system_prompt: str, tools: list, build_input, output_key: str):
    agent = create_agent(
        get_llm(role),
        tools=tools,
        system_prompt=f"{system_prompt} Today's date is {TODAY}. You MUST call your tool before answering.",
    )

    def node(state: MSAState) -> dict:
        result = agent.invoke({"messages": [HumanMessage(content=build_input(state))]})
        return {output_key: result["messages"][-1].content}

    return node


def image_data_url(path: str) -> str:
    """Encode a local image as a base64 data URL, converting formats the model can't read to PNG."""
    try:
        with Image.open(path) as image:
            if image.format in MODEL_FORMATS:
                data, mime = Path(path).read_bytes(), image.get_format_mimetype()
            else:
                buffer = BytesIO()
                image.convert("RGBA").save(buffer, format="PNG")
                data, mime = buffer.getvalue(), "image/png"
    except UnidentifiedImageError as error:
        raise ValueError(f"Not an image file: {path}") from error
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def build_vision_node(llm: Runnable) -> Callable[[MSAState], dict]:
    """Build a node that sends the image straight to a multimodal model."""

    def vision(state: MSAState) -> dict:
        response = llm.invoke([
            SystemMessage(content=VISION_AGENT_SYSTEM_PROMPT.format(today=TODAY)),
            HumanMessage(content=[
                {"type": "text", "text": f"User request: {state['user_request']}"},
                {"type": "image_url", "image_url": {"url": image_data_url(state["image_path"])}},
            ]),
        ])
        return {"vision_agent": response.content}

    return vision


def conversation(state: MSAState) -> list[str]:
    """Return the earlier chat turns as a text block, or nothing if there are none."""
    turns = [f"{turn['role']}: {turn['content']}" for turn in state.get("history", [])]
    return ["Conversation so far:\n" + "\n".join(turns)] if turns else []


def gathered(state: MSAState) -> str:
    """Join the chat history, the request and every agent output collected so far."""
    parts = conversation(state) + [f"Request: {state['user_request']}"]
    parts += [f"{name}: {state[name]}" for name in AGENTS if state.get(name)]
    return "\n\n".join(parts)


def writer_node(state: MSAState) -> dict:
    """Write the final report from the request and gathered material."""
    response = get_llm("report").invoke([
        SystemMessage(content=REPORT_WRITER_SYSTEM_PROMPT.format(today=TODAY)),
        HumanMessage(content=gathered(state)),
    ])
    return {"report": response.content}


class Route(BaseModel):
    """The supervisor's choice of the next agent to run."""

    reasoning: str = Field(description="One sentence on why this is the next step.")
    next: Literal["vision_agent", "audio_agent", "research_agent", "writer"]


def status(state: MSAState) -> str:
    """Describe what the supervisor has to work with."""
    lines = conversation(state) + [
        f"User request: {state['user_request']}",
        f"Image provided: {'yes' if state.get('image_path') else 'no'}",
        f"Audio file provided: {'yes' if state.get('audio_path') else 'no'}",
    ]
    lines += [f"{name}: {'done' if state.get(name) else 'not yet'}" for name in AGENTS]
    return "\n".join(lines)


def can_run(state: MSAState, agent: str) -> bool:
    """An agent can run if it has not run yet and its input file was provided."""
    needed = REQUIRES.get(agent)
    return not state.get(agent) and (needed is None or bool(state.get(needed)))


def build_supervisor(router: Runnable) -> Callable[[MSAState], Command]:
    """Build the supervisor node around a router that returns a Route."""

    def supervisor(state: MSAState) -> Command[Literal["vision_agent", "audio_agent", "research_agent", "writer"]]:
        steps = state.get("steps", 0)
        if steps >= MAX_STEPS:
            return Command(goto="writer")

        decision = router.invoke([
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT.format(today=TODAY)),
            HumanMessage(content=status(state)),
        ])
        nxt = decision.next
        if nxt != "writer" and not can_run(state, nxt):
            nxt = "writer"

        logger.info("supervisor -> %s: %s", nxt, decision.reasoning)
        return Command(goto=nxt, update={"steps": steps + 1})

    return supervisor
