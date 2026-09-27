from typing import NotRequired, TypedDict


class MSAState(TypedDict):
    """Shared state passed between the supervisor, the agents and the writer."""

    user_request: str
    history: NotRequired[list[dict]]
    image_path: NotRequired[str]
    audio_path: NotRequired[str]
    vision_agent: NotRequired[str]
    audio_agent: NotRequired[str]
    research_agent: NotRequired[str]
    code_agent: NotRequired[str]
    report: NotRequired[str]
    steps: NotRequired[int]
