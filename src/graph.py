from langgraph.graph import END, START, StateGraph

from src.llm import get_llm
from src.nodes import AGENTS, Route, build_supervisor, build_vision_node, gathered, specialist_agent, writer_node
from src.prompts import AUDIO_AGENT_SYSTEM_PROMPT, RESEARCH_AGENT_SYSTEM_PROMPT
from src.state import MSAState
from src.tools import transcribe_audio, web_search


def build_graph() -> StateGraph:
    """Wire the supervisor, the specialist agents and the writer together."""
    graph = StateGraph(MSAState)
    graph.add_node("supervisor", build_supervisor(get_llm("supervisor").with_structured_output(Route)))
    graph.add_node("vision_agent", build_vision_node(get_llm("vision")))
    graph.add_node("audio_agent", specialist_agent(
        "audio",
        AUDIO_AGENT_SYSTEM_PROMPT,
        tools=[transcribe_audio],
        build_input=lambda s: s["audio_path"],
        output_key="audio_agent",
    ))
    graph.add_node("research_agent", specialist_agent(
        "research",
        RESEARCH_AGENT_SYSTEM_PROMPT,
        tools=[web_search],
        build_input=gathered,
        output_key="research_agent",
    ))
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "supervisor")
    for agent in AGENTS:
        graph.add_edge(agent, "supervisor")
    graph.add_edge("writer", END)
    return graph

