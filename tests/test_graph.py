import pytest

from src.graph import build_graph


@pytest.fixture
def multi_agent():
    return build_graph().compile()


@pytest.mark.integration
class TestGraph:
    def test_has_expected_nodes(self, multi_agent):
        assert {"supervisor", "vision_agent", "audio_agent", "research_agent", "writer"} <= set(multi_agent.nodes)

    def test_answers_current_events(self, multi_agent):
        result = multi_agent.invoke({"user_request": "What is going on in the news with Nvidia?"})
        assert result["research_agent"]
        assert result["report"]
