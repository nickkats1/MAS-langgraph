from typing import get_type_hints

import pytest

from src.state import MSAState


@pytest.fixture
def msa_state() -> MSAState:
    return MSAState(user_request="describe this image", image_path="photo.png")


class TestMSAState:
    def test_has_expected_keys(self):
        assert set(MSAState.__annotations__) == {
            "user_request", "history", "image_path", "audio_path", "vision_agent", "audio_agent",
            "research_agent", "code_agent", "report", "steps",
        }

    def test_only_request_is_required(self):
        assert MSAState.__required_keys__ == frozenset({"user_request"})

    def test_steps_is_int(self):
        assert get_type_hints(MSAState)["steps"] is int

    def test_instance_is_a_dict(self, msa_state):
        assert isinstance(msa_state, dict)
        assert msa_state["user_request"] == "describe this image"
