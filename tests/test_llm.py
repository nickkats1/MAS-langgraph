import pytest
from langchain_core.language_models import BaseChatModel

from src.llm import TEMPERATURES, get_llm


class TestTemperatures:
    def test_roles(self):
        assert set(TEMPERATURES) == {"supervisor", "vision", "audio", "code", "research", "report"}


class TestGetLlm:
    def test_unknown_role_raises(self):
        with pytest.raises(KeyError):
            get_llm("nope")

    @pytest.mark.integration
    @pytest.mark.parametrize("role", TEMPERATURES)
    def test_builds_chat_model(self, role):
        assert isinstance(get_llm(role), BaseChatModel)
