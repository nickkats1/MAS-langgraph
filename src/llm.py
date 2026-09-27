import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

load_dotenv()

TEMPERATURES = {
    "supervisor": 0.0,
    "vision": 0.0,
    "audio": 0.0,
    "code": 0.0,
    "research": 0.0,
    "report": 0.3,
}


def get_llm(role: str) -> BaseChatModel:
    """Build the chat model set by MODEL_NAME with the role's temperature."""
    temperature = TEMPERATURES[role]
    model_name = os.environ.get("MODEL_NAME", "openai:gpt-4o")
    return init_chat_model(model_name, temperature=temperature)
