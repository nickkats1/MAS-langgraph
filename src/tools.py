from pathlib import Path
import os
from openai import OpenAI

from langchain_community.tools import tool, DuckDuckGoSearchResults
from langchain_core.tools import tool

# --- Web Search Tool ---
@tool("web-search")
def web_search(query: str) -> str:
    """search web for relevant information given the users request.
    
    Args:
        query: the user input.
        
    Returns:
        llm-generated output after using search tool
    """
    ddg = DuckDuckGoSearchResults(num_results=5)
    return ddg.invoke(query)


# --- Tool to Transcribe Audio ---

@tool("transcribe-audio")
def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file (mp3, wav, m4a) into text."""
    client = OpenAI()
    with Path(file_path).open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
    return transcript.text


tools = [web_search, transcribe_audio]
tool_name = [tool.name for tool in tools]
