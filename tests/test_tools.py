import os
import wave

import pytest

from src.tools import tool_name, tools, transcribe_audio, web_search


class TestToolRegistry:
    def test_tool_names(self):
        assert tool_name == ["web-search", "transcribe-audio"]

    @pytest.mark.parametrize(
        "tool, expected_args",
        [
            (web_search, {"query"}),
            (transcribe_audio, {"file_path"}),
        ],
    )
    def test_tool_args(self, tool, expected_args):
        assert set(tool.args) == expected_args
        assert tool in tools

    @pytest.mark.parametrize("tool", tools)
    def test_tool_has_description(self, tool):
        assert tool.description


class TestWebSearch:
    def test_returns_search_results(self, fake_search):
        assert web_search.invoke({"query": "langgraph"}) == "results for langgraph"

    def test_passes_query_and_limit(self, fake_search):
        web_search.invoke({"query": "langgraph"})
        assert fake_search.query == "langgraph"
        assert fake_search.num_results == 5


class TestTranscribeAudio:
    def test_returns_transcript_text(self, fake_openai, audio_file):
        assert transcribe_audio.invoke({"file_path": str(audio_file)}) == "hello world"

    def test_sends_file_to_whisper(self, fake_openai, audio_file):
        transcribe_audio.invoke({"file_path": str(audio_file)})
        assert fake_openai.model == "whisper-1"
        assert fake_openai.content == b"fake audio bytes"

    def test_missing_file_raises(self, fake_openai, tmp_path):
        with pytest.raises(FileNotFoundError):
            transcribe_audio.invoke({"file_path": str(tmp_path / "missing.mp3")})


@pytest.mark.integration
class TestIntegration:
    def test_real_web_search(self):
        assert web_search.invoke({"query": "python programming language"})

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
    def test_real_transcription(self, tmp_path):
        path = tmp_path / "silence.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 16000)
        assert isinstance(transcribe_audio.invoke({"file_path": str(path)}), str)

