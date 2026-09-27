import base64
from io import BytesIO

import pytest
from PIL import Image
from pydantic import ValidationError

from src.nodes import (
    MAX_STEPS,
    Route,
    build_supervisor,
    build_vision_node,
    can_run,
    gathered,
    image_data_url,
    status,
)


@pytest.fixture
def supervisor(fake_router):
    return build_supervisor(fake_router)


class TestRoute:
    def test_rejects_unknown_agent(self):
        with pytest.raises(ValidationError):
            Route(reasoning="x", next="code_agent")


class TestStatus:
    def test_reports_files_and_progress(self):
        text = status({"user_request": "hi", "image_path": "a.png", "research_agent": "found it"})
        assert "Image provided: yes" in text
        assert "Audio file provided: no" in text
        assert "research_agent: done" in text
        assert "vision_agent: not yet" in text

    def test_includes_history(self, history_state):
        text = status(history_state)
        assert "user: news on Nvidia" in text
        assert "assistant: Nvidia stock rose." in text


class TestGathered:
    def test_includes_only_finished_outputs(self):
        text = gathered({"user_request": "hi", "audio_agent": "la la la"})
        assert text == "Request: hi\n\naudio_agent: la la la"

    def test_puts_history_before_request(self, history_state):
        text = gathered(history_state)
        assert text == (
            "Conversation so far:\nuser: news on Nvidia\nassistant: Nvidia stock rose."
            "\n\nRequest: what about AMD?"
        )


class TestCanRun:
    @pytest.mark.parametrize(
        "state, agent, expected",
        [
            ({"user_request": "hi"}, "vision_agent", False),
            ({"user_request": "hi", "image_path": "a.png"}, "vision_agent", True),
            ({"user_request": "hi", "audio_path": "a.mp3", "audio_agent": "done"}, "audio_agent", False),
            ({"user_request": "hi"}, "research_agent", True),
            ({"user_request": "hi", "research_agent": "done"}, "research_agent", False),
        ],
    )
    def test_can_run(self, state, agent, expected):
        assert can_run(state, agent) is expected


class TestSupervisor:
    def test_routes_to_router_choice(self, supervisor, fake_router):
        fake_router.route = Route(reasoning="needs news", next="research_agent")
        command = supervisor({"user_request": "news on nvidia"})
        assert command.goto == "research_agent"
        assert command.update == {"steps": 1}

    def test_sends_status_to_router(self, supervisor, fake_router):
        supervisor({"user_request": "news on nvidia"})
        assert "User request: news on nvidia" in fake_router.messages[-1].content

    def test_guardrail_skips_agent_without_file(self, supervisor, fake_router):
        fake_router.route = Route(reasoning="look at it", next="vision_agent")
        assert supervisor({"user_request": "what is this?"}).goto == "writer"

    def test_guardrail_skips_agent_that_already_ran(self, supervisor, fake_router):
        fake_router.route = Route(reasoning="again", next="research_agent")
        assert supervisor({"user_request": "hi", "research_agent": "done"}).goto == "writer"

    def test_goes_to_writer_at_max_steps(self, supervisor, fake_router):
        command = supervisor({"user_request": "hi", "steps": MAX_STEPS})
        assert command.goto == "writer"
        assert fake_router.messages is None


class TestImageDataUrl:
    def test_encodes_png(self, image_file):
        assert image_data_url(str(image_file)).startswith("data:image/png;base64,")

    @pytest.mark.parametrize(
        ("name", "image_format", "mime"),
        [("photo.jpg", "JPEG", "image/jpeg"), ("anim.gif", "GIF", "image/gif"), ("photo.webp", "WEBP", "image/webp")],
    )
    def test_keeps_model_formats(self, image_factory, name, image_format, mime):
        url = image_data_url(str(image_factory(name, image_format)))
        assert url.startswith(f"data:{mime};base64,")

    @pytest.mark.parametrize(
        ("name", "image_format", "mode"),
        [
            ("photo.bmp", "BMP", "RGB"),
            ("scan.tiff", "TIFF", "RGB"),
            ("print.tiff", "TIFF", "CMYK"),
            ("icon.ico", "ICO", "RGB"),
            ("iphone.heic", "HEIF", "RGB"),
        ],
    )
    def test_converts_other_formats_to_png(self, image_factory, name, image_format, mode):
        url = image_data_url(str(image_factory(name, image_format, mode)))
        header, data = url.split(",", 1)
        assert header == "data:image/png;base64"
        assert Image.open(BytesIO(base64.b64decode(data))).format == "PNG"

    def test_detects_format_from_content(self, image_factory):
        url = image_data_url(str(image_factory("photo.bmp", "PNG")))
        assert url.startswith("data:image/png;base64,")

    def test_rejects_non_image(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hello")
        with pytest.raises(ValueError):
            image_data_url(str(path))


class TestVisionNode:
    def test_stores_description(self, fake_llm, image_file):
        node = build_vision_node(fake_llm)
        assert node({"user_request": "describe it", "image_path": str(image_file)}) == {"vision_agent": "a red square"}

    def test_sends_image_and_request(self, fake_llm, image_file):
        build_vision_node(fake_llm)({"user_request": "describe it", "image_path": str(image_file)})
        text, image = fake_llm.messages[-1].content
        assert "describe it" in text["text"]
        assert image["image_url"]["url"].startswith("data:image/png;base64,")
