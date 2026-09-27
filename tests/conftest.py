from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.tools as tools
from main import app, get_graph
from src.nodes import Route


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-integration", action="store_true", help="run tests that call real services")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: calls real external services")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="needs --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


class FakeSearch:
    """Stands in for DuckDuckGoSearchResults and records how it was used."""

    def __init__(self) -> None:
        self.num_results = None
        self.query = None

    def __call__(self, num_results: int) -> "FakeSearch":
        self.num_results = num_results
        return self

    def invoke(self, query: str) -> str:
        self.query = query
        return f"results for {query}"


class FakeOpenAI:
    """Stands in for the OpenAI client's audio transcription endpoint."""

    def __init__(self) -> None:
        self.model = None
        self.content = None
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self.create))

    def __call__(self) -> "FakeOpenAI":
        return self

    def create(self, model: str, file) -> SimpleNamespace:
        self.model = model
        self.content = file.read()
        return SimpleNamespace(text="hello world")


class FakeLLM:
    """Stands in for a chat model and records the messages it was given."""

    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages: list) -> SimpleNamespace:
        self.messages = messages
        return SimpleNamespace(content="a red square")


@pytest.fixture
def fake_search(monkeypatch: pytest.MonkeyPatch) -> FakeSearch:
    fake = FakeSearch()
    monkeypatch.setattr(tools, "DuckDuckGoSearchResults", fake)
    return fake


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    fake = FakeOpenAI()
    monkeypatch.setattr(tools, "OpenAI", fake)
    return fake


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"fake audio bytes")
    return path


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    path = tmp_path / "photo.png"
    Image.new("L", (8, 8), color=128).save(path)
    return path


@pytest.fixture
def image_factory(tmp_path: Path):
    def make(name: str, image_format: str, mode: str = "RGB") -> Path:
        path = tmp_path / name
        Image.new(mode, (16, 16)).save(path, format=image_format)
        return path

    return make


class FakeRouter:
    """Stands in for the structured supervisor LLM and returns a fixed Route."""

    def __init__(self) -> None:
        self.route = Route(reasoning="test", next="writer")
        self.messages = None

    def invoke(self, messages: list) -> Route:
        self.messages = messages
        return self.route


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_router() -> FakeRouter:
    return FakeRouter()


class FakeGraph:
    """Stands in for the compiled graph and records the state it was given."""

    def __init__(self) -> None:
        self.state = None
        self.files = {}

    def invoke(self, state: dict) -> dict:
        self.state = state
        self.files = {key: Path(state[key]).read_bytes() for key in ("image_path", "audio_path") if key in state}
        return {"report": "fake report"}


@pytest.fixture
def fake_graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture
def client(fake_graph: FakeGraph):
    app.dependency_overrides[get_graph] = lambda: fake_graph
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def history_state() -> dict:
    return {
        "user_request": "what about AMD?",
        "history": [
            {"role": "user", "content": "news on Nvidia"},
            {"role": "assistant", "content": "Nvidia stock rose."},
        ],
    }
