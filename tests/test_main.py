import json
from pathlib import Path


class TestPages:
    def test_index_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="messages"' in response.text
        assert 'id="new-chat"' in response.text

    def test_serves_app_js(self, client):
        response = client.get("/static/app.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]

    def test_serves_stylesheet(self, client):
        response = client.get("/static/style.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]


class TestAsk:
    def test_text_only(self, client, fake_graph):
        response = client.post("/ask", data={"user_request": "news on Nvidia"})
        assert response.json() == {"report": "fake report"}
        assert fake_graph.state == {"user_request": "news on Nvidia"}

    def test_history_is_passed_to_graph(self, client, fake_graph, history_state):
        history = json.dumps(history_state["history"])
        client.post("/ask", data={"user_request": "what about AMD?", "history": history})
        assert fake_graph.state == history_state

    def test_with_uploads(self, client, fake_graph):
        files = {"image": ("photo.png", b"img"), "audio": ("clip.mp3", b"snd")}
        response = client.post("/ask", data={"user_request": "what is this?"}, files=files)
        assert response.status_code == 200
        assert Path(fake_graph.state["image_path"]).suffix == ".png"
        assert Path(fake_graph.state["audio_path"]).suffix == ".mp3"
        assert fake_graph.files == {"image_path": b"img", "audio_path": b"snd"}

    def test_temp_files_are_removed(self, client, fake_graph):
        client.post("/ask", data={"user_request": "hi"}, files={"image": ("photo.png", b"img")})
        assert not Path(fake_graph.state["image_path"]).exists()

    def test_missing_request_is_rejected(self, client):
        assert client.post("/ask", data={}).status_code == 422
