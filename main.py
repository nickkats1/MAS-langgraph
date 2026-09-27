import json
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.graph import build_graph

ROOT = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = build_graph().compile()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def get_graph(request: Request):
    """Return the compiled multi-agent graph built at startup."""
    return request.app.state.graph


def save_upload(upload: UploadFile, folder: str) -> str:
    """Write an uploaded file into folder and return its path."""
    path = Path(folder) / Path(upload.filename).name
    with path.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    return str(path)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "templates" / "index.html")


@app.post("/ask")
def ask(
    user_request: str = Form(),
    history: str = Form("[]"),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    graph=Depends(get_graph),
) -> dict:
    """Run the request through the agents and return the writer's report."""
    state = {"user_request": user_request}
    if turns := json.loads(history):
        state["history"] = turns
    with tempfile.TemporaryDirectory() as folder:
        if image:
            state["image_path"] = save_upload(image, folder)
        if audio:
            state["audio_path"] = save_upload(audio, folder)
        result = graph.invoke(state)
    return {"report": result["report"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8080)
