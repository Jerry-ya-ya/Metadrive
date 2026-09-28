from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from webapp.service import (
    command_for_tool,
    job_manager,
    scan_models,
    tool_inventory,
    training_steps,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"


class PostTestRequest(BaseModel):
    enabled: bool = True
    test_name: str | None = None
    episodes: int = Field(default=5, ge=1)
    max_steps: int = Field(default=1000, ge=1)
    record_steps: int = Field(default=1000, ge=1)
    seed: int = Field(default=0, ge=0)
    fps: int = Field(default=30, ge=1)
    screen_size: int = Field(default=672, ge=1)


class TrainingRequest(BaseModel):
    mode: Literal["new", "continue"]
    model_path: str = Field(min_length=1)
    timesteps: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    post_test: PostTestRequest = Field(default_factory=PostTestRequest)


class ToolRunRequest(BaseModel):
    module: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="MetaDrive PPO Control Center", version="1.0.0")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


def model_data(model):
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "port": 4000}


@app.get("/api/tools")
def tools():
    return tool_inventory()


@app.get("/api/models")
def models():
    return scan_models()


@app.get("/api/jobs")
def jobs():
    return job_manager.list()


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    result = job_manager.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return result


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    result = job_manager.cancel(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return result


@app.post("/api/training", status_code=202)
def start_training(request: TrainingRequest):
    payload = model_data(request)
    try:
        steps, summary = training_steps(payload)
        return job_manager.start(f"training:{request.mode}", steps, summary)
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/tools/run", status_code=202)
def run_tool(request: ToolRunRequest):
    try:
        tool, command = command_for_tool(request.module, request.arguments)
        return job_manager.start(
            f"tool:{tool.module}",
            [(tool.module, command)],
        )
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
