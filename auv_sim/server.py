"""Local HTTP interface. Start with ``python -m auv_sim``."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .experiment import Experiment


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ResetRequest(RequestModel):
    seed: int = Field(default=7, ge=0, le=2**31 - 1)
    scenario: Literal["open", "reef", "channel"] = "reef"
    current_strength: float = Field(default=0.08, ge=0, le=0.4)


class ControlRequest(RequestModel):
    command: Literal["play", "pause", "step"]
    controller: Literal["baseline", "td3", "random"] = "baseline"
    speed: float = Field(default=1, gt=0, le=32)


class TrainRequest(RequestModel):
    steps: int = Field(default=10_000, ge=1, le=1_000_000)
    exploration_noise: float = Field(default=0.15, ge=0, le=1)


class EvaluationRequest(RequestModel):
    episodes: int = Field(default=10, ge=1, le=100)
    controller: Literal["baseline", "td3", "random"] = "td3"


class LoadRequest(RequestModel):
    name: str = Field(min_length=1, max_length=100)


def create_app(data_dir: Path | str = "runs", experiment: Experiment | None = None) -> FastAPI:
    lab = experiment or Experiment(data_dir)

    @asynccontextmanager
    async def lifespan(app):
        yield
        lab.close()

    app = FastAPI(title="AUV Practice", version="0.1.0", lifespan=lifespan)
    app.state.experiment = lab
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    def execute(operation, **kwargs):
        try:
            return operation(**kwargs)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "algorithm": "TD3", "version": "0.1.0"}

    @app.get("/api/state")
    def state():
        return lab.state()

    @app.post("/api/reset")
    def reset(request: ResetRequest):
        return execute(lab.reset, **request.model_dump())

    @app.post("/api/control")
    def control(request: ControlRequest):
        return execute(lab.control, **request.model_dump())

    @app.post("/api/train")
    def train(request: TrainRequest):
        return execute(lab.start_training, **request.model_dump())

    @app.post("/api/evaluate")
    def evaluate(request: EvaluationRequest):
        return execute(lab.start_evaluation, **request.model_dump())

    @app.post("/api/save")
    def save():
        return execute(lab.save)

    @app.post("/api/load")
    def load(request: LoadRequest):
        return execute(lab.load, **request.model_dump())

    @app.get("/api/results")
    def results():
        snapshot = lab.state()
        return {key: snapshot[key] for key in ["config", "training", "history", "evaluation"]}

    return app
