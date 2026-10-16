"""FastAPI HTTP Endpoints for WALL-E: No Signal, No Chill."""
from __future__ import annotations

from typing import Dict, List
from fastapi import APIRouter, HTTPException, status

from app.api.session_store import MissionStore
from app.environment.mission_config import PRESET_CONFIGS
from app.experiments.benchmark import run_benchmark_experiment
from app.schemas.mission import (
    CreateMissionRequest,
    ExperimentRequest,
    ExperimentResponse,
    PostMissionEvaluation,
    PublicMissionConfig,
    PublicMissionState,
    RunMissionRequest,
    StepTraceEntry,
)

router = APIRouter(prefix="/api", tags=["mission-control"])
store = MissionStore(max_sessions=50)


@router.get("/health")
def health_check() -> Dict[str, str | int]:
    """Return backend service health and available preset count."""
    return {
        "status": "ok",
        "service": "walle-no-signal-no-chill-backend",
        "presets_available": len(PRESET_CONFIGS),
    }


@router.get("/presets", response_model=List[PublicMissionConfig])
def list_presets() -> List[PublicMissionConfig]:
    """Return all available predefined mission configurations (sanitized of hidden coordinates)."""
    return [cfg.to_public() for cfg in PRESET_CONFIGS.values()]


@router.post("/missions", response_model=PublicMissionState, status_code=status.HTTP_201_CREATED)
def create_mission(req: CreateMissionRequest) -> PublicMissionState:
    """Create and initialize a new mission session from preset, custom JSON, or YAML/JSON file."""
    if req.custom_config is None and req.config_file is None and req.preset_id not in PRESET_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown preset_id '{req.preset_id}'. Valid presets: {list(PRESET_CONFIGS.keys())}",
        )
    try:
        session = store.create(
            preset_id=req.preset_id,
            custom_config=req.custom_config,
            config_file=req.config_file,
            humor_enabled=req.humor_enabled,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return session.get_public_state()


@router.get("/missions/{mission_id}", response_model=PublicMissionState)
def get_mission(mission_id: str) -> PublicMissionState:
    """Return the current public mission state (never exposes hidden target coordinates while active)."""
    session = store.get(mission_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mission '{mission_id}' not found in active session store.",
        )
    return session.get_public_state()


@router.post("/missions/{mission_id}/step", response_model=PublicMissionState)
def step_mission(mission_id: str, humor_enabled: bool | None = None) -> PublicMissionState:
    """Execute one autonomous agent step and return the updated public mission state."""
    session = store.get(mission_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mission '{mission_id}' not found.",
        )
    if humor_enabled is not None:
        session.set_humor(humor_enabled)
    return session.step_once()


@router.post("/missions/{mission_id}/run", response_model=PublicMissionState)
def run_mission(mission_id: str, req: RunMissionRequest | None = None) -> PublicMissionState:
    """Run the mission autonomously until termination or `max_steps` limit."""
    session = store.get(mission_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mission '{mission_id}' not found.",
        )
    max_steps = req.max_steps if req is not None else 120
    if req is not None:
        session.set_humor(req.humor_enabled)
    return session.run_until_done(max_steps=max_steps)


@router.post("/missions/{mission_id}/pause", response_model=PublicMissionState)
def pause_mission(mission_id: str) -> PublicMissionState:
    """Pause an active mission."""
    session = store.get(mission_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mission '{mission_id}' not found.",
        )
    return session.pause()


@router.post("/missions/{mission_id}/reset", response_model=PublicMissionState)
def reset_mission(mission_id: str, humor_enabled: bool | None = None) -> PublicMissionState:
    """Reset a mission safely back to initial step 0."""
    session = store.get(mission_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mission '{mission_id}' not found.",
        )
    return session.reset(humor_enabled=humor_enabled)


@router.get("/missions/{mission_id}/trace", response_model=List[StepTraceEntry])
def get_mission_trace(mission_id: str) -> List[StepTraceEntry]:
    """Return the full chronological action, observation, belief, and commentary trace."""
    session = store.get(mission_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mission '{mission_id}' not found.",
        )
    return session.trace


@router.get("/missions/{mission_id}/report", response_model=PostMissionEvaluation)
def get_mission_report(mission_id: str) -> PostMissionEvaluation:
    """Return the final post-mission evaluation report (available once mission has terminated)."""
    session = store.get(mission_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mission '{mission_id}' not found.",
        )
    state = session.get_public_state()
    if state.post_mission_evaluation is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Mission is still active; final report is generated automatically upon completion, timeout, or abort.",
        )
    return state.post_mission_evaluation


@router.post("/experiments/run", response_model=ExperimentResponse)
def run_experiments(req: ExperimentRequest) -> ExperimentResponse:
    """Execute a reproducible multi-seed benchmark comparing the Bayesian agent against baselines."""
    return run_benchmark_experiment(req)
