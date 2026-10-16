"""Unit tests for A* pathfinding, unreachable targets, resource management, and agent autonomy."""
from __future__ import annotations

import pytest

from app.agent.agent import AutonomousAgent
from app.agent.pathfinding import astar_shortest_path
from app.environment.mission_config import PRESET_CONFIGS
from app.environment.simulator import SearchAndRescueEnvironment
from app.schemas.mission import ActionType, FullMissionConfig, MissionStatus


def test_astar_pathfinding_and_unreachable_routes() -> None:
    obstacles = {(1, 0), (1, 1), (1, 2)}
    path = astar_shortest_path((0, 0), (2, 0), width=5, height=5, obstacles=obstacles)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 0)
    for cell in path:
        assert cell not in obstacles

    # Completely walled-off goal
    sealed = {(1, 0), (0, 1), (1, 1)}
    unreachable_path = astar_shortest_path((0, 0), (4, 4), width=5, height=5, obstacles=sealed)
    assert unreachable_path is None


def test_agent_rejects_leaked_target_config() -> None:
    full_cfg = PRESET_CONFIGS["normal"]
    with pytest.raises(ValueError, match="SECURITY VIOLATION"):
        AutonomousAgent(full_cfg)  # type: ignore[arg-type]


def test_agent_completes_normal_mission_and_unreachable_mission() -> None:
    # Normal mission should succeed
    env_norm = SearchAndRescueEnvironment(PRESET_CONFIGS["normal"])
    agent_norm = AutonomousAgent(env_norm.get_public_config())

    steps = 0
    while env_norm.status == MissionStatus.ACTIVE and steps < 100:
        act, target_cell, expl = agent_norm.decide()
        assert expl.technical_rationale != ""
        obs = env_norm.step(act, target_cell=target_cell)
        agent_norm.ingest_observation(obs)
        steps += 1

    assert env_norm.status == MissionStatus.SUCCESS
    assert env_norm.target_verified is True

    # Unreachable mission should cleanly abort via RETURN without infinite looping
    env_unr = SearchAndRescueEnvironment(PRESET_CONFIGS["unreachable"])
    agent_unr = AutonomousAgent(env_unr.get_public_config())
    steps_unr = 0
    while env_unr.status == MissionStatus.ACTIVE and steps_unr < 100:
        act, target_cell, _ = agent_unr.decide()
        obs = env_unr.step(act, target_cell=target_cell)
        agent_unr.ingest_observation(obs)
        steps_unr += 1

    assert env_unr.status in (MissionStatus.FAILED_UNREACHABLE, MissionStatus.SAFE_RETURNED)


def test_low_battery_mission_uses_charger() -> None:
    env_low = SearchAndRescueEnvironment(PRESET_CONFIGS["low_battery"])
    agent_low = AutonomousAgent(env_low.get_public_config())

    while env_low.status == MissionStatus.ACTIVE:
        act, target_cell, _ = agent_low.decide()
        obs = env_low.step(act, target_cell=target_cell)
        agent_low.ingest_observation(obs)

    assert env_low.total_recharges >= 1
    assert env_low.status == MissionStatus.SUCCESS
