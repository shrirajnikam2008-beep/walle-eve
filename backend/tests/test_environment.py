"""Comprehensive unit tests for SearchAndRescueEnvironment, physics, and hidden-state isolation."""
from __future__ import annotations

import json
from app.environment.mission_config import PRESET_CONFIGS, generate_seeded_mission
from app.environment.simulator import SearchAndRescueEnvironment
from app.schemas.mission import ActionType, FullMissionConfig, MissionStatus


def test_environment_initialization() -> None:
    cfg = PRESET_CONFIGS["normal"]
    env = SearchAndRescueEnvironment(cfg)
    obs = env.observe()
    assert env.walle_position == (0, 0)
    assert env.battery == cfg.battery_capacity
    assert env.step_count == 0
    assert env.status == MissionStatus.ACTIVE
    assert obs.position == (0, 0)
    assert obs.target_verified_here is False


def test_valid_movement_and_energy_cost() -> None:
    cfg = PRESET_CONFIGS["normal"]
    env = SearchAndRescueEnvironment(cfg)
    obs = env.step(ActionType.MOVE, target_cell=(1, 0))
    assert obs.action_succeeded is True
    assert env.walle_position == (1, 0)
    assert env.battery == cfg.battery_capacity - cfg.move_cost
    assert obs.energy_delta == -cfg.move_cost
    assert (1, 0) in env.explored_cells


def test_map_boundaries_and_obstacle_blocking() -> None:
    cfg = FullMissionConfig(
        grid_width=6,
        grid_height=6,
        start_position=(0, 0),
        target_position=(5, 5),
        obstacles=[(1, 0)],
        battery_capacity=50,
    )
    env = SearchAndRescueEnvironment(cfg)

    # Out of bounds move (-1, 0)
    obs_oob = env.step(ActionType.MOVE, target_cell=(-1, 0))
    assert obs_oob.action_succeeded is False
    assert env.walle_position == (0, 0)

    # Blocked by obstacle at (1, 0)
    obs_obs = env.step(ActionType.MOVE, target_cell=(1, 0))
    assert obs_obs.action_succeeded is False
    assert env.walle_position == (0, 0)
    assert (1, 0) in env.known_obstacles


def test_charging_and_insufficient_battery() -> None:
    cfg = FullMissionConfig(
        grid_width=6,
        grid_height=6,
        start_position=(0, 0),
        target_position=(5, 5),
        charging_stations=[(0, 0)],
        battery_capacity=20,
        move_cost=6,
        scan_cost=5,
        charge_rate=10,
    )
    env = SearchAndRescueEnvironment(cfg)
    # Move away and back to consume energy
    env.step(ActionType.MOVE, target_cell=(0, 1))
    env.step(ActionType.MOVE, target_cell=(0, 0))
    assert env.battery == 8

    # Recharge at (0, 0)
    obs_charge = env.step(ActionType.RECHARGE)
    assert obs_charge.action_succeeded is True
    assert env.battery == 18

    # Drain battery away from charger
    env.step(ActionType.MOVE, target_cell=(0, 1))  # 12 left
    env.step(ActionType.MOVE, target_cell=(0, 2))  # 6 left
    env.step(ActionType.MOVE, target_cell=(0, 3))  # 0 left -> depleted away from charger
    assert env.status == MissionStatus.FAILED_BATTERY


def test_mission_timeout_and_discovery() -> None:
    # Timeout test
    cfg_timeout = FullMissionConfig(
        grid_width=6,
        grid_height=6,
        start_position=(0, 0),
        target_position=(5, 5),
        battery_capacity=100,
        max_actions=10,
    )
    env_t = SearchAndRescueEnvironment(cfg_timeout)
    for _ in range(10):
        env_t.step(ActionType.SCAN)
    assert env_t.status == MissionStatus.FAILED_TIMEOUT

    # Discovery test
    cfg_win = FullMissionConfig(
        grid_width=6,
        grid_height=6,
        start_position=(0, 0),
        target_position=(0, 1),
        battery_capacity=100,
        max_actions=20,
    )
    env_w = SearchAndRescueEnvironment(cfg_win)
    obs_win = env_w.step(ActionType.MOVE, target_cell=(0, 1))
    assert obs_win.target_verified_here is True
    assert env_w.status == MissionStatus.SUCCESS


def test_hidden_state_isolation_and_seed_reproducibility() -> None:
    cfg = FullMissionConfig(
        grid_width=12,
        grid_height=12,
        seed=777,
        start_position=(0, 0),
        target_position=(8, 9),
        decoy_positions=[(3, 7)],
    )
    env1 = SearchAndRescueEnvironment(cfg)
    env2 = SearchAndRescueEnvironment(cfg)

    pub_cfg = env1.get_public_config()
    pub_dump = pub_cfg.model_dump()
    assert "target_position" not in pub_dump
    assert "decoy_positions" not in pub_dump
    assert "(8, 9)" not in json.dumps(pub_dump)
    assert "[8, 9]" not in json.dumps(pub_dump)

    # Active evaluation must be None while mission is active
    assert env1.build_post_mission_evaluation((0, 0), 0.1, "active") is None

    # Seed reproducibility check
    obs1 = env1.step(ActionType.SCAN)
    obs2 = env2.step(ActionType.SCAN)
    assert obs1.signal_strength == obs2.signal_strength

    gen1 = generate_seeded_mission(seed=123, noise_std=0.1)
    gen2 = generate_seeded_mission(seed=123, noise_std=0.1)
    assert gen1.target_position == gen2.target_position
    assert gen1.obstacles == gen2.obstacles
