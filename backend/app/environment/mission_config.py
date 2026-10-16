"""Mission configuration presets, YAML/JSON loader, and deterministic procedural world generator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import yaml

from app.schemas.mission import Coordinate, EveOperationalStatus, FullMissionConfig


PRESET_CONFIGS: Dict[str, FullMissionConfig] = {
    "normal": FullMissionConfig(
        mission_name="Sector 7G — Standard Search",
        preset_id="normal",
        description="Balanced 12x12 sector with moderate noise, scattered debris obstacles, and 1 decoy beacon.",
        grid_width=12,
        grid_height=12,
        seed=42,
        start_position=(0, 0),
        target_position=(9, 10),
        true_eve_status=EveOperationalStatus.CRITICAL_LOW_POWER,
        decoy_positions=[(2, 9)],
        obstacles=[
            (2, 2), (2, 3), (3, 3), (5, 1), (5, 2),
            (4, 6), (5, 6), (6, 4), (8, 3), (8, 4),
            (3, 8), (3, 9), (7, 8), (8, 8), (9, 6), (10, 6)
        ],
        charging_stations=[(0, 0), (6, 6)],
        battery_capacity=100,
        move_cost=2,
        scan_cost=3,
        wait_cost=1,
        charge_rate=40,
        sensor_noise_std=0.07,
        signal_decay_rate=0.14,
        decoy_count=1,
        decoy_strength=0.45,
        max_actions=95,
    ),
    "misleading_decoy": FullMissionConfig(
        mission_name="Echo Canyon — Ghost Signals",
        preset_id="misleading_decoy",
        description="High-interference zone with 2 decoy transponders designed to bait naive greedy trackers.",
        grid_width=12,
        grid_height=12,
        seed=108,
        start_position=(0, 0),
        target_position=(10, 9),
        true_eve_status=EveOperationalStatus.DORMANT_INTERFERENCE,
        decoy_positions=[(4, 3), (2, 10)],
        obstacles=[
            (1, 4), (2, 4), (3, 4), (6, 2), (6, 3),
            (5, 7), (6, 7), (7, 7), (9, 4), (9, 5),
            (4, 10), (8, 10), (8, 9)
        ],
        charging_stations=[(0, 0), (5, 5), (10, 2)],
        battery_capacity=110,
        move_cost=2,
        scan_cost=3,
        wait_cost=1,
        charge_rate=45,
        sensor_noise_std=0.08,
        signal_decay_rate=0.14,
        decoy_count=2,
        decoy_strength=0.52,
        max_actions=110,
    ),
    "low_battery": FullMissionConfig(
        mission_name="Brownout — Solar Minimum",
        preset_id="low_battery",
        description="Restricted battery reserve (56 units). Requires disciplined solar charging hops to avoid stranding.",
        grid_width=12,
        grid_height=12,
        seed=77,
        start_position=(0, 0),
        target_position=(8, 9),
        true_eve_status=EveOperationalStatus.CRITICAL_LOW_POWER,
        decoy_positions=[(2, 8)],
        obstacles=[
            (3, 1), (3, 2), (2, 6), (3, 6), (4, 6),
            (7, 4), (7, 5), (6, 9), (7, 9)
        ],
        charging_stations=[(0, 0), (5, 5), (8, 7)],
        battery_capacity=56,
        move_cost=2,
        scan_cost=3,
        wait_cost=1,
        charge_rate=42,
        sensor_noise_std=0.06,
        signal_decay_rate=0.14,
        decoy_count=1,
        decoy_strength=0.40,
        max_actions=120,
    ),
    "unreachable": FullMissionConfig(
        mission_name="Quarantine Vault — Sealed Sector",
        preset_id="unreachable",
        description="EVE's signal originates behind an impenetrable containment wall. Tests safe failure detection.",
        grid_width=12,
        grid_height=12,
        seed=999,
        start_position=(0, 0),
        target_position=(10, 10),
        true_eve_status=EveOperationalStatus.DORMANT_INTERFERENCE,
        decoy_positions=[],
        obstacles=[
            # Complete wall sealing off the bottom-right corner (x>=8, y>=8)
            (8, 8), (8, 9), (8, 10), (8, 11),
            (9, 8), (10, 8), (11, 8),
            (3, 3), (3, 4), (5, 6)
        ],
        charging_stations=[(0, 0), (5, 5)],
        battery_capacity=90,
        move_cost=2,
        scan_cost=3,
        wait_cost=1,
        charge_rate=40,
        sensor_noise_std=0.07,
        signal_decay_rate=0.14,
        decoy_count=0,
        decoy_strength=0.0,
        max_actions=85,
    ),
    "high_noise": FullMissionConfig(
        mission_name="Ion Storm — Maximum Entropy",
        preset_id="high_noise",
        description="Severe electromagnetic interference (sigma=0.14). Single scans are untrustworthy; Bayesian aggregation is essential.",
        grid_width=12,
        grid_height=12,
        seed=2026,
        start_position=(0, 0),
        target_position=(8, 10),
        true_eve_status=EveOperationalStatus.DORMANT_INTERFERENCE,
        decoy_positions=[(9, 2)],
        obstacles=[
            (2, 3), (3, 3), (4, 3), (6, 5), (6, 6),
            (5, 9), (6, 9), (9, 6), (9, 7)
        ],
        charging_stations=[(0, 0), (5, 4), (8, 8)],
        battery_capacity=110,
        move_cost=2,
        scan_cost=3,
        wait_cost=1,
        charge_rate=45,
        sensor_noise_std=0.14,
        signal_decay_rate=0.14,
        decoy_count=1,
        decoy_strength=0.48,
        max_actions=115,
    ),
}


def generate_seeded_mission(
    seed: int,
    noise_std: float = 0.08,
    grid_width: int = 12,
    grid_height: int = 12,
    decoy_count: int = 1,
) -> FullMissionConfig:
    """Generate a reproducible, guaranteed-reachable mission for a given random seed."""
    rng = np.random.default_rng(seed)
    start: Coordinate = (0, 0)
    chargers: List[Coordinate] = [(0, 0), (grid_width // 2, grid_height // 2)]

    all_cells = [
        (x, y)
        for x in range(grid_width)
        for y in range(grid_height)
        if (x, y) not in chargers
    ]

    # Pick target at moderate-to-far distance from start so search is non-trivial
    far_cells = [c for c in all_cells if (c[0] + c[1]) >= (grid_width + grid_height) // 2]
    target_idx = int(rng.integers(0, len(far_cells)))
    target: Coordinate = far_cells[target_idx]

    # Generate obstacles (~11% of grid) while keeping target and chargers reachable
    candidate_obstacles = [c for c in all_cells if c != target]
    rng.shuffle(candidate_obstacles)
    obstacles: List[Coordinate] = []
    target_obstacle_count = max(8, (grid_width * grid_height) // 10)

    for cell in candidate_obstacles:
        if len(obstacles) >= target_obstacle_count:
            break
        trial_obstacles = set(obstacles + [cell])
        if _is_reachable(start, target, grid_width, grid_height, trial_obstacles) and all(
            _is_reachable(start, ch, grid_width, grid_height, trial_obstacles) for ch in chargers
        ):
            obstacles.append(cell)

    # Pick decoys on valid non-obstacle, non-target cells
    valid_decoy_cells = [
        c for c in all_cells
        if c != target and c not in obstacles and abs(c[0] - target[0]) + abs(c[1] - target[1]) >= 4
    ]
    decoys: List[Coordinate] = []
    if valid_decoy_cells and decoy_count > 0:
        chosen_indices = rng.choice(
            len(valid_decoy_cells),
            size=min(decoy_count, len(valid_decoy_cells)),
            replace=False,
        )
        decoys = [valid_decoy_cells[int(i)] for i in chosen_indices]

    statuses = [
        EveOperationalStatus.OPERATIONAL_SIGNAL,
        EveOperationalStatus.CRITICAL_LOW_POWER,
        EveOperationalStatus.DORMANT_INTERFERENCE,
    ]
    chosen_status = statuses[seed % len(statuses)]

    return FullMissionConfig(
        mission_name=f"Benchmark Trial #{seed}",
        preset_id="seeded",
        description=f"Procedural reproducible mission (seed={seed}, noise={noise_std})",
        grid_width=grid_width,
        grid_height=grid_height,
        seed=seed,
        start_position=start,
        target_position=target,
        true_eve_status=chosen_status,
        decoy_positions=decoys,
        obstacles=obstacles,
        charging_stations=chargers,
        battery_capacity=105,
        move_cost=2,
        scan_cost=3,
        wait_cost=1,
        charge_rate=40,
        sensor_noise_std=noise_std,
        signal_decay_rate=0.14,
        decoy_count=len(decoys),
        decoy_strength=0.45,
        max_actions=95,
    )


def _is_reachable(
    start: Coordinate,
    goal: Coordinate,
    width: int,
    height: int,
    obstacles: set[Coordinate],
) -> bool:
    """Simple BFS connectivity check used during procedural map generation."""
    if start == goal:
        return True
    visited = {start}
    queue = [start]
    while queue:
        cx, cy = queue.pop(0)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            nxt = (nx, ny)
            if 0 <= nx < width and 0 <= ny < height and nxt not in obstacles and nxt not in visited:
                if nxt == goal:
                    return True
                visited.add(nxt)
                queue.append(nxt)
    return False


def load_mission_from_file(path: str | Path) -> FullMissionConfig:
    """Load and validate a FullMissionConfig from a YAML (`.yaml`/`.yml`) or JSON (`.json`) file."""
    file_path = Path(path)
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.suffix.lower() in (".yaml", ".yml"):
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    return FullMissionConfig(**raw)


def load_mission_from_json(path: str | Path) -> FullMissionConfig:
    """Alias for loading a FullMissionConfig from JSON or YAML."""
    return load_mission_from_file(path)
