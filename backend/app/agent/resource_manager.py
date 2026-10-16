"""Energy-aware Resource Manager for WALL-E."""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from app.agent.pathfinding import astar_shortest_path, compute_distance_map
from app.schemas.mission import Coordinate, PublicMissionConfig


class ResourceManager:
    """Evaluates energy feasibility of candidate actions and manages safe-return routing to solar chargers."""

    def __init__(self, config: PublicMissionConfig) -> None:
        self.width: int = config.grid_width
        self.height: int = config.grid_height
        self.battery_capacity: int = config.battery_capacity
        self.move_cost: int = config.move_cost
        self.scan_cost: int = config.scan_cost
        self.charge_rate: int = config.charge_rate
        self.charging_stations: List[Coordinate] = [tuple(c) for c in config.charging_stations]
        # Keep a small safety buffer of 1 move_cost whenever chargers exist
        self.safety_buffer: int = config.move_cost

    def nearest_charger_path(
        self,
        from_pos: Coordinate,
        known_obstacles: Set[Coordinate],
    ) -> Tuple[Optional[Coordinate], Optional[List[Coordinate]], int]:
        """Find the closest reachable charging station and its A* path."""
        best_charger: Optional[Coordinate] = None
        best_path: Optional[List[Coordinate]] = None
        best_steps = 1_000_000

        for ch in self.charging_stations:
            if ch in known_obstacles:
                continue
            path = astar_shortest_path(from_pos, ch, self.width, self.height, known_obstacles)
            if path is not None:
                steps = len(path) - 1
                if steps < best_steps:
                    best_steps = steps
                    best_charger = ch
                    best_path = path

        return best_charger, best_path, best_steps

    def can_execute_and_reach_charger(
        self,
        next_pos: Coordinate,
        action_energy_cost: int,
        current_battery: int,
        known_obstacles: Set[Coordinate],
    ) -> Tuple[bool, int, int]:
        """Check whether taking an action ending at `next_pos` leaves enough energy to reach a charger.

        Returns:
            (is_feasible, energy_after_action, energy_needed_for_charger)
        """
        energy_after = current_battery - action_energy_cost
        if energy_after < 0:
            return False, energy_after, 999_999

        if not self.charging_stations:
            return True, energy_after, 0

        _, path, steps = self.nearest_charger_path(next_pos, known_obstacles)
        if path is None:
            # No reachable charger on map; allow action if energy_after >= 0
            return True, energy_after, 0

        energy_for_charger = steps * self.move_cost
        is_safe = energy_after >= (energy_for_charger + self.safety_buffer) or next_pos in self.charging_stations
        return is_safe, energy_after, energy_for_charger

    def should_recharge_now(
        self,
        current_pos: Coordinate,
        current_battery: int,
        known_obstacles: Set[Coordinate],
    ) -> Tuple[bool, Optional[List[Coordinate]], str]:
        """Determine if WALL-E should immediately recharge or head to a charging station."""
        if not self.charging_stations:
            return False, None, ""

        # If sitting on a charger and battery is below 75%, top up before venturing out
        if current_pos in self.charging_stations and current_battery <= int(0.72 * self.battery_capacity):
            return (
                True,
                [current_pos],
                f"Docked at solar charger {current_pos} with battery {current_battery}/{self.battery_capacity}; topping up.",
            )

        charger, path, steps = self.nearest_charger_path(current_pos, known_obstacles)
        if charger is None or path is None:
            return False, None, ""

        energy_to_return = steps * self.move_cost
        # Trigger return-to-charger when remaining battery is within 1 scan + 1 move of the minimum return energy
        reserve_threshold = energy_to_return + self.scan_cost + self.move_cost
        if current_battery <= reserve_threshold:
            return (
                True,
                path,
                f"Battery ({current_battery}) approaching minimum return reserve ({reserve_threshold} needed to reach charger at {charger} in {steps} steps).",
            )

        return False, None, ""
