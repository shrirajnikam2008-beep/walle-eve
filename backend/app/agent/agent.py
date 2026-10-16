"""Autonomous Search-and-Rescue Agent (WALL-E)."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from app.agent.beliefs import BeliefTracker
from app.agent.planner import ActionPlanner
from app.agent.resource_manager import ResourceManager
from app.schemas.mission import (
    ActionType,
    Coordinate,
    DecisionExplanation,
    PublicMissionConfig,
    SensorObservation,
)


class AutonomousAgent:
    """Maintains WALL-E's internal belief map, explored memory, resource estimates, and planner."""

    def __init__(self, public_config: PublicMissionConfig) -> None:
        if hasattr(public_config, "target_position") and getattr(public_config, "target_position") is not None:
            raise ValueError(
                "SECURITY VIOLATION: AutonomousAgent received a config containing hidden target_position!"
            )

        self.config: PublicMissionConfig = public_config
        self.beliefs = BeliefTracker(public_config)
        self.resources = ResourceManager(public_config)
        self.planner = ActionPlanner(public_config)

        self.position: Coordinate = tuple(public_config.start_position)
        self.battery: int = public_config.battery_capacity
        self.step_count: int = 0
        self.explored_cells: Set[Coordinate] = {self.position}
        self.known_obstacles: Set[Coordinate] = set(tuple(c) for c in public_config.obstacles)
        self.scans_per_cell: Dict[Coordinate, int] = {}
        self.visit_counts: Dict[Coordinate, int] = {self.position: 1}
        self.steps_since_last_scan: int = 0

        self.current_objective: str = "INITIALIZE_SEARCH"
        self.planned_route: List[Coordinate] = [self.position]
        self.latest_explanation: Optional[DecisionExplanation] = None
        self.previous_top_hypothesis: Optional[Coordinate] = None
        self.last_step_replanned: bool = False
        self.important_decisions: List[str] = []

        self.beliefs.mark_cell_visited(self.position, target_found=False)

    def ingest_observation(self, obs: SensorObservation) -> Dict[str, bool]:
        """Update internal state and Bayesian beliefs from an environment observation."""
        self.step_count = obs.step
        self.position = tuple(obs.position)
        self.battery = obs.battery_remaining
        self.explored_cells.add(self.position)

        events: Dict[str, bool] = {
            "high_signal": False,
            "weak_scan": False,
            "plan_changed": False,
            "repeated_visit": False,
            "new_obstacle_sensed": False,
            "repeated_scan": False,
        }

        for adj_obs in obs.adjacent_obstacles:
            c = tuple(adj_obs)
            if c not in self.known_obstacles:
                self.known_obstacles.add(c)
                events["new_obstacle_sensed"] = True
        self.beliefs.mark_obstacles(self.known_obstacles)

        self.beliefs.mark_cell_visited(
            self.position,
            target_found=obs.target_verified_here,
            verified_status=obs.verified_eve_status,
        )

        prev_top, _ = self.beliefs.most_likely_cell()

        if obs.action == ActionType.MOVE and obs.action_succeeded:
            self.steps_since_last_scan += 1
            visits = self.visit_counts.get(self.position, 0) + 1
            self.visit_counts[self.position] = visits
            if visits >= 3:
                events["repeated_visit"] = True

        elif obs.action == ActionType.SCAN and obs.action_succeeded:
            self.steps_since_last_scan = 0
            count = self.scans_per_cell.get(self.position, 0) + 1
            self.scans_per_cell[self.position] = count
            if count >= 2:
                events["repeated_scan"] = True

            self.beliefs.update_from_scan(
                self.position,
                obs.signal_strength,
                modulation_hint=obs.signal_modulation_hint,
            )
            if obs.signal_strength is not None:
                if obs.signal_strength >= 0.42:
                    events["high_signal"] = True
                elif obs.signal_strength < 0.22:
                    events["weak_scan"] = True

        new_top, new_prob = self.beliefs.most_likely_cell()
        if (
            self.previous_top_hypothesis is not None
            and new_top != self.previous_top_hypothesis
            and (abs(new_top[0] - prev_top[0]) + abs(new_top[1] - prev_top[1]) >= 2)
        ):
            events["plan_changed"] = True
            self.last_step_replanned = True
            self.important_decisions.append(
                f"Step {self.step_count}: Replanned route after Bayesian posterior shifted MAP estimate from {self.previous_top_hypothesis} to {new_top} (P={new_prob:.1%})."
            )
        else:
            self.last_step_replanned = False

        self.previous_top_hypothesis = new_top

        if obs.action == ActionType.RECHARGE and obs.action_succeeded:
            self.important_decisions.append(
                f"Step {self.step_count}: Executed solar recharge at {self.position} (battery restored to {self.battery})."
            )
        elif obs.action in (ActionType.ABORT, ActionType.RETURN):
            self.important_decisions.append(
                f"Step {self.step_count}: Executed {obs.action.value} protocol ({obs.status_note})."
            )

        return events

    def decide(self) -> Tuple[ActionType, Coordinate, DecisionExplanation]:
        """Select the next autonomous action using the planner and record the explanation."""
        action, target_cell, explanation = self.planner.select_next_action(
            current_pos=self.position,
            current_battery=self.battery,
            step_count=self.step_count,
            beliefs=self.beliefs,
            resources=self.resources,
            explored_cells=self.explored_cells,
            known_obstacles=self.known_obstacles,
            scans_per_cell=self.scans_per_cell,
            steps_since_last_scan=self.steps_since_last_scan,
            replanned_due_to_evidence=self.last_step_replanned,
        )
        self.current_objective = explanation.objective_mode
        self.planned_route = explanation.planned_route
        self.latest_explanation = explanation
        return action, target_cell, explanation
