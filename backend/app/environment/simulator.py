"""Grid-world Search & Rescue Simulator owning all hidden ground-truth state.

CRITICAL HIDDEN-STATE ISOLATION:
- `_target_position`, `_decoy_positions`, and `_true_eve_status` are private to `SearchAndRescueEnvironment`.
- `get_public_config()` and `observe()` never expose `_target_position` or `_decoy_positions`.
- Ground-truth evaluation is ONLY produced via `build_post_mission_evaluation()` after
  the mission has terminated.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set
import numpy as np

from app.environment.mission_config import PRESET_CONFIGS, generate_seeded_mission
from app.environment.sensors import manhattan_distance, simulate_true_sensor_reading
from app.schemas.mission import (
    ActionType,
    Coordinate,
    EveOperationalStatus,
    FullMissionConfig,
    MissionStatus,
    PostMissionEvaluation,
    PublicMissionConfig,
    SensorObservation,
)


class SearchAndRescueEnvironment:
    """Stateful grid environment that enforces physics, energy, obstacles, and hidden target isolation."""

    def __init__(self, config: Optional[FullMissionConfig] = None) -> None:
        self._full_config: FullMissionConfig = config or PRESET_CONFIGS["normal"]
        self.reset(self._full_config)

    def reset(self, config: Optional[FullMissionConfig] = None) -> SensorObservation:
        """Reset environment state with a deterministic RNG seed."""
        if config is not None:
            self._full_config = config

        cfg = self._full_config
        self._rng = np.random.default_rng(cfg.seed)
        self.width: int = cfg.grid_width
        self.height: int = cfg.grid_height

        self._obstacles: Set[Coordinate] = set(tuple(c) for c in cfg.obstacles)
        self._chargers: Set[Coordinate] = set(tuple(c) for c in cfg.charging_stations)
        self.start_position: Coordinate = tuple(cfg.start_position)

        self._obstacles.discard(self.start_position)

        # Private hidden state — NEVER exposed to agent or active API
        if cfg.target_position is not None:
            self._target_position: Coordinate = tuple(cfg.target_position)
        else:
            generated = generate_seeded_mission(
                seed=cfg.seed,
                noise_std=cfg.sensor_noise_std,
                grid_width=cfg.grid_width,
                grid_height=cfg.grid_height,
                decoy_count=cfg.decoy_count,
            )
            self._target_position = tuple(generated.target_position)  # type: ignore[arg-type]

        if cfg.decoy_positions is not None:
            self._decoy_positions: List[Coordinate] = [tuple(d) for d in cfg.decoy_positions]
        else:
            generated = generate_seeded_mission(
                seed=cfg.seed,
                noise_std=cfg.sensor_noise_std,
                grid_width=cfg.grid_width,
                grid_height=cfg.grid_height,
                decoy_count=cfg.decoy_count,
            )
            self._decoy_positions = [
                tuple(d) for d in (generated.decoy_positions or []) if tuple(d) != self._target_position
            ]

        self._true_eve_status: EveOperationalStatus = cfg.true_eve_status

        self.walle_position: Coordinate = self.start_position
        self.battery: int = cfg.battery_capacity
        self.step_count: int = 0
        self.status: MissionStatus = MissionStatus.ACTIVE
        self.target_verified: bool = False
        self.failure_reason: Optional[str] = None

        self.explored_cells: Set[Coordinate] = {self.walle_position}
        self.known_obstacles: Set[Coordinate] = set()
        self.total_energy_consumed: int = 0
        self.total_scans: int = 0
        self.total_recharges: int = 0
        self.scan_history: List[Dict[str, float | int | List[int] | str]] = []

        self._sense_immediate_neighbors()

        if self.walle_position == self._target_position:
            self.target_verified = True
            self.status = MissionStatus.SUCCESS

        self._latest_observation = SensorObservation(
            step=0,
            action=ActionType.MOVE,
            position=self.walle_position,
            signal_strength=None,
            signal_modulation_hint=None,
            adjacent_clear_cells=self._get_adjacent_clear(self.walle_position),
            adjacent_obstacles=self._get_adjacent_obstacles(self.walle_position),
            cell_inspected=self.walle_position,
            target_verified_here=self.target_verified,
            verified_eve_status=self._true_eve_status if self.target_verified else None,
            is_charging_station=(self.walle_position in self._chargers),
            battery_remaining=self.battery,
            energy_delta=0,
            time_remaining_steps=cfg.max_actions - self.step_count,
            mission_status=self.status,
            action_succeeded=True,
            status_note="Mission initialized. Systems online.",
        )
        return self._latest_observation

    def get_public_config(self) -> PublicMissionConfig:
        """Return a strictly sanitized public mission config without target or decoy locations."""
        return self._full_config.to_public()

    def observe(self) -> SensorObservation:
        """Return the latest agent-safe observation."""
        return self._latest_observation

    def is_within_bounds(self, cell: Coordinate) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def is_obstacle(self, cell: Coordinate) -> bool:
        return cell in self._obstacles

    def _get_adjacent_clear(self, pos: Coordinate) -> List[Coordinate]:
        res: List[Coordinate] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (pos[0] + dx, pos[1] + dy)
            if self.is_within_bounds(nxt) and nxt not in self._obstacles:
                res.append(nxt)
        return res

    def _get_adjacent_obstacles(self, pos: Coordinate) -> List[Coordinate]:
        res: List[Coordinate] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (pos[0] + dx, pos[1] + dy)
            if self.is_within_bounds(nxt) and nxt in self._obstacles:
                res.append(nxt)
        return res

    def _sense_immediate_neighbors(self) -> None:
        """Reveal any obstacle directly adjacent to WALL-E's current position."""
        for obs in self._get_adjacent_obstacles(self.walle_position):
            self.known_obstacles.add(obs)

    def _compute_modulation_hint(self) -> str:
        """Provide a noisy non-positional telemetry hint about signal modulation for EVE status inference."""
        if self._true_eve_status == EveOperationalStatus.OPERATIONAL_SIGNAL:
            return "STEADY_CARRIER"
        if self._true_eve_status == EveOperationalStatus.CRITICAL_LOW_POWER:
            return "PULSED_LOW_VOLTAGE"
        return "MULTIPATH_INTERFERENCE"

    def step(self, action: ActionType, target_cell: Optional[Coordinate] = None) -> SensorObservation:
        """Execute one environment action, enforcing energy, obstacle, and time rules."""
        cfg = self._full_config
        if self.status not in (MissionStatus.ACTIVE, MissionStatus.PAUSED):
            return SensorObservation(
                step=self.step_count,
                action=action,
                position=self.walle_position,
                signal_strength=None,
                signal_modulation_hint=None,
                adjacent_clear_cells=self._get_adjacent_clear(self.walle_position),
                adjacent_obstacles=self._get_adjacent_obstacles(self.walle_position),
                cell_inspected=self.walle_position,
                target_verified_here=self.target_verified,
                verified_eve_status=self._true_eve_status if self.target_verified else None,
                is_charging_station=(self.walle_position in self._chargers),
                battery_remaining=self.battery,
                energy_delta=0,
                time_remaining_steps=max(0, cfg.max_actions - self.step_count),
                mission_status=self.status,
                action_succeeded=False,
                status_note=f"Mission already terminated with status {self.status.value}.",
            )

        if self.status == MissionStatus.PAUSED:
            self.status = MissionStatus.ACTIVE

        self.step_count += 1
        energy_delta = 0
        signal_reading: Optional[float] = None
        modulation_hint: Optional[str] = None
        action_succeeded = True
        status_note = ""

        if action == ActionType.ABORT:
            self.status = MissionStatus.FAILED_UNREACHABLE
            self.failure_reason = (
                "Autonomous agent aborted mission: remaining candidate locations are unreachable or unsafe."
            )
            status_note = "Agent executed mission ABORT protocol due to unreachable target sector."

        elif action == ActionType.RETURN:
            self.status = (
                MissionStatus.FAILED_UNREACHABLE
                if len(self.explored_cells) > 5
                else MissionStatus.SAFE_RETURNED
            )
            self.failure_reason = (
                "Autonomous agent executed safe RETURN protocol before battery exhaustion."
            )
            status_note = "Agent executed safe return-to-base protocol."

        elif action == ActionType.WAIT:
            cost = cfg.wait_cost
            self.battery = max(0, self.battery - cost)
            self.total_energy_consumed += cost
            energy_delta = -cost
            status_note = f"Held position at {self.walle_position} to stabilize telemetry (-{cost} energy)."

        elif action == ActionType.RECHARGE:
            if self.walle_position in self._chargers:
                prev_batt = self.battery
                self.battery = min(cfg.battery_capacity, self.battery + cfg.charge_rate)
                energy_delta = self.battery - prev_batt
                self.total_recharges += 1
                status_note = f"Recharged +{energy_delta} energy units at solar dock {self.walle_position}."
            else:
                cost = 1
                self.battery = max(0, self.battery - cost)
                self.total_energy_consumed += cost
                energy_delta = -cost
                action_succeeded = False
                status_note = f"Recharge failed: no charging dock at {self.walle_position}."

        elif action == ActionType.SCAN:
            cost = cfg.scan_cost
            if self.battery < cost:
                self.battery = 0
                self.status = MissionStatus.FAILED_BATTERY
                self.failure_reason = "Battery depleted while attempting sensor scan."
                action_succeeded = False
                status_note = self.failure_reason
            else:
                self.battery -= cost
                self.total_energy_consumed += cost
                energy_delta = -cost
                self.total_scans += 1
                signal_reading = simulate_true_sensor_reading(
                    sensor_pos=self.walle_position,
                    true_target=self._target_position,
                    decoys=self._decoy_positions,
                    decay_rate=cfg.signal_decay_rate,
                    decoy_strength=cfg.decoy_strength,
                    noise_std=cfg.sensor_noise_std,
                    rng=self._rng,
                )
                modulation_hint = self._compute_modulation_hint()
                self.scan_history.append(
                    {
                        "step": self.step_count,
                        "position": [self.walle_position[0], self.walle_position[1]],
                        "signal": round(signal_reading, 4),
                        "modulation": modulation_hint,
                    }
                )
                if self.walle_position == self._target_position:
                    self.target_verified = True
                    self.status = MissionStatus.SUCCESS
                    status_note = "EVE verified at current coordinates!"
                else:
                    status_note = (
                        f"Radio scan completed at {self.walle_position}: "
                        f"signal={signal_reading:.3f} ({modulation_hint})."
                    )

        elif action == ActionType.MOVE:
            cost = cfg.move_cost
            if self.battery < cost:
                self.battery = 0
                self.status = MissionStatus.FAILED_BATTERY
                self.failure_reason = "Battery depleted prior to movement execution."
                action_succeeded = False
                status_note = self.failure_reason
            elif target_cell is None:
                action_succeeded = False
                status_note = "Move action missing target_cell coordinate."
            else:
                dest = (int(target_cell[0]), int(target_cell[1]))
                dist = manhattan_distance(self.walle_position, dest)
                if dist != 1:
                    action_succeeded = False
                    status_note = f"Invalid move from {self.walle_position} to non-adjacent {dest}."
                elif not self.is_within_bounds(dest):
                    self.battery = max(0, self.battery - 1)
                    self.total_energy_consumed += 1
                    energy_delta = -1
                    action_succeeded = False
                    status_note = f"Movement blocked: {dest} is outside sector boundaries."
                elif self.is_obstacle(dest):
                    self.known_obstacles.add(dest)
                    self.battery = max(0, self.battery - 1)
                    self.total_energy_consumed += 1
                    energy_delta = -1
                    action_succeeded = False
                    status_note = f"Movement blocked: obstacle encountered at {dest}."
                else:
                    self.walle_position = dest
                    self.battery -= cost
                    self.total_energy_consumed += cost
                    energy_delta = -cost
                    self.explored_cells.add(dest)
                    self._sense_immediate_neighbors()

                    if self.walle_position == self._target_position:
                        self.target_verified = True
                        self.status = MissionStatus.SUCCESS
                        status_note = f"Moved to {dest} — physical contact with EVE confirmed!"
                    else:
                        status_note = f"Moved to {dest}. Sector cell clear."

        if self.status == MissionStatus.ACTIVE:
            if self.battery <= 0 and self.walle_position not in self._chargers:
                self.status = MissionStatus.FAILED_BATTERY
                self.failure_reason = "Battery completely exhausted away from charging station."
                status_note = self.failure_reason
            elif self.step_count >= cfg.max_actions:
                self.status = MissionStatus.FAILED_TIMEOUT
                self.failure_reason = f"Maximum mission action budget ({cfg.max_actions} steps) exhausted."
                status_note = self.failure_reason

        self._latest_observation = SensorObservation(
            step=self.step_count,
            action=action,
            position=self.walle_position,
            signal_strength=signal_reading,
            signal_modulation_hint=modulation_hint,
            adjacent_clear_cells=self._get_adjacent_clear(self.walle_position),
            adjacent_obstacles=self._get_adjacent_obstacles(self.walle_position),
            cell_inspected=self.walle_position,
            target_verified_here=self.target_verified,
            verified_eve_status=self._true_eve_status if self.target_verified else None,
            is_charging_station=(self.walle_position in self._chargers),
            battery_remaining=self.battery,
            energy_delta=energy_delta,
            time_remaining_steps=max(0, cfg.max_actions - self.step_count),
            mission_status=self.status,
            action_succeeded=action_succeeded,
            status_note=status_note,
        )
        return self._latest_observation

    def check_is_false_alarm(self, cell: Coordinate) -> bool:
        """Check if a physical visit to `cell` landed on or adjacent to a decoy rather than EVE."""
        if cell == self._target_position:
            return False
        return any(manhattan_distance(cell, d) <= 1 for d in self._decoy_positions)

    def build_post_mission_evaluation(
        self,
        estimated_cell: Coordinate,
        final_confidence: float,
        closing_commentary: str = "Mission concluded.",
        inferred_status: str = "CRITICAL_LOW_POWER",
        inferred_status_conf: float = 0.5,
        status_distribution: Optional[Dict[str, float]] = None,
        remaining_entropy: float = 0.0,
        important_decisions: Optional[List[str]] = None,
    ) -> Optional[PostMissionEvaluation]:
        """Construct automatic final report ONLY after mission termination."""
        if self.status in (MissionStatus.ACTIVE, MissionStatus.PAUSED):
            return None

        status_dist = status_distribution or {inferred_status: inferred_status_conf}
        decisions_list = important_decisions or []

        reveal = self._full_config.reveal_target_on_end
        true_target = self._target_position if reveal else None
        true_decoys = list(self._decoy_positions) if reveal else []
        true_status = self._true_eve_status.value if reveal else None
        err = (
            manhattan_distance(estimated_cell, self._target_position)
            if reveal
            else None
        )
        total_valid_cells = max(1, self.width * self.height - len(self._obstacles))
        succeeded = self.status == MissionStatus.SUCCESS

        evidence_summary = [
            f"Completed {self.total_scans} radio transceiver scans across {len(self.explored_cells)} physically inspected cells ({round(100.0 * len(self.explored_cells) / total_valid_cells, 1)}% sector coverage).",
            f"Executed {self.total_recharges} solar dock recharges; total energy consumed: {self.total_energy_consumed} units.",
        ]
        if self.scan_history:
            peak_scan = max(self.scan_history, key=lambda item: float(item["signal"]))  # type: ignore[arg-type]
            evidence_summary.append(
                f"Strongest radio reading recorded: {peak_scan['signal']} at cell {peak_scan['position']} (step {peak_scan['step']})."
            )

        if succeeded:
            uncertainty_summary = (
                "Zero remaining positional uncertainty (H = 0.0000 nats). EVE physically verified at target coordinates."
            )
            recommended_next = (
                f"Deploy extraction transport to verified coordinates {estimated_cell} with replacement power module for {inferred_status}."
            )
        elif self.status in (MissionStatus.FAILED_UNREACHABLE, MissionStatus.ABORTED):
            uncertainty_summary = (
                f"Residual Shannon entropy H = {remaining_entropy:.4f} nats. Highest posterior probability ({final_confidence:.1%}) is concentrated at {estimated_cell} behind impassable terrain."
            )
            recommended_next = (
                f"Deploy aerial drone or heavy debris-clearing unit to breach containment barrier around {estimated_cell}."
            )
        else:
            uncertainty_summary = (
                f"Mission ended unverified with residual entropy H = {remaining_entropy:.4f} nats. Best unverified Bayesian estimate is {estimated_cell} (confidence {final_confidence:.1%})."
            )
            recommended_next = (
                f"Launch follow-up sortie prioritized toward sector {estimated_cell} with expanded battery reserve."
            )

        return PostMissionEvaluation(
            mission_succeeded=succeeded,
            termination_status=self.status,
            failure_reason=self.failure_reason,
            true_target_position=true_target,
            true_decoy_positions=true_decoys,
            true_eve_status=true_status,
            final_estimated_position=estimated_cell,
            final_estimation_error_manhattan=err,
            final_confidence=round(final_confidence, 4),
            inferred_eve_status=inferred_status,
            inferred_eve_status_confidence=round(inferred_status_conf, 4),
            eve_status_distribution=status_dist,
            remaining_entropy_nats=round(remaining_entropy, 4),
            remaining_uncertainty_summary=uncertainty_summary,
            evidence_collected_summary=evidence_summary,
            important_decisions=decisions_list[-6:],
            recommended_next_action=recommended_next,
            total_actions=self.step_count,
            total_energy_consumed=self.total_energy_consumed,
            total_scans=self.total_scans,
            total_recharges=self.total_recharges,
            cells_explored_count=len(self.explored_cells),
            coverage_ratio=round(len(self.explored_cells) / total_valid_cells, 4),
            closing_commentary=closing_commentary,
        )
