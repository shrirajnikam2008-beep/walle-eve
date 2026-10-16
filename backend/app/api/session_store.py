"""Mission Session Orchestrator & In-Memory Store."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional

from app.agent.agent import AutonomousAgent
from app.environment.mission_config import PRESET_CONFIGS, load_mission_from_file
from app.environment.simulator import SearchAndRescueEnvironment
from app.personality.chaos_meter import ChaosMeter
from app.personality.commentary import CommentaryEngine
from app.schemas.mission import (
    ActionType,
    FullMissionConfig,
    MissionStatus,
    PublicMissionState,
    RadioMessage,
    StepTraceEntry,
)


class MissionSession:
    """Encapsulates one active or completed mission with strict hidden-state isolation."""

    def __init__(
        self,
        mission_id: str,
        full_config: FullMissionConfig,
        humor_enabled: bool = True,
    ) -> None:
        self.mission_id: str = mission_id
        self._full_config: FullMissionConfig = full_config
        self.humor_enabled: bool = humor_enabled

        self.env = SearchAndRescueEnvironment(full_config)
        self.agent = AutonomousAgent(self.env.get_public_config())
        self.chaos = ChaosMeter()
        self.commentary = CommentaryEngine(enabled=humor_enabled)

        self.commentary_log: List[RadioMessage] = []
        self.trace: List[StepTraceEntry] = []
        self.latest_commentary: Optional[RadioMessage] = None

        start_msg = self.commentary.generate(0, "MISSION_START", force=True)
        if start_msg is not None:
            self.latest_commentary = start_msg
            self.commentary_log.append(start_msg)

    def reset(self, humor_enabled: Optional[bool] = None) -> PublicMissionState:
        """Reset the mission to step 0 with identical seed and configuration."""
        if humor_enabled is not None:
            self.humor_enabled = humor_enabled

        self.env.reset(self._full_config)
        self.agent = AutonomousAgent(self.env.get_public_config())
        self.chaos.reset()
        self.commentary.reset(enabled=self.humor_enabled)
        self.commentary_log.clear()
        self.trace.clear()

        start_msg = self.commentary.generate(0, "MISSION_START", force=True)
        self.latest_commentary = start_msg
        if start_msg is not None:
            self.commentary_log.append(start_msg)

        return self.get_public_state()

    def set_humor(self, enabled: bool) -> None:
        self.humor_enabled = enabled
        self.commentary.enabled = enabled

    def pause(self) -> PublicMissionState:
        if self.env.status == MissionStatus.ACTIVE:
            self.env.status = MissionStatus.PAUSED
        return self.get_public_state()

    def step_once(self) -> PublicMissionState:
        """Execute a single autonomous decision-action-observation-belief cycle."""
        if self.env.status not in (MissionStatus.ACTIVE, MissionStatus.PAUSED):
            return self.get_public_state()

        if self.env.status == MissionStatus.PAUSED:
            self.env.status = MissionStatus.ACTIVE

        prev_pos = self.env.walle_position
        batt_before = self.env.battery
        status_before = self.env.status
        entropy_before = self.agent.beliefs.entropy()

        action, target_cell, explanation = self.agent.decide()
        obs = self.env.step(action, target_cell=target_cell)
        cog_events = self.agent.ingest_observation(obs)

        is_false_alarm = (
            action == ActionType.MOVE
            and obs.action_succeeded
            and not obs.target_verified_here
            and self.env.check_is_false_alarm(obs.position)
        )

        mission_failed = self.env.status in (
            MissionStatus.FAILED_BATTERY,
            MissionStatus.FAILED_TIMEOUT,
            MissionStatus.FAILED_UNREACHABLE,
            MissionStatus.SAFE_RETURNED,
            MissionStatus.ABORTED,
        )

        chaos_state = self.chaos.evaluate_step(
            step=obs.step,
            battery=obs.battery_remaining,
            battery_capacity=self.env.get_public_config().battery_capacity,
            action_succeeded=obs.action_succeeded,
            is_false_alarm=is_false_alarm,
            weak_scan=cog_events["weak_scan"],
            repeated_action=(cog_events["repeated_visit"] or cog_events["repeated_scan"]),
            plan_changed=cog_events["plan_changed"],
            recharged=(action == ActionType.RECHARGE and obs.action_succeeded),
            mission_failed=mission_failed,
        )

        chosen_event: Optional[str] = None
        if self.env.status == MissionStatus.SUCCESS:
            chosen_event = "MISSION_SUCCESS"
        elif self.env.status == MissionStatus.FAILED_TIMEOUT:
            chosen_event = "MISSION_TIMEOUT"
        elif self.env.status in (MissionStatus.FAILED_UNREACHABLE, MissionStatus.SAFE_RETURNED, MissionStatus.ABORTED):
            chosen_event = "MISSION_UNREACHABLE"
        elif self.env.status == MissionStatus.FAILED_BATTERY:
            chosen_event = "MISSION_FAILURE"
        elif is_false_alarm:
            chosen_event = "FALSE_ALARM"
        elif action == ActionType.RECHARGE and obs.action_succeeded:
            chosen_event = "CHARGING"
        elif obs.battery_remaining <= int(0.22 * self.env.get_public_config().battery_capacity):
            chosen_event = "LOW_BATTERY"
        elif cog_events["high_signal"]:
            chosen_event = "SIGNAL_DETECTED"
        elif cog_events["plan_changed"]:
            chosen_event = "PLAN_CHANGED"
        elif cog_events["new_obstacle_sensed"] or not obs.action_succeeded:
            chosen_event = "OBSTACLE"
        elif cog_events["repeated_scan"] or cog_events["repeated_visit"]:
            chosen_event = "INEFFICIENT_ROUTE"
        elif cog_events["weak_scan"]:
            chosen_event = "WEAK_SCAN"

        msg: Optional[RadioMessage] = None
        if chosen_event is not None:
            msg = self.commentary.generate(obs.step, chosen_event)
            if msg is not None:
                self.latest_commentary = msg
                self.commentary_log.append(msg)

        top_cell, top_prob = self.agent.beliefs.most_likely_cell()
        inferred_status, status_prob = self.agent.beliefs.most_likely_status()
        entropy_after = self.agent.beliefs.entropy()

        # Construct structured Observed Facts vs. Inferred Conclusions vs. Uncertainty
        observed_facts = [
            f"Position={obs.position}, Battery={obs.battery_remaining} (delta={obs.energy_delta:+d}), TimeLeft={obs.time_remaining_steps} steps.",
            f"Target verified at {obs.position}: {obs.target_verified_here}.",
        ]
        if obs.signal_strength is not None:
            observed_facts.append(
                f"Measured radio signal={obs.signal_strength:.4f} (modulation={obs.signal_modulation_hint})."
            )
        if obs.adjacent_obstacles:
            observed_facts.append(f"Adjacent impassable obstacles sensed at: {obs.adjacent_obstacles}.")

        inferred_conclusions = [
            f"Maximum-a-posteriori (MAP) target coordinate estimate: {top_cell} (unverified posterior P={top_prob:.2%}).",
            f"Inferred EVE operational status: {inferred_status} (P={status_prob:.1%}).",
        ]
        if cog_events["plan_changed"]:
            inferred_conclusions.append(
                f"Evidence triggered route replanning toward new probability peak {top_cell}."
            )

        uncertainty_metrics: Dict[str, float | str] = {
            "entropy_before_nats": round(entropy_before, 4),
            "entropy_after_nats": round(entropy_after, 4),
            "entropy_delta_nats": round(entropy_after - entropy_before, 4),
            "map_confidence": round(top_prob, 4),
            "verification_state": "VERIFIED" if obs.target_verified_here else "UNVERIFIED_ESTIMATE",
        }

        self.trace.append(
            StepTraceEntry(
                step=obs.step,
                action=action,
                from_position=prev_pos,
                to_position=obs.position,
                battery_before=batt_before,
                battery_after=obs.battery_remaining,
                energy_delta=obs.energy_delta,
                mission_status_before=status_before,
                mission_status_after=self.env.status,
                signal_reading=obs.signal_strength,
                observed_facts=observed_facts,
                inferred_conclusions=inferred_conclusions,
                uncertainty_metrics=uncertainty_metrics,
                entropy_before=round(entropy_before, 4),
                entropy_after=round(entropy_after, 4),
                most_likely_cell=top_cell,
                max_confidence=round(top_prob, 4),
                inferred_eve_status=inferred_status,
                eve_status_belief=dict(self.agent.beliefs.status_beliefs),
                replanned=cog_events["plan_changed"],
                objective_mode=explanation.objective_mode,
                technical_rationale=explanation.technical_rationale,
                commentary=msg,
                chaos_stars=chaos_state.stars,
            )
        )

        return self.get_public_state()

    def run_until_done(self, max_steps: int = 120) -> PublicMissionState:
        """Run bounded autonomous steps until mission termination or `max_steps`."""
        bounded_limit = min(max_steps, self.env.get_public_config().max_actions + 5)
        steps_taken = 0
        while (
            self.env.status in (MissionStatus.ACTIVE, MissionStatus.PAUSED)
            and steps_taken < bounded_limit
        ):
            self.step_once()
            steps_taken += 1
        return self.get_public_state()

    def get_public_state(self) -> PublicMissionState:
        """Construct the public mission state without leaking hidden target coordinates."""
        pub_cfg = self.env.get_public_config()
        top_cell, top_prob = self.agent.beliefs.most_likely_cell()
        inferred_status, status_conf = self.agent.beliefs.most_likely_status()
        entropy_val = self.agent.beliefs.entropy()

        closing_text = (
            self.latest_commentary.message
            if self.latest_commentary is not None
            else "Mission concluded. Telemetry archived."
        )
        post_eval = self.env.build_post_mission_evaluation(
            estimated_cell=top_cell,
            final_confidence=top_prob,
            inferred_status=inferred_status,
            inferred_status_conf=status_conf,
            status_distribution=dict(self.agent.beliefs.status_beliefs),
            remaining_entropy=entropy_val,
            important_decisions=self.agent.important_decisions,
            closing_commentary=closing_text,
        )

        total_valid_cells = max(1, pub_cfg.grid_width * pub_cfg.grid_height - len(pub_cfg.obstacles))

        return PublicMissionState(
            mission_id=self.mission_id,
            config=pub_cfg,
            status=self.env.status,
            step_count=self.env.step_count,
            max_actions=pub_cfg.max_actions,
            walle_position=self.env.walle_position,
            battery=self.env.battery,
            battery_capacity=pub_cfg.battery_capacity,
            explored_cells=sorted(list(self.env.explored_cells)),
            known_obstacles=sorted(list(self.agent.known_obstacles)),
            scan_history=list(self.env.scan_history),
            belief_map=self.agent.beliefs.to_nested_list(),
            entropy=round(entropy_val, 4),
            most_likely_cell=top_cell,
            max_confidence=round(top_prob, 4),
            inferred_eve_status=inferred_status,
            eve_status_belief=dict(self.agent.beliefs.status_beliefs),
            target_verified=self.env.target_verified,
            current_objective=self.agent.current_objective,
            planned_route=list(self.agent.planned_route),
            latest_explanation=self.agent.latest_explanation,
            latest_observation=self.env.observe(),
            latest_commentary=self.latest_commentary,
            commentary_log=list(reversed(self.commentary_log[-15:])),
            chaos=self.chaos.get_state(),
            metrics={
                "total_energy_consumed": self.env.total_energy_consumed,
                "total_scans": self.env.total_scans,
                "total_recharges": self.env.total_recharges,
                "cells_explored": len(self.env.explored_cells),
                "coverage_pct": round(100.0 * len(self.env.explored_cells) / total_valid_cells, 1),
            },
            post_mission_evaluation=post_eval,
        )


class MissionStore:
    """Bounded in-memory session store for local missions."""

    def __init__(self, max_sessions: int = 50) -> None:
        self._sessions: Dict[str, MissionSession] = {}
        self.max_sessions = max_sessions

    def create(
        self,
        preset_id: Optional[str] = "normal",
        custom_config: Optional[FullMissionConfig] = None,
        config_file: Optional[str] = None,
        humor_enabled: bool = True,
    ) -> MissionSession:
        if len(self._sessions) >= self.max_sessions:
            oldest_key = next(iter(self._sessions))
            del self._sessions[oldest_key]

        if config_file is not None:
            base_dir = Path(__file__).resolve().parents[2] / "configs"
            candidate_path = (base_dir / Path(config_file).name) if not Path(config_file).is_absolute() else Path(config_file)
            cfg = load_mission_from_file(candidate_path)
        elif custom_config is not None:
            cfg = custom_config
        else:
            key = preset_id if (preset_id and preset_id in PRESET_CONFIGS) else "normal"
            cfg = PRESET_CONFIGS[key]

        mission_id = f"msn-{uuid.uuid4().hex[:8]}"
        session = MissionSession(mission_id=mission_id, full_config=cfg, humor_enabled=humor_enabled)
        self._sessions[mission_id] = session
        return session

    def get(self, mission_id: str) -> Optional[MissionSession]:
        return self._sessions.get(mission_id)
