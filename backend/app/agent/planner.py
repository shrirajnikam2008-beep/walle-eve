"""Information-Gain & Utility-Driven Action Planner for WALL-E."""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from app.agent.beliefs import BeliefTracker
from app.agent.pathfinding import astar_shortest_path, compute_distance_map
from app.agent.resource_manager import ResourceManager
from app.schemas.mission import (
    ActionType,
    Coordinate,
    DecisionExplanation,
    PublicMissionConfig,
)


class ActionPlanner:
    """Selects autonomous actions using Bayesian beliefs, EIG, A* pathfinding, and energy constraints."""

    # Calibrated utility weights so even modest probability peaks beat distant exploration penalty
    WEIGHT_PROB: float = 28.0
    WEIGHT_NEIGHBOR_MASS: float = 8.0
    WEIGHT_INFO_GAIN: float = 2.5
    WEIGHT_DISTANCE_PENALTY: float = 0.65

    def __init__(self, config: PublicMissionConfig) -> None:
        self.config = config
        self.width = config.grid_width
        self.height = config.grid_height

    def select_next_action(
        self,
        current_pos: Coordinate,
        current_battery: int,
        step_count: int,
        beliefs: BeliefTracker,
        resources: ResourceManager,
        explored_cells: Set[Coordinate],
        known_obstacles: Set[Coordinate],
        scans_per_cell: Dict[Coordinate, int],
        steps_since_last_scan: int,
        replanned_due_to_evidence: bool = False,
    ) -> Tuple[ActionType, Coordinate, DecisionExplanation]:
        """Compute the next autonomous action and a detailed `DecisionExplanation`."""
        entropy_now = beliefs.entropy()
        top_cell, top_prob = beliefs.most_likely_cell()
        inferred_status, status_conf = beliefs.most_likely_status()
        status_dist = dict(beliefs.status_beliefs)
        eig_here = beliefs.expected_information_gain(current_pos)

        _, charger_path, charger_steps = resources.nearest_charger_path(current_pos, known_obstacles)
        nearest_charger_dist = charger_steps if charger_path is not None else -1
        safety_margin = current_battery - (
            (charger_steps * self.config.move_cost) if charger_path is not None else 0
        )
        risk_level = (
            "CRITICAL"
            if safety_margin <= 4
            else ("MODERATE" if safety_margin <= 14 else "LOW")
        )

        # 1. Check if emergency recharge or charger navigation is required
        need_charge, charge_route, charge_reason = resources.should_recharge_now(
            current_pos, current_battery, known_obstacles
        )
        if need_charge and charge_route is not None:
            if len(charge_route) == 1 and current_pos == charge_route[0]:
                explanation = DecisionExplanation(
                    chosen_action=ActionType.RECHARGE,
                    target_cell=current_pos,
                    objective_mode="SOLAR_RECHARGE",
                    entropy_before=round(entropy_now, 4),
                    expected_entropy_reduction=0.0,
                    top_hypothesis_cell=top_cell,
                    top_hypothesis_prob=round(top_prob, 4),
                    inferred_eve_status=inferred_status,
                    eve_status_distribution=status_dist,
                    planned_route=[current_pos],
                    travel_cost_steps=0,
                    estimated_energy_required=0,
                    nearest_charger_distance=0,
                    safety_margin_energy=safety_margin,
                    risk_assessment=risk_level,
                    replanned_due_to_evidence=replanned_due_to_evidence,
                    candidate_scores_summary=[{"action": "RECHARGE", "score": 10.0}],
                    technical_rationale=charge_reason,
                )
                return ActionType.RECHARGE, current_pos, explanation
            elif len(charge_route) >= 2:
                next_step = charge_route[1]
                if current_battery >= self.config.move_cost:
                    explanation = DecisionExplanation(
                        chosen_action=ActionType.MOVE,
                        target_cell=next_step,
                        objective_mode="RETURN_TO_CHARGER",
                        entropy_before=round(entropy_now, 4),
                        expected_entropy_reduction=0.0,
                        top_hypothesis_cell=top_cell,
                        top_hypothesis_prob=round(top_prob, 4),
                        inferred_eve_status=inferred_status,
                        eve_status_distribution=status_dist,
                        planned_route=charge_route,
                        travel_cost_steps=len(charge_route) - 1,
                        estimated_energy_required=(len(charge_route) - 1) * self.config.move_cost,
                        nearest_charger_distance=nearest_charger_dist,
                        safety_margin_energy=safety_margin,
                        risk_assessment=risk_level,
                        replanned_due_to_evidence=replanned_due_to_evidence,
                        candidate_scores_summary=[
                            {"action": f"MOVE->{next_step} (charger)", "score": 9.5}
                        ],
                        technical_rationale=charge_reason,
                    )
                    return ActionType.MOVE, next_step, explanation

        # 2. Compute BFS distances from current_pos to all reachable cells
        dist_map = compute_distance_map([current_pos], self.width, self.height, known_obstacles)

        reachable_candidates: List[Coordinate] = []
        for x in range(self.width):
            for y in range(self.height):
                cell = (x, y)
                if (
                    cell not in explored_cells
                    and cell not in known_obstacles
                    and cell in dist_map
                    and beliefs.grid[x, y] > 0.0
                ):
                    reachable_candidates.append(cell)

        # If no reachable candidate cell exists anywhere on the map, abort/terminate safely
        if not reachable_candidates:
            explanation = DecisionExplanation(
                chosen_action=ActionType.ABORT,
                target_cell=current_pos,
                objective_mode="ABORT_UNREACHABLE",
                entropy_before=round(entropy_now, 4),
                expected_entropy_reduction=0.0,
                top_hypothesis_cell=top_cell,
                top_hypothesis_prob=round(top_prob, 4),
                inferred_eve_status=inferred_status,
                eve_status_distribution=status_dist,
                planned_route=[current_pos],
                travel_cost_steps=0,
                estimated_energy_required=0,
                nearest_charger_distance=nearest_charger_dist,
                safety_margin_energy=safety_margin,
                risk_assessment="HIGH",
                replanned_due_to_evidence=replanned_due_to_evidence,
                candidate_scores_summary=[{"action": "ABORT", "score": 1.0}],
                technical_rationale=(
                    f"All remaining unvisited candidate cells (including top hypothesis {top_cell} "
                    f"at p={top_prob:.2%}) are sealed behind impassable obstacles. Executing safe ABORT."
                ),
            )
            return ActionType.ABORT, current_pos, explanation

        # Also check if >= 80% of total probability mass is concentrated in unreachable cells
        reachable_mass = sum(float(beliefs.grid[c[0], c[1]]) for c in reachable_candidates)
        if reachable_mass < 0.12 and step_count >= 12:
            explanation = DecisionExplanation(
                chosen_action=ActionType.ABORT,
                target_cell=current_pos,
                objective_mode="ABORT_UNREACHABLE_POSTERIOR",
                entropy_before=round(entropy_now, 4),
                expected_entropy_reduction=0.0,
                top_hypothesis_cell=top_cell,
                top_hypothesis_prob=round(top_prob, 4),
                inferred_eve_status=inferred_status,
                eve_status_distribution=status_dist,
                planned_route=[current_pos],
                travel_cost_steps=0,
                estimated_energy_required=0,
                nearest_charger_distance=nearest_charger_dist,
                safety_margin_energy=safety_margin,
                risk_assessment="HIGH",
                replanned_due_to_evidence=replanned_due_to_evidence,
                candidate_scores_summary=[{"action": "ABORT", "score": 1.0}],
                technical_rationale=(
                    f"Over {(1.0 - reachable_mass):.1%} of Bayesian posterior probability mass is located inside "
                    f"unreachable sealed terrain (top hypothesis {top_cell} at P={top_prob:.1%}). Aborting safely."
                ),
            )
            return ActionType.ABORT, current_pos, explanation

        # 3. Evaluate whether a SCAN at current_pos is high-value right now
        scans_here = scans_per_cell.get(current_pos, 0)
        can_scan_safely, _, _ = resources.can_execute_and_reach_charger(
            current_pos, self.config.scan_cost, current_battery, known_obstacles
        )

        should_scan = False
        scan_rationale = ""
        if can_scan_safely and current_battery >= self.config.scan_cost and scans_here == 0:
            if step_count == 0:
                should_scan = True
                scan_rationale = (
                    f"Initial baseline scan at {current_pos}: prior entropy H={entropy_now:.3f} nats, "
                    f"expected entropy reduction EIG={eig_here:.3f} nats."
                )
            elif steps_since_last_scan >= 4 and top_prob < 0.28 and eig_here >= 0.04:
                should_scan = True
                scan_rationale = (
                    f"Triangulation scan at {current_pos} after {steps_since_last_scan} movement steps: "
                    f"current entropy H={entropy_now:.3f} nats, max P={top_prob:.1%} at {top_cell}, "
                    f"expected entropy reduction EIG={eig_here:.3f} nats."
                )

        if should_scan:
            explanation = DecisionExplanation(
                chosen_action=ActionType.SCAN,
                target_cell=current_pos,
                objective_mode="INFORMATION_GAIN_SCAN",
                entropy_before=round(entropy_now, 4),
                expected_entropy_reduction=round(eig_here, 4),
                top_hypothesis_cell=top_cell,
                top_hypothesis_prob=round(top_prob, 4),
                inferred_eve_status=inferred_status,
                eve_status_distribution=status_dist,
                planned_route=[current_pos],
                travel_cost_steps=0,
                estimated_energy_required=self.config.scan_cost,
                nearest_charger_distance=nearest_charger_dist,
                safety_margin_energy=safety_margin,
                risk_assessment=risk_level,
                replanned_due_to_evidence=replanned_due_to_evidence,
                candidate_scores_summary=[
                    {"action": f"SCAN@{current_pos}", "score": round(eig_here * 10.0, 3)},
                    {"action": f"MOVE->{top_cell}", "score": round(top_prob * 28.0, 3)},
                ],
                technical_rationale=scan_rationale,
            )
            return ActionType.SCAN, current_pos, explanation

        # 4. Score all reachable candidate cells using Bayesian probability + local cluster mass - travel cost
        max_grid_dist = float(self.width + self.height)
        scored_candidates: List[Tuple[float, Coordinate, float, int]] = []

        for cell in reachable_candidates:
            cx, cy = cell
            prob = float(beliefs.grid[cx, cy])
            steps_to_cell = dist_map[cell]

            neighbor_mass = 0.0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor_mass += float(beliefs.grid[nx, ny])

            norm_dist = steps_to_cell / max_grid_dist
            score = (
                self.WEIGHT_PROB * prob
                + self.WEIGHT_NEIGHBOR_MASS * neighbor_mass
                - self.WEIGHT_DISTANCE_PENALTY * norm_dist
            )
            scored_candidates.append((score, cell, prob, steps_to_cell))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_goal, best_goal_prob, best_goal_steps = scored_candidates[0]

        path = astar_shortest_path(current_pos, best_goal, self.width, self.height, known_obstacles)
        if path is None or len(path) < 2:
            explanation = DecisionExplanation(
                chosen_action=ActionType.RETURN,
                target_cell=current_pos,
                objective_mode="SAFE_FALLBACK",
                entropy_before=round(entropy_now, 4),
                expected_entropy_reduction=0.0,
                top_hypothesis_cell=top_cell,
                top_hypothesis_prob=round(top_prob, 4),
                inferred_eve_status=inferred_status,
                eve_status_distribution=status_dist,
                planned_route=[current_pos],
                travel_cost_steps=0,
                estimated_energy_required=0,
                nearest_charger_distance=nearest_charger_dist,
                safety_margin_energy=safety_margin,
                risk_assessment="HIGH",
                replanned_due_to_evidence=replanned_due_to_evidence,
                candidate_scores_summary=[],
                technical_rationale="No valid A* route found to any remaining candidate cell; executing safe fallback.",
            )
            return ActionType.RETURN, current_pos, explanation

        next_cell = path[1]
        if current_battery < self.config.move_cost:
            if current_pos in self.config.charging_stations:
                return (
                    ActionType.RECHARGE,
                    current_pos,
                    DecisionExplanation(
                        chosen_action=ActionType.RECHARGE,
                        target_cell=current_pos,
                        objective_mode="EMERGENCY_RECHARGE",
                        entropy_before=round(entropy_now, 4),
                        expected_entropy_reduction=0.0,
                        top_hypothesis_cell=top_cell,
                        top_hypothesis_prob=round(top_prob, 4),
                        inferred_eve_status=inferred_status,
                        eve_status_distribution=status_dist,
                        planned_route=[current_pos],
                        travel_cost_steps=0,
                        estimated_energy_required=0,
                        nearest_charger_distance=0,
                        safety_margin_energy=0,
                        risk_assessment="CRITICAL",
                        replanned_due_to_evidence=replanned_due_to_evidence,
                        candidate_scores_summary=[{"action": "RECHARGE", "score": 10.0}],
                        technical_rationale="Insufficient battery for movement; recharging at current dock.",
                    ),
                )
            return (
                ActionType.RETURN,
                current_pos,
                DecisionExplanation(
                    chosen_action=ActionType.RETURN,
                    target_cell=current_pos,
                    objective_mode="BATTERY_RESERVE_RETURN",
                    entropy_before=round(entropy_now, 4),
                    expected_entropy_reduction=0.0,
                    top_hypothesis_cell=top_cell,
                    top_hypothesis_prob=round(top_prob, 4),
                    inferred_eve_status=inferred_status,
                    eve_status_distribution=status_dist,
                    planned_route=[current_pos],
                    travel_cost_steps=0,
                    estimated_energy_required=0,
                    nearest_charger_distance=nearest_charger_dist,
                    safety_margin_energy=safety_margin,
                    risk_assessment="CRITICAL",
                    replanned_due_to_evidence=replanned_due_to_evidence,
                    candidate_scores_summary=[{"action": "RETURN", "score": 1.0}],
                    technical_rationale=f"Battery ({current_battery}) is below movement cost ({self.config.move_cost}); terminating safely.",
                ),
            )

        objective_mode = (
            "CONVERGE_HIGH_PROBABILITY" if best_goal_prob >= 0.04 else "EXPLORE_UNCERTAIN_FRONTIER"
        )
        top_candidates_summary = [
            {
                "action": f"GOAL {c} (d={st})",
                "score": round(sc, 3),
                "prob": round(pr, 4),
            }
            for sc, c, pr, st in scored_candidates[:3]
        ]

        replan_note = " [REPLANNED after posterior shift] " if replanned_due_to_evidence else " "
        rationale = (
            f"Navigating via A* toward candidate {best_goal} (next step {next_cell}, distance={best_goal_steps} steps, "
            f"utility={best_score:.3f}, posterior P={best_goal_prob:.2%}).{replan_note}"
            f"Entropy H={entropy_now:.3f} nats; top hypothesis={top_cell} ({top_prob:.2%}); "
            f"inferred EVE status={inferred_status} ({status_conf:.0%})."
        )

        explanation = DecisionExplanation(
            chosen_action=ActionType.MOVE,
            target_cell=next_cell,
            objective_mode=objective_mode,
            entropy_before=round(entropy_now, 4),
            expected_entropy_reduction=round(eig_here, 4),
            top_hypothesis_cell=top_cell,
            top_hypothesis_prob=round(top_prob, 4),
            inferred_eve_status=inferred_status,
            eve_status_distribution=status_dist,
            planned_route=path,
            travel_cost_steps=best_goal_steps,
            estimated_energy_required=best_goal_steps * self.config.move_cost,
            nearest_charger_distance=nearest_charger_dist,
            safety_margin_energy=safety_margin,
            risk_assessment=risk_level,
            replanned_due_to_evidence=replanned_due_to_evidence,
            candidate_scores_summary=top_candidates_summary,
            technical_rationale=rationale,
        )
        return ActionType.MOVE, next_cell, explanation
