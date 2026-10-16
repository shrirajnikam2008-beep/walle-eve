"""Verification script that executes all 6 required mission scenarios and the 30-seed benchmark."""
from __future__ import annotations

import json
from app.api.session_store import MissionSession
from app.environment.mission_config import PRESET_CONFIGS
from app.experiments.benchmark import run_benchmark_experiment
from app.schemas.mission import ExperimentRequest, FullMissionConfig


def main() -> None:
    print("=== PART 1: MANUAL SCENARIO VERIFICATION ===")
    scenarios = [
        ("1. Normal Mission", PRESET_CONFIGS["normal"]),
        ("2. Misleading-Signal (Decoy) Mission", PRESET_CONFIGS["misleading_decoy"]),
        ("3. Low-Battery Mission", PRESET_CONFIGS["low_battery"]),
        ("4. Unreachable-Target Mission", PRESET_CONFIGS["unreachable"]),
        ("5. High-Noise Successful Mission", PRESET_CONFIGS["high_noise"]),
        (
            "6. Failed (Timeout/Depleted) Mission",
            FullMissionConfig(
                mission_name="Severe Timeout Test",
                preset_id="failed_demo",
                grid_width=12,
                grid_height=12,
                seed=99,
                start_position=(0, 0),
                target_position=(11, 11),
                battery_capacity=100,
                max_actions=10,
            ),
        ),
    ]

    for label, cfg in scenarios:
        session = MissionSession(mission_id=f"verify-{cfg.preset_id}", full_config=cfg, humor_enabled=True)
        state = session.run_until_done(max_steps=150)
        ev = state.post_mission_evaluation
        assert ev is not None, f"PostMissionEvaluation missing for {label}"
        print(
            f"{label:38s} | Status: {state.status.value:18s} | Steps: {state.step_count:3d} | "
            f"Scans: {ev.total_scans:2d} | Recharges: {ev.total_recharges:2d} | "
            f"Energy: {ev.total_energy_consumed:3d} | MAP Est: {str(ev.final_estimated_position):8s} "
            f"(Conf: {ev.final_confidence:.2f}) | Inferred Status: {ev.inferred_eve_status}"
        )

    print("\n=== PART 2: 30-SEED REPRODUCIBLE BENCHMARK ===")
    bench = run_benchmark_experiment(
        ExperimentRequest(
            num_seeds=30,
            base_seed=100,
            noise_levels=[0.06, 0.10, 0.16],
            include_rl_agent=True,
        )
    )
    print(f"{'Policy':35s} | {'Noise':5s} | {'Trials':6s} | {'Success%':8s} | {'AvgAct(Win)':11s} | {'AvgAct(All)':11s} | {'AvgEnergy':9s} | {'AvgScans':8s} | {'Failed%':7s}")
    print("-" * 118)
    for r in bench.summary_table:
        print(
            f"{r.policy_name:35s} | {r.noise_std:5.2f} | {r.trials:6d} | {r.success_rate:7.1f}% | "
            f"{r.avg_actions_on_success:11.1f} | {r.avg_actions_overall:11.1f} | {r.avg_energy_consumed:9.1f} | "
            f"{r.avg_scans:8.1f} | {r.stranded_or_failed_rate:6.1f}%"
        )
    print("\nInterpretation:")
    print(bench.interpretation)


if __name__ == "__main__":
    main()
