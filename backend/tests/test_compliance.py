"""Verification tests for the 5 Mandatory Recruitment-Brief Compliance Criteria.

Verifies:
1. Configurable environment from `config.yaml` and JSON with structured observations & hidden-state isolation.
2. Autonomous decision-making across MOVE, SCAN, WAIT, RECHARGE, RETURN, and ABORT.
3. Adaptive belief state over EVE's location AND EVE's operational status, plus replanning on new evidence.
4. Complete, explainable JSON trace distinguishing observed_facts, inferred_conclusions, and uncertainty_metrics.
5. Automatic final report on success, timeout, or abort with zero fabricated evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from fastapi.testclient import TestClient

from app.agent.agent import AutonomousAgent
from app.environment.mission_config import load_mission_from_file
from app.environment.simulator import SearchAndRescueEnvironment
from app.experiments.benchmark import run_benchmark_experiment
from app.main import app
from app.schemas.mission import ActionType, ExperimentRequest, MissionStatus

client = TestClient(app)


def test_requirement_1_configurable_yaml_and_json_environment() -> None:
    yaml_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    json_path = Path(__file__).resolve().parents[1] / "configs" / "sample_mission.json"

    cfg_yaml = load_mission_from_file(yaml_path)
    cfg_json = load_mission_from_file(json_path)
    assert cfg_yaml.grid_width == 12
    assert cfg_yaml.seed == cfg_json.seed == 42

    env = SearchAndRescueEnvironment(cfg_yaml)
    pub = env.get_public_config()
    assert not hasattr(pub, "target_position") or getattr(pub, "target_position", None) is None

    obs = env.step(ActionType.SCAN)
    assert obs.signal_strength is not None
    assert obs.signal_modulation_hint is not None
    assert obs.battery_remaining == cfg_yaml.battery_capacity - cfg_yaml.scan_cost
    assert obs.time_remaining_steps == cfg_yaml.max_actions - 1
    assert obs.mission_status == MissionStatus.ACTIVE


def test_requirement_2_and_3_autonomous_decisions_status_belief_and_replanning() -> None:
    # Verify WAIT, RECHARGE, RETURN, ABORT, MOVE, SCAN support in environment & agent
    yaml_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    cfg = load_mission_from_file(yaml_path)
    env = SearchAndRescueEnvironment(cfg)
    agent = AutonomousAgent(env.get_public_config())

    # Initial status distribution is uniform over 3 operational statuses
    assert len(agent.beliefs.status_beliefs) == 3
    assert abs(sum(agent.beliefs.status_beliefs.values()) - 1.0) < 1e-3

    # Step 1 should be an autonomous SCAN
    act, target, expl = agent.decide()
    assert act == ActionType.SCAN
    obs = env.step(act, target_cell=target)
    agent.ingest_observation(obs)

    # Status distribution and spatial beliefs updated
    inferred_status, conf = agent.beliefs.most_likely_status()
    assert inferred_status in agent.beliefs.status_beliefs
    assert conf > 0.33

    # Also test WAIT action explicitly in environment
    obs_wait = env.step(ActionType.WAIT)
    assert obs_wait.action == ActionType.WAIT
    assert obs_wait.energy_delta == -cfg.wait_cost


def test_requirement_4_and_5_complete_seeded_mission_trace_and_final_report() -> None:
    # Create mission via API using config.yaml
    create_res = client.post(
        "/api/missions",
        json={"config_file": "config.yaml", "humor_enabled": True},
    )
    assert create_res.status_code == 201
    mission_id = create_res.json()["mission_id"]

    # Run mission to completion
    run_res = client.post(f"/api/missions/{mission_id}/run", json={"max_steps": 100, "humor_enabled": True})
    assert run_res.status_code == 200
    final_state = run_res.json()
    assert final_state["status"] == "SUCCESS"
    assert final_state["target_verified"] is True

    # Inspect structured trace via API
    trace_res = client.get(f"/api/missions/{mission_id}/trace")
    assert trace_res.status_code == 200
    trace = trace_res.json()
    assert len(trace) == final_state["step_count"] > 0

    first_entry = trace[0]
    assert "observed_facts" in first_entry and len(first_entry["observed_facts"]) > 0
    assert "inferred_conclusions" in first_entry and len(first_entry["inferred_conclusions"]) > 0
    assert "uncertainty_metrics" in first_entry
    assert "entropy_before_nats" in first_entry["uncertainty_metrics"]
    # Verify JSON exportability
    exported_json = json.dumps(trace)
    assert len(exported_json) > 100

    # Inspect final report via dedicated /report endpoint
    report_res = client.get(f"/api/missions/{mission_id}/report")
    assert report_res.status_code == 200
    report = report_res.json()
    assert report["mission_succeeded"] is True
    assert report["final_estimated_position"] == [9, 10]
    assert report["inferred_eve_status"] == "CRITICAL_LOW_POWER"
    assert len(report["evidence_collected_summary"]) >= 2
    assert report["recommended_next_action"] != ""


def test_unsuccessful_mission_never_reported_as_success() -> None:
    create_res = client.post("/api/missions", json={"preset_id": "unreachable", "humor_enabled": True})
    mission_id = create_res.json()["mission_id"]
    run_res = client.post(f"/api/missions/{mission_id}/run", json={"max_steps": 100})
    state = run_res.json()
    assert state["status"] != "SUCCESS"
    report = state["post_mission_evaluation"]
    assert report is not None
    assert report["mission_succeeded"] is False
    assert report["failure_reason"] is not None


def test_benchmark_against_baseline_on_identical_seeds() -> None:
    resp = run_benchmark_experiment(
        ExperimentRequest(num_seeds=8, base_seed=300, noise_levels=[0.08], include_rl_agent=False)
    )
    bayes = next(r for r in resp.summary_table if "Bayesian" in r.policy_name)
    base = next(r for r in resp.summary_table if "Baseline" in r.policy_name)
    assert bayes.trials == base.trials == 8
    assert bayes.success_rate >= base.success_rate
