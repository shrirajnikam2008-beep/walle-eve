"""Integration tests for FastAPI endpoints, response schemas, hidden-state isolation, and benchmarks."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_and_presets() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    presets_res = client.get("/api/presets")
    assert presets_res.status_code == 200
    presets = presets_res.json()
    assert len(presets) >= 5
    for p in presets:
        assert "target_position" not in p
        assert "decoy_positions" not in p


def test_mission_lifecycle_and_hidden_state_isolation() -> None:
    # 1. Create mission
    create_res = client.post("/api/missions", json={"preset_id": "normal", "humor_enabled": True})
    assert create_res.status_code == 201
    state = create_res.json()
    mission_id = state["mission_id"]

    # Ensure active state does NOT leak target coordinates
    assert state["post_mission_evaluation"] is None
    assert "target_position" not in state["config"]

    # 2. Step mission
    step_res = client.post(f"/api/missions/{mission_id}/step")
    assert step_res.status_code == 200
    step_state = step_res.json()
    assert step_state["step_count"] == 1
    assert step_state["latest_explanation"] is not None

    # 3. Pause mission
    pause_res = client.post(f"/api/missions/{mission_id}/pause")
    assert pause_res.status_code == 200
    assert pause_res.json()["status"] == "PAUSED"

    # 4. Run mission to completion
    run_res = client.post(f"/api/missions/{mission_id}/run", json={"max_steps": 100, "humor_enabled": True})
    assert run_res.status_code == 200
    final_state = run_res.json()
    assert final_state["status"] == "SUCCESS"
    assert final_state["post_mission_evaluation"] is not None
    assert final_state["post_mission_evaluation"]["true_target_position"] == [9, 10]

    # 5. Fetch trace
    trace_res = client.get(f"/api/missions/{mission_id}/trace")
    assert trace_res.status_code == 200
    assert len(trace_res.json()) == final_state["step_count"]

    # 6. Reset mission
    reset_res = client.post(f"/api/missions/{mission_id}/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["step_count"] == 0
    assert reset_res.json()["post_mission_evaluation"] is None


def test_invalid_mission_id_and_experiment_endpoint() -> None:
    not_found = client.get("/api/missions/msn-does-not-exist")
    assert not_found.status_code == 404

    exp_res = client.post(
        "/api/experiments/run",
        json={"num_seeds": 4, "base_seed": 200, "noise_levels": [0.08], "include_rl_agent": True},
    )
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert len(exp_data["summary_table"]) == 3
    assert exp_data["interpretation"] != ""
