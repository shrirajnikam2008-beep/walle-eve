"""Unit tests for Bayesian belief tracking, normalization, underflow protection, and EIG."""
from __future__ import annotations

import math
import numpy as np

from app.agent.beliefs import BeliefTracker
from app.schemas.mission import PublicMissionConfig


def test_uniform_prior_and_obstacle_exclusion() -> None:
    cfg = PublicMissionConfig(
        grid_width=6,
        grid_height=6,
        obstacles=[(1, 1), (2, 2), (3, 3)],
    )
    tracker = BeliefTracker(cfg)
    assert math.isclose(float(np.sum(tracker.grid)), 1.0, rel_tol=1e-6)
    assert tracker.grid[1, 1] == 0.0
    assert tracker.grid[2, 2] == 0.0
    expected_p = 1.0 / (36 - 3)
    assert math.isclose(float(tracker.grid[0, 0]), expected_p, rel_tol=1e-5)


def test_bayesian_update_increases_local_probability_on_strong_signal() -> None:
    cfg = PublicMissionConfig(
        grid_width=8,
        grid_height=8,
        obstacles=[],
        sensor_noise_std=0.08,
        signal_decay_rate=0.35,
    )
    tracker = BeliefTracker(cfg)
    prior_near = float(tracker.grid[2, 3])
    prior_far = float(tracker.grid[7, 7])

    # High reading (0.82) at (2, 2) should boost adjacent (2, 3) and lower distant (7, 7)
    tracker.update_from_scan(sensor_pos=(2, 2), signal_reading=0.82)
    assert math.isclose(float(np.sum(tracker.grid)), 1.0, rel_tol=1e-6)
    assert float(tracker.grid[2, 3]) > prior_near
    assert float(tracker.grid[7, 7]) < prior_far


def test_numerical_stability_and_invalid_readings() -> None:
    cfg = PublicMissionConfig(grid_width=6, grid_height=6, obstacles=[])
    tracker = BeliefTracker(cfg)

    # Repeated extreme observations should never produce NaN or sum != 1.0
    for _ in range(35):
        tracker.update_from_scan((0, 0), 0.99)
        tracker.update_from_scan((5, 5), 0.01)

    total = float(np.sum(tracker.grid))
    assert not math.isnan(total)
    assert math.isclose(total, 1.0, rel_tol=1e-6)

    # Invalid readings (None, NaN, out-of-range) should leave beliefs intact and normalized
    snapshot = tracker.grid.copy()
    tracker.update_from_scan((0, 0), None)
    tracker.update_from_scan((0, 0), float("nan"))
    tracker.update_from_scan((0, 0), -0.5)
    assert np.allclose(tracker.grid, snapshot)


def test_expected_information_gain_and_cell_visit() -> None:
    cfg = PublicMissionConfig(grid_width=8, grid_height=8, obstacles=[])
    tracker = BeliefTracker(cfg)
    eig = tracker.expected_information_gain((4, 4))
    assert eig > 0.0

    # Visiting a cell without finding EVE zeroes that cell and renormalizes
    tracker.mark_cell_visited((4, 4), target_found=False)
    assert tracker.grid[4, 4] == 0.0
    assert math.isclose(float(np.sum(tracker.grid)), 1.0, rel_tol=1e-6)

    # Visiting and verifying EVE collapses distribution to 1.0 at target cell and 0 entropy
    tracker.mark_cell_visited((5, 5), target_found=True)
    assert tracker.grid[5, 5] == 1.0
    assert math.isclose(tracker.entropy(), 0.0, abs_tol=1e-6)
