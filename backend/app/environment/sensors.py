"""Sensor simulation and likelihood models for WALL-E's radio transceiver."""
from __future__ import annotations

import math
from typing import List
import numpy as np

from app.schemas.mission import Coordinate


def euclidean_distance(a: Coordinate, b: Coordinate) -> float:
    """Compute Euclidean distance between two grid coordinates."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def manhattan_distance(a: Coordinate, b: Coordinate) -> int:
    """Compute Manhattan distance between two grid coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def expected_signal_from_target(
    sensor_pos: Coordinate,
    hypothesized_target: Coordinate,
    decay_rate: float,
) -> float:
    """Expected clean signal strength at sensor_pos if EVE is at hypothesized_target.

    Mathematical Model:
        mu(x, t) = exp(-lambda * ||x - t||_2)
    where lambda is `decay_rate`. At distance 0, expected signal is 1.0.
    """
    dist = euclidean_distance(sensor_pos, hypothesized_target)
    return float(math.exp(-decay_rate * dist))


def simulate_true_sensor_reading(
    sensor_pos: Coordinate,
    true_target: Coordinate,
    decoys: List[Coordinate],
    decay_rate: float,
    decoy_strength: float,
    noise_std: float,
    rng: np.random.Generator,
) -> float:
    """Simulate a noisy radio signal reading in the environment.

    Note: ONLY the environment calls this function because it uses `true_target`
    and `decoys`. The agent never has access to those coordinates.
    """
    true_emit = expected_signal_from_target(sensor_pos, true_target, decay_rate)
    max_emit = true_emit
    for decoy in decoys:
        decoy_emit = decoy_strength * expected_signal_from_target(sensor_pos, decoy, decay_rate)
        if decoy_emit > max_emit:
            max_emit = decoy_emit

    noisy = max_emit + float(rng.normal(0.0, noise_std))
    return float(np.clip(noisy, 0.0, 1.0))


def observation_likelihood(
    observed_signal: float,
    sensor_pos: Coordinate,
    candidate_cell: Coordinate,
    decay_rate: float,
    noise_std: float,
    decoy_count: int = 1,
) -> float:
    """Compute P(observation | EVE at candidate_cell) for Bayesian updates.

    Uses a robust Gaussian-Cauchy mixture likelihood to account for unknown decoy
    interference and outlier noise spikes without zeroing out valid cells:
        L(s | c) = (1 - alpha) * N(s; mu(x, c), sigma^2) + alpha * U(0, 1)
    where alpha is a small heavy-tail floor proportional to decoy presence.
    """
    if observed_signal is None or math.isnan(observed_signal) or not (0.0 <= observed_signal <= 1.0):
        # Invalid reading provides uniform (non-informative) likelihood
        return 1.0

    mu = expected_signal_from_target(sensor_pos, candidate_cell, decay_rate)
    sigma = max(noise_std, 1e-3)
    diff = observed_signal - mu
    gaussian = math.exp(-0.5 * (diff / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))

    # Heavy-tail floor protects against unmodeled decoy interference far from the true target
    # If observed signal is HIGHER than expected for candidate_cell, a local decoy near sensor_pos
    # could explain the elevated reading even if EVE is actually at candidate_cell.
    decoy_tail = 0.08 if (decoy_count > 0 and observed_signal > mu + sigma) else 0.02
    return float((1.0 - decoy_tail) * gaussian + decoy_tail * 1.0)
